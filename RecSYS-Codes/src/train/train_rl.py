"""Unified RL training driver.

CLI:
    python -m src.train.train_rl \
        --dataset ml1m --agent dqn --variant ug_mors --seed 0 --steps 50000

Writes per-run outputs to:
    results/runs/<dataset>/<agent>/<variant>/seed_<k>/{metrics.jsonl, model.pt, simreal.json}

Supports all combinations of:
    --agent    : dqn, ppo, a2c, trpo
    --variant  : baseline_vote, naive_continuous, ug_mors, ug_mors_fixed,
                 ug_mors_no_div, ug_mors_no_per, ug_mors_no_ret, ug_pbrs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.common.seed import set_seed
from src.env.rec_env import EnvConfig, RecEnv
from src.reward.common import RewardEngine


AGENTS = ("dqn", "ppo", "a2c", "trpo")


# ----------------------------------------------------------------------
# Agent factory
# ----------------------------------------------------------------------

def make_agent(agent: str, d_state: int, n_items: int, cfg: dict, device: torch.device, item_emb_init=None):
    rl = cfg["rl"]
    hidden = 256
    if agent == "dqn":
        from src.agents.dqn import DQNAgent, DQNConfig
        return DQNAgent(
            DQNConfig(
                d_state=d_state, n_items=n_items, hidden=hidden,
                lr=rl["lr"], gamma=rl["gamma"],
                replay_size=rl["dqn"]["replay"],
                batch=rl["dqn"]["batch"],
                target_update=rl["dqn"]["target_update"],
                eps_start=rl["dqn"]["eps_start"],
                eps_end=rl["dqn"]["eps_end"],
                eps_decay_steps=rl["dqn"]["eps_decay_steps"],
            ),
            device=device,
            item_emb_init=item_emb_init,
        )
    if agent == "ppo":
        from src.agents.ppo import PPOAgent, PPOConfig
        return PPOAgent(
            PPOConfig(
                d_state=d_state, n_items=n_items, hidden=hidden,
                lr=rl["lr"], gamma=rl["gamma"],
                gae_lambda=rl["ppo"]["gae_lambda"],
                clip=rl["ppo"]["clip"],
                epochs=rl["ppo"]["epochs"],
                batch=rl["ppo"]["batch"],
                rollout=rl["ppo"]["rollout"],
                entropy=rl["ppo"]["entropy"],
            ),
            device=device,
            item_emb_init=item_emb_init,
        )
    if agent == "a2c":
        from src.agents.a2c import A2CAgent, A2CConfig
        return A2CAgent(
            A2CConfig(
                d_state=d_state, n_items=n_items, hidden=hidden,
                lr=rl["lr"], gamma=rl["gamma"],
                rollout=rl["a2c"]["rollout"],
                entropy=rl["a2c"]["entropy"],
            ),
            device=device,
            item_emb_init=item_emb_init,
        )
    if agent == "trpo":
        from src.agents.trpo import TRPOAgent, TRPOConfig
        return TRPOAgent(
            TRPOConfig(
                d_state=d_state, n_items=n_items, hidden=hidden,
                critic_lr=rl["lr"], gamma=rl["gamma"],
                max_kl=rl["trpo"]["max_kl"],
                cg_iters=rl["trpo"]["cg_iters"],
                damping=rl["trpo"]["damping"],
                rollout=rl["trpo"]["rollout"],
                entropy=0.01,
            ),
            device=device,
            item_emb_init=item_emb_init,
        )
    raise ValueError(f"Unknown agent: {agent}")


# ----------------------------------------------------------------------
# GAE (for PPO/A2C/TRPO)
# ----------------------------------------------------------------------

def gae(rewards, values, dones, gamma, lam, last_value):
    """rewards/values/dones shape (T, N). Returns advantages (T, N) and returns (T, N)."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    gae_val = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae_val = delta + gamma * lam * nonterminal * gae_val
        adv[t] = gae_val
    ret = adv + values
    return adv, ret


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

def train(
    dataset: str,
    agent_name: str,
    variant: str,
    seed: int,
    steps: int,
    out_dir: Path,
    log_every: int = 2000,
) -> dict:
    cfg = dataset_config(dataset)
    device = get_device()
    set_seed(seed)
    engine = RewardEngine(dataset, device=device)
    engine.load_nli()

    env_cfg = EnvConfig(
        n_envs=cfg["env"]["n_envs"],
        horizon=cfg["env"]["horizon"],
        seed=seed,
        div_k=cfg["reward"]["div_window_k"],
    )
    env = RecEnv(engine, variant=variant, cfg=env_cfg)
    d_state = engine.sasrec.cfg.d_model
    # Initialize agent's item embeddings from SASRec so it starts near SASRec's ranking quality.
    # SASRec has n_items + 1 rows (last = pad); we take the first n_items.
    sasrec_item_emb = engine.sasrec.item_emb.weight.detach().clone()
    agent = make_agent(agent_name, d_state=d_state, n_items=engine.n_items, cfg=cfg, device=device, item_emb_init=sasrec_item_emb)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    mf = metrics_path.open("w", encoding="utf-8")

    t0 = time.time()
    total_env_steps = 0
    ep_returns = []
    cur_return = torch.zeros(env_cfg.n_envs, device=device)

    state = env.state_repr()

    # --- Rollout buffer shapes differ per agent
    if agent_name == "dqn":
        # Step loop: collect transitions, add to replay, update every step
        while total_env_steps < steps:
            valid = env.valid_action_mask()
            a = agent.act(state, valid_mask=valid)
            r, done, next_state = env.step(a)
            agent.replay.add_batch(state, a, r, next_state, done)
            info = agent.update()
            cur_return += r
            total_env_steps += env_cfg.n_envs
            if done.any():
                mask_idx = torch.nonzero(done, as_tuple=False).flatten().tolist()
                for i in mask_idx:
                    ep_returns.append(float(cur_return[i].item()))
                    cur_return[i] = 0.0
            state = next_state
            if total_env_steps % log_every < env_cfg.n_envs:
                msg = {
                    "step": total_env_steps,
                    "mean_ep_return": float(np.mean(ep_returns[-50:]) if ep_returns else 0.0),
                    "reward_mean": float(r.mean().item()),
                    "reward_std": float(r.std().item()),
                    "loss": info.get("loss"),
                    "eps": info.get("eps"),
                    "time": time.time() - t0,
                }
                mf.write(json.dumps(msg) + "\n"); mf.flush()

    else:
        # On-policy agents: collect rollouts of length T then update
        rollout_T = cfg["rl"][agent_name]["rollout"]
        # For A2C, rollout small; for PPO/TRPO rollout large
        T = rollout_T if agent_name != "a2c" else max(8, rollout_T // env_cfg.n_envs)
        dim = d_state

        while total_env_steps < steps:
            buf_s = torch.zeros((T, env_cfg.n_envs, dim), dtype=torch.float32, device=device)
            buf_a = torch.zeros((T, env_cfg.n_envs), dtype=torch.long, device=device)
            buf_lp = torch.zeros((T, env_cfg.n_envs), dtype=torch.float32, device=device)
            buf_r = torch.zeros((T, env_cfg.n_envs), dtype=torch.float32, device=device)
            buf_v = torch.zeros((T, env_cfg.n_envs), dtype=torch.float32, device=device)
            buf_d = torch.zeros((T, env_cfg.n_envs), dtype=torch.float32, device=device)

            for t in range(T):
                valid = env.valid_action_mask()
                a, lp, v = agent.act(state, valid_mask=valid)
                r, done, next_state = env.step(a)
                buf_s[t] = state; buf_a[t] = a; buf_lp[t] = lp; buf_v[t] = v; buf_r[t] = r; buf_d[t] = done.float()
                cur_return += r
                total_env_steps += env_cfg.n_envs
                if done.any():
                    mask_idx = torch.nonzero(done, as_tuple=False).flatten().tolist()
                    for i in mask_idx:
                        ep_returns.append(float(cur_return[i].item()))
                        cur_return[i] = 0.0
                state = next_state
            with torch.no_grad():
                last_v = agent.value(state)
            lam = cfg["rl"].get("ppo", {}).get("gae_lambda", 0.95)
            gamma = cfg["rl"]["gamma"]
            adv, ret = gae(buf_r, buf_v, buf_d, gamma=gamma, lam=lam, last_value=last_v)
            # flatten
            s_flat = buf_s.reshape(-1, dim)
            a_flat = buf_a.reshape(-1)
            lp_flat = buf_lp.reshape(-1)
            adv_flat = adv.reshape(-1)
            ret_flat = ret.reshape(-1)
            info = agent.update({"s": s_flat, "a": a_flat, "lp_old": lp_flat, "adv": adv_flat, "ret": ret_flat})
            if total_env_steps // log_every > (total_env_steps - T * env_cfg.n_envs) // log_every:
                msg = {
                    "step": total_env_steps,
                    "mean_ep_return": float(np.mean(ep_returns[-50:]) if ep_returns else 0.0),
                    "reward_mean": float(buf_r.mean().item()),
                    "reward_std": float(buf_r.std().item()),
                    **{k: float(v) for k, v in info.items()},
                    "time": time.time() - t0,
                }
                mf.write(json.dumps(msg) + "\n"); mf.flush()

    mf.close()

    # Save final model
    model_path = out_dir / "model.pt"
    if hasattr(agent, "q"):
        torch.save({"q": agent.q.state_dict()}, model_path)
    else:
        torch.save({"net": agent.net.state_dict()}, model_path)

    # Print summary
    final_return = float(np.mean(ep_returns[-50:]) if ep_returns else 0.0)
    dt = time.time() - t0
    print(f"[train {dataset}/{agent_name}/{variant}/seed{seed}] done {steps} steps in {dt:.0f}s  final_ep_return={final_return:.3f}")
    return dict(final_return=final_return, runtime_s=dt, total_steps=total_env_steps, out_dir=str(out_dir))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--agent", required=True, choices=AGENTS)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50_000)
    args = ap.parse_args()

    out = project_root() / f"results/runs/{args.dataset}/{args.agent}/{args.variant}/seed_{args.seed}"
    train(args.dataset, args.agent, args.variant, args.seed, args.steps, out)


if __name__ == "__main__":
    main()
