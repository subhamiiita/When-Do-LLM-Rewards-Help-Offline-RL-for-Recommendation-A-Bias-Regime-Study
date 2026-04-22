"""Simulator-internal metrics: A. Rwd, T. Rwd, Liking% — the three metrics the
base paper (Zhang et al., AAAI 2025, Table 2) reports.

Differs from SimReal (src/eval/simreal.py):
  * SimReal:           rank real held-out interactions → NDCG@10 / HR@10.
  * simulator_metrics: run the trained policy INSIDE the simulator for K steps
                       per user, tally the baseline-vote outcome (0/1) per step,
                       report average/total reward + fraction of vote=1.

Why the distinction matters: the base paper uses simulator-internal metrics only;
the new-idea paper (UG-MORS) introduces SimReal as a stricter test. Running both
lets us (a) compare to the base paper in its own terms, and (b) show that UG-MORS
also wins on the paper-internal metric.

IMPORTANT: Liking% and rewards are computed using the baseline-vote label (f_mat
+ f_sim + f_sta >= 2) regardless of the training reward variant. This is because
Liking% is a fixed simulator-output metric — "what the simulator thinks the user
would like", not the shaped reward the agent was trained on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.env.rec_env import EnvConfig, RecEnv
from src.eval.simreal import load_agent_head
from src.reward.baseline_vote import compute_baseline_vote
from src.reward.common import RewardEngine, RewardState


def evaluate_simulator_metrics(
    dataset: str,
    agent_name: str,
    variant: str,
    seed: int,
    horizon: int = 10,
    n_eval_episodes: int = 200,
    n_envs: int = 32,
) -> dict:
    """Run trained policy in simulator for `n_eval_episodes` distinct user-trajectories
    of `horizon` steps each. Tally T. Rwd, A. Rwd, Liking%.

    Liking% is defined as (fraction of env steps where the baseline vote was 1).
    This matches the base paper: 'Liking% is the liking items ratio in the top-10
    recommendations.' With horizon=10, each episode = 10 recommendations.
    """
    cfg = dataset_config(dataset)
    device = get_device()
    engine = RewardEngine(dataset, device=device)
    # Build env. Variant doesn't matter here (we compute baseline vote manually).
    env_cfg = EnvConfig(
        n_envs=n_envs,
        horizon=horizon,
        seed=seed + 1000,  # different from training seed to avoid identical rollouts
        div_k=cfg["reward"]["div_window_k"],
    )
    env = RecEnv(engine, variant="baseline_vote", cfg=env_cfg)

    d_state = engine.sasrec.cfg.d_model
    run_dir = project_root() / f"results/runs/{dataset}/{agent_name}/{variant}/seed_{seed}"
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model at {ckpt}")
    score_fn = load_agent_head(agent_name, ckpt, d_state=d_state, n_items=engine.n_items, device=device)

    total_reward = 0.0
    total_likes = 0
    n_steps = 0
    n_episodes = 0

    while n_episodes < n_eval_episodes:
        state = env.state_repr()
        valid = env.valid_action_mask()
        with torch.no_grad():
            # Use score_fn for all-item scoring. Our score_fn is gather-based, so compute full Q/logits once.
            # Easier: call the loaded head's .net(state) directly.
            scores = _score_all_items(agent_name, run_dir, d_state, engine.n_items, device, state)
        scores = scores.masked_fill(~valid, float("-inf"))
        a = scores.argmax(dim=1)                                         # greedy per env

        # Compute BASELINE VOTE for this action (not the training reward).
        current_state = RewardState(
            hist_ids=env.hist_ids, hist_labels=env.hist_labels, hist_mask=env.hist_mask,
            user_idx=env.user_idx, t=env.t, recent_actions=env.recent_actions,
        )
        bv = compute_baseline_vote(engine, current_state, a)
        vote_np = bv["vote"].detach().cpu().numpy()
        total_reward += float(vote_np.sum())
        total_likes += int(vote_np.sum())
        n_steps += n_envs

        _, done, _ = env.step(a)
        n_episodes += int(done.sum().item())

    a_rwd = total_reward / max(1, n_episodes)            # avg reward per completed episode
    t_rwd = total_reward                                  # total reward summed
    liking = total_likes / max(1, n_steps)                # fraction of vote=1 steps

    out = dict(
        dataset=dataset,
        agent=agent_name,
        variant=variant,
        seed=seed,
        horizon=horizon,
        n_eval_episodes=int(n_episodes),
        n_eval_steps=int(n_steps),
        A_Rwd=a_rwd,
        T_Rwd=t_rwd,
        Liking_pct=liking * 100.0,
    )
    print(
        f"[sim {dataset}/{agent_name}/{variant}/seed{seed}]  "
        f"A.Rwd={a_rwd:.3f}  T.Rwd={t_rwd:.1f}  Liking%={liking*100:.2f}  "
        f"(episodes={n_episodes}, steps={n_steps})"
    )
    with (run_dir / "simulator_metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


def _score_all_items(agent_name, run_dir, d_state, n_items, device, state):
    """Return (B, n_items) Q or logits. Mirrors load_agent_head but returns full vector."""
    from src.agents.common import HeadConfig, ItemDotActorCriticHead, ItemDotQHead

    head_cfg = HeadConfig(d_state=d_state, n_items=n_items)
    ck = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    if agent_name == "dqn":
        head = ItemDotQHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["q"])
        return head(state)
    else:
        head = ItemDotActorCriticHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["net"])
        logits, _ = head(state)
        return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=10, help="steps per evaluation episode (paper uses ~10)")
    ap.add_argument("--n_eval_episodes", type=int, default=200)
    args = ap.parse_args()
    evaluate_simulator_metrics(
        args.dataset, args.agent, args.variant, args.seed,
        horizon=args.horizon, n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
