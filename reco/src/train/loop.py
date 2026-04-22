"""Unified offline-RL training loop for the ablation.

Steps:
  0. build SimulatorCache, FrozenSimulator
  1. build splits, user_hist, transitions, replay
  2. build encoder (SASRec), wrap in DQN or PPO agent
  3. supervised warmup on next-item prediction (stabilises embeddings)
  4. (UG-MORS only) fit ConformalGate on calibration users
  5. RL training with the chosen reward
  6. eval every epoch (HR@K, NDCG@K, sim-real-gap)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..agents.dqn import DQNAgent, DQNBatch
from ..agents.ppo import PPOAgent, PPOBatch
from ..data.splits import make_splits, user_sequences
from ..data.sequence_dataset import compute_item_popularity
from ..encoders.sasrec import SASRec
from ..eval.metrics import evaluate_ranking
from ..rewards import build_reward
from ..rewards.conformal import ConformalGate
from ..simulator.cache import build_cache
from ..simulator.frozen_sim import FrozenSimulator
from ..utils.seed import set_seed
from .replay import build_transitions, ReplayBuffer

ROOT = Path(__file__).resolve().parents[2]


def _build_user_profiles(sim: FrozenSimulator, hist: Dict[int, list],
                         batch_size: int = 256) -> Dict[int, torch.Tensor]:
    """Pre-compute (pos_vec, neg_vec) for every user. Returns dict user -> (2D,)."""
    profiles = {}
    for u, seq in hist.items():
        items = [it for (it, _r, _t) in seq]
        ratings = [r for (_it, r, _t) in seq]
        p = sim.build_profile(items, ratings)
        profiles[u] = torch.stack([p.pos_vec, p.neg_vec], dim=0)   # (2, D)
    return profiles


def _gather_profiles(profiles: Dict[int, torch.Tensor],
                     users: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack([profiles[int(u)] for u in users.cpu().tolist()])  # (B, 2, D)
    return stacked[:, 0], stacked[:, 1]


def _supervised_warmup(agent, replay: ReplayBuffer, steps: int, device: str,
                       lr: float):
    """Train encoder with BPR loss on (action > negs) so embeddings are sane."""
    enc = agent.online if hasattr(agent, "online") else agent.encoder
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    pbar = tqdm(range(steps), desc="warmup", leave=False)
    for _ in pbar:
        b = replay.sample(256, device=device)
        h = enc.encode(b["seq"])                                  # (B, L, D)
        last = h[:, -1, :]                                        # (B, D)
        cand = b["cand"]
        e = enc.item_emb(cand)                                    # (B, K, D)
        logits = torch.einsum("bd,bkd->bk", last, e)              # (B, K) — col 0 = positive
        # cross-entropy with target=0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        pbar.set_postfix(loss=float(loss))


def _fit_calibration_gate(sim: FrozenSimulator, profiles: Dict[int, torch.Tensor],
                          val_df, test_df, calib_users: np.ndarray,
                          reward_cfg: dict, device: str) -> ConformalGate:
    """For users held out as calibration, compare sim-reward vs real rating on
    their TEST interaction. Build conformal gate from that."""
    mask = val_df["user_idx"].isin(calib_users)
    cal_val = val_df[mask]                                       # use VAL as proxy
    if len(cal_val) == 0:
        mask = test_df["user_idx"].isin(calib_users)
        cal_val = test_df[mask]

    users = cal_val["user_idx"].values
    items = torch.as_tensor(cal_val["item_idx"].values, device=device, dtype=torch.long)
    ratings = cal_val["rating"].values.astype(np.float32)
    # scale real rating to [-1,1]  (1->-1, 5->+1)
    r_real = (ratings - 3.0) / 2.0

    # sim score (tanh-squashed semantic)
    pos = torch.stack([profiles[int(u)][0] for u in users]).to(device)
    neg = torch.stack([profiles[int(u)][1] for u in users]).to(device)
    with torch.no_grad():
        s = sim.semantic_score(pos, neg, items)
        r_sim = torch.tanh(s / reward_cfg["sim_temperature"]).cpu().numpy()

    w = tuple(reward_cfg.get("epistemic_weights", [0.0, 0.5, 0.5]))
    u_epi = (w[0] * sim.u_jml[items].cpu().numpy()
             + w[1] * sim.u_sem[items].cpu().numpy()
             + w[2] * sim.u_nli[items].cpu().numpy())

    gate = ConformalGate(alpha=reward_cfg["crc_alpha"],
                         temperature=reward_cfg["gate_temperature"],
                         confidence_floor=reward_cfg["confidence_floor"])
    res = gate.fit(r_sim, r_real, u_epi)
    print(f"[CRC] n={res.n}  q_hat={res.q_hat:.3f}  u*={res.u_threshold:.3f}  T={gate.T:.4f}")
    return gate


def train(cfg: dict, out_dir: Path) -> dict:
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    # ---- data ----
    ds = cfg["dataset"]["name"]
    proc = ROOT / "processed" / ds / "interactions.parquet"
    splits = make_splits(proc, val_frac=cfg["dataset"]["val_frac"],
                          seed=cfg["seed"])
    hist = user_sequences(splits.train)
    item_pop = compute_item_popularity(splits.train, splits.num_items)
    transitions = build_transitions(hist, cfg["dataset"]["max_seq_len"])
    replay = ReplayBuffer(transitions, splits.num_items, item_pop,
                           neg_k=cfg["eval"]["sample_negatives"] + 1,
                           neg_sampling=cfg["dataset"]["neg_sampling"])

    print(f"[data] users={splits.num_users} items={splits.num_items} "
          f"transitions={len(replay):,} calib_users={len(splits.calib_users)}")

    # ---- simulator + profiles ----
    cache = build_cache(ds, cfg["reward"]["topk_keywords"])
    sim = FrozenSimulator(cache, device=device)
    print(f"[sim] item_emb={cache.item_emb.shape} kw_emb={cache.kw_emb.shape} "
          f"mean_u_llm={float(cache.u_llm.mean()):.3f}")
    profiles = _build_user_profiles(sim, hist)

    # ---- encoder + agent ----
    enc = SASRec(num_items=splits.num_items,
                 hidden_dim=cfg["encoder"]["hidden_dim"],
                 max_seq_len=cfg["dataset"]["max_seq_len"],
                 num_blocks=cfg["encoder"]["num_blocks"],
                 num_heads=cfg["encoder"]["num_heads"],
                 dropout=cfg["encoder"]["dropout"],
                 attn_dropout=cfg["encoder"]["attn_dropout"]).to(device)

    if cfg["agent"]["name"] == "dqn":
        agent = DQNAgent(enc, splits.num_items,
                          gamma=cfg["agent"]["gamma"],
                          tau=cfg["agent"]["tau"]).to(device)
        # warmup still trains the encoder end-to-end
        opt = torch.optim.Adam(agent.online.parameters(),
                                lr=cfg["agent"].get("warmup_lr", cfg["agent"]["lr"]))
    elif cfg["agent"]["name"] == "ppo":
        agent = PPOAgent(enc, clip=cfg["agent"]["ppo_clip"],
                          value_coef=cfg["agent"]["value_coef"],
                          entropy_coef=cfg["agent"]["entropy_coef"]).to(device)
        opt = torch.optim.Adam(agent.parameters(), lr=cfg["agent"]["lr"])
    else:
        raise ValueError(f"unknown agent {cfg['agent']['name']}")

    # ---- warmup ----
    warm_steps = cfg["agent"]["warmup_epochs"] * cfg["agent"]["steps_per_epoch"]
    if warm_steps > 0:
        warm_lr = cfg["agent"].get("warmup_lr", cfg["agent"]["lr"])
        _supervised_warmup(agent, replay, warm_steps, device, warm_lr)

    # ---- switch to RL: optionally freeze encoder, rebuild optimizer ----
    freeze = bool(cfg["agent"].get("freeze_encoder", False))
    if cfg["agent"]["name"] == "dqn":
        agent.set_freeze_encoder(freeze)
        opt = torch.optim.Adam(agent.rl_parameters(), lr=cfg["agent"]["lr"])
        print(f"[rl] freeze_encoder={freeze} "
              f"trainable_params={sum(p.numel() for p in agent.rl_parameters()):,}")

    # ---- reward ----
    reward = build_reward(cfg["reward"]["name"], cfg["reward"])
    if cfg["reward"]["name"] == "ug_mors":
        gate = _fit_calibration_gate(sim, profiles, splits.val, splits.test,
                                      splits.calib_users, cfg["reward"], device)
        reward.set_calibration(gate)

    # ---- eval harness ----
    def do_eval() -> dict:
        return evaluate_ranking(agent=agent, enc=enc,
                                 splits=splits, hist=hist,
                                 sim=sim, profiles=profiles,
                                 topks=cfg["eval"]["topks"],
                                 num_negatives=cfg["eval"]["sample_negatives"],
                                 device=device)

    # ---- RL training ----
    history: List[dict] = []
    best_ndcg = -1.0
    for ep in range(1, cfg["agent"]["epochs"] + 1):
        t0 = time.time()
        agent.train()
        stats = {"loss": 0.0, "r_mean": 0.0, "u_epi": 0.0, "gate": 0.0, "n": 0}
        for _ in range(cfg["agent"]["steps_per_epoch"]):
            b = replay.sample(cfg["agent"]["batch_size"], device=device)
            up, un = _gather_profiles(profiles, b["user"])
            r, diag = reward.compute({
                "sim": sim,
                "user_pos": up,
                "user_neg": un,
                "item_idx": b["action"],
            })

            if cfg["agent"]["name"] == "dqn":
                batch = DQNBatch(seq=b["seq"], action=b["action"], reward=r,
                                  next_seq=b["next_seq"], done=b["done"],
                                  cand=b["cand"])
                loss, aux = agent.td_loss(batch)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.rl_parameters()),
                                                cfg["agent"]["clip_grad"])
                opt.step()
                agent.soft_update()
            else:   # ppo
                ppo_b = PPOBatch(seq=b["seq"], cand=b["cand"], reward=r)
                with torch.no_grad():
                    old_logp = agent.collect_old_logprob(ppo_b)
                for _ in range(cfg["agent"]["ppo_epochs"]):
                    loss, aux = agent.loss(ppo_b, old_logp)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                    cfg["agent"]["clip_grad"])
                    opt.step()

            stats["loss"] += float(loss); stats["r_mean"] += float(r.mean())
            if "u_epi" in diag: stats["u_epi"] += float(diag["u_epi"].mean())
            if "gate" in diag: stats["gate"] += float(diag["gate"].mean())
            stats["n"] += 1

        for k in ("loss", "r_mean", "u_epi", "gate"):
            stats[k] = stats[k] / max(stats["n"], 1)
        metrics = do_eval()
        elapsed = time.time() - t0
        row = {"epoch": ep, "time_s": round(elapsed, 1), **stats, **metrics}
        history.append(row)
        msg = f"[ep {ep:3d}] loss={stats['loss']:.3f} r={stats['r_mean']:+.3f} " \
              f"HR@10={metrics['HR@10']:.4f} NDCG@10={metrics['NDCG@10']:.4f}"
        if "gate" in stats and stats["gate"] > 0:
            msg += f" gate={stats['gate']:.2f}"
        print(msg)

        if metrics["NDCG@10"] > best_ndcg:
            best_ndcg = metrics["NDCG@10"]
            torch.save({"agent": agent.state_dict(), "cfg": cfg},
                       out_dir / "best.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # final sim-real-gap eval
    from ..eval.sim_real_gap import sim_real_gap
    gap = sim_real_gap(sim=sim, profiles=profiles, splits=splits,
                        reward_cfg=cfg["reward"], device=device)
    with open(out_dir / "sim_real_gap.json", "w") as f:
        json.dump(gap, f, indent=2)
    print(f"[sim-real-gap] {json.dumps(gap, indent=2)}")

    return {"history": history, "best_ndcg": best_ndcg, "gap": gap}
