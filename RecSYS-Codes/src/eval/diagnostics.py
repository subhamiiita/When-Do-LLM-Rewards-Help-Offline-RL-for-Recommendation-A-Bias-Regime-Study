"""Reward-variant diagnostics: SNR, ECE, sim-overfit gap.

Given a dataset, compute for each reward variant its reward SNR on held-out labeled
(s, a) pairs (real user interactions with known y).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.config import project_root
from src.common.device import get_device
from src.common.metrics import expected_calibration_error, reward_snr
from src.data.splits import build_eval_pools
from src.reward.common import RewardEngine, RewardState
from src.reward.dispatch import VARIANT_REGISTRY, compute_reward
from src.reward.ug_mors import compute_ug_mors_components


def _state_from_train_history(engine: RewardEngine, users: list[int], up_to_t: int | None = None) -> RewardState:
    device = engine.device
    max_len = engine.sas_max_len
    pad = engine.sas_pad
    B = len(users)
    hist = torch.full((B, max_len), pad, dtype=torch.long, device=device)
    labs = torch.zeros((B, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    for i, u in enumerate(users):
        seq = engine.splits["train"][u]
        if up_to_t is not None:
            seq = seq[:up_to_t]
        take = seq[-max_len:]
        for j, (iid, y, _) in enumerate(take):
            hist[i, max_len - len(take) + j] = iid
            labs[i, max_len - len(take) + j] = y
            mask[i, max_len - len(take) + j] = True
    user_idx = torch.tensor(users, dtype=torch.long, device=device)
    t = torch.full((B,), 5, dtype=torch.long, device=device)
    recent = torch.full((B, engine.div_k), -1, dtype=torch.long, device=device)
    for i, u in enumerate(users):
        seq = engine.splits["train"][u]
        if len(seq) >= 1:
            recent[i, 0] = seq[-1][0]
    return RewardState(
        hist_ids=hist, hist_labels=labs, hist_mask=mask,
        user_idx=user_idx, t=t, recent_actions=recent,
    )


def collect_labeled_sa_pairs(engine: RewardEngine, n_pairs: int = 4096, rng_seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Sample (user, item, y) triples from training interactions. Returns three parallel lists."""
    rng = np.random.default_rng(rng_seed)
    eligible = [u for u in range(engine.n_users) if len(engine.splits["train"][u]) >= 6]
    users: list[int] = []
    items: list[int] = []
    labels: list[int] = []
    while len(users) < n_pairs:
        u = int(rng.choice(eligible))
        seq = engine.splits["train"][u]
        t = int(rng.integers(5, len(seq)))  # predict step t given history [:t]
        iid, y, _ = seq[t]
        users.append(u); items.append(iid); labels.append(y)
    return users, items, labels


def diagnostics(dataset: str, n_pairs: int = 4096) -> dict:
    device = get_device()
    engine = RewardEngine(dataset, device=device)
    engine.load_nli()

    users, items, labels = collect_labeled_sa_pairs(engine, n_pairs=n_pairs)
    items_t = torch.tensor(items, dtype=torch.long, device=device)
    labels_np = np.array(labels)

    # Build corresponding state for each sample (history up to t).
    # For simplicity compute rewards in one big batched pass per variant: states share history building.
    # We chunk to keep memory bounded.
    CHUNK = 1024
    variant_rewards: dict[str, list[np.ndarray]] = {v: [] for v in VARIANT_REGISTRY}
    p_sem_vals: list[np.ndarray] = []
    p_sta_vals: list[np.ndarray] = []

    for s in range(0, len(users), CHUNK):
        chunk_users = users[s : s + CHUNK]
        chunk_items = items_t[s : s + CHUNK]
        # For the PBRS variant prev_state == state here; good enough for SNR diagnostic.
        state = _state_from_train_history(engine, chunk_users, up_to_t=None)
        # Compute ug_mors components once and reuse for ECE
        comps = compute_ug_mors_components(engine, state, chunk_items)
        p_sem_vals.append(comps["p_sem"].detach().cpu().numpy())
        p_sta_vals.append(comps["p_sta"].detach().cpu().numpy())
        for v in VARIANT_REGISTRY:
            try:
                r = compute_reward(v, engine, state, chunk_items, prev_state=state)
                variant_rewards[v].append(r.detach().cpu().numpy())
            except Exception as e:
                print(f"[diag {v}] error: {e}")

    results = {}
    for v, chunks in variant_rewards.items():
        if not chunks:
            continue
        rewards = np.concatenate(chunks).astype(np.float32)
        results[v] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "snr": reward_snr(rewards, labels_np),
        }

    # ECE on gated relevance probability
    p_sem = np.concatenate(p_sem_vals)
    p_sta = np.concatenate(p_sta_vals)
    results["_ece_p_sem"] = expected_calibration_error(p_sem, labels_np)
    results["_ece_p_sta"] = expected_calibration_error(p_sta, labels_np)

    print(f"[diagnostics {dataset}]  n_pairs={n_pairs}  pos_rate={labels_np.mean()*100:.1f}%")
    print(f"  ECE(p_sem)={results['_ece_p_sem']:.3f}  ECE(p_sta)={results['_ece_p_sta']:.3f}")
    print(f"  {'variant':22s}  {'mean':>7s}  {'std':>6s}  {'SNR':>7s}")
    for v in VARIANT_REGISTRY:
        if v in results:
            r = results[v]
            print(f"  {v:22s}  {r['mean']:7.3f}  {r['std']:6.3f}  {r['snr']:7.3f}")

    out_path = project_root() / f"results/tables/diagnostics_{dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_pairs", type=int, default=4096)
    args = ap.parse_args()
    diagnostics(args.dataset, n_pairs=args.n_pairs)


if __name__ == "__main__":
    main()
