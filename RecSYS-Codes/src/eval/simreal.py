"""SimReal evaluation: rank real held-out interactions under a trained policy.

For each user with a val/test positive interaction, construct a candidate pool
(1 positive + 99 sampled negatives) and score via the trained policy's actor
or Q-head. Compute NDCG@10 and HR@10.

This is the ONLY meaningful success signal — training reward is not comparable
across UG-MORS variants (different scales).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.common.metrics import ndcg_at_k, hit_at_k
from src.data.splits import build_eval_pools
from src.reward.common import RewardEngine


def build_state_for_users(engine: RewardEngine, users: list[int], use_val: bool = False) -> torch.Tensor:
    """Build state tensor for each user from their full TRAIN history (right-aligned, pad left).
    If use_val=True, include the val interaction as the last history entry so we evaluate on test."""
    max_len = engine.sas_max_len
    pad = engine.sas_pad
    device = engine.device
    B = len(users)
    seq = torch.full((B, max_len), pad, dtype=torch.long, device=device)
    for i, u in enumerate(users):
        hist = [it for (it, _, _) in engine.splits["train"][u]]
        if use_val and engine.splits["val"][u] is not None:
            hist.append(engine.splits["val"][u][0])
        take = hist[-max_len:]
        if take:
            seq[i, max_len - len(take):] = torch.tensor(take, dtype=torch.long, device=device)
    with torch.no_grad():
        return engine.sasrec.last_hidden(seq)


def load_agent_head(agent_name: str, ckpt_path: Path, d_state: int, n_items: int, device: torch.device):
    """Reconstruct the head from saved weights for scoring."""
    from src.agents.common import HeadConfig, ItemDotActorCriticHead, ItemDotQHead

    head_cfg = HeadConfig(d_state=d_state, n_items=n_items)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if agent_name == "dqn":
        head = ItemDotQHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["q"])
        def score_fn(s, cand):
            q = head(s)
            return q.gather(1, cand)
    else:
        head = ItemDotActorCriticHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["net"])
        def score_fn(s, cand):
            logits, _ = head(s)
            return logits.gather(1, cand)
    return score_fn


def evaluate_run(
    dataset: str,
    agent_name: str,
    variant: str,
    seed: int,
    split: str = "test",
    n_negs: int = 99,
    batch: int = 256,
) -> dict:
    cfg = dataset_config(dataset)
    device = get_device()
    engine = RewardEngine(dataset, device=device)

    run_dir = project_root() / f"results/runs/{dataset}/{agent_name}/{variant}/seed_{seed}"
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model at {ckpt}")

    d_state = engine.sasrec.cfg.d_model
    n_items = engine.n_items
    score_fn = load_agent_head(agent_name, ckpt, d_state=d_state, n_items=n_items, device=device)

    # Build eval pools (users × 100 candidates)
    pool = build_eval_pools(engine.splits, n_negs=n_negs, seed=42, which=split).to(device)
    # Users with valid pool rows (col 0 != -1)
    valid_users = [u for u in range(engine.n_users) if int(pool[u, 0].item()) != -1]

    # Limit to only users whose target is positive (per leave-last-out) — include all for NDCG
    ndcg_sum, hit_sum, n_eval = 0.0, 0.0, 0
    i = 0
    while i < len(valid_users):
        users = valid_users[i : i + batch]
        state = build_state_for_users(engine, users, use_val=(split == "test"))
        cand = pool[torch.tensor(users, device=device)]          # (B, 100)
        scores = score_fn(state, cand)                           # (B, 100)
        # Target is col 0
        target_idx = torch.zeros(len(users), dtype=torch.long, device=device)
        ndcg = ndcg_at_k(scores, target_idx, k=10)
        hit = hit_at_k(scores, target_idx, k=10)
        ndcg_sum += float(ndcg.sum().item())
        hit_sum += float(hit.sum().item())
        n_eval += len(users)
        i += batch

    ndcg10 = ndcg_sum / max(1, n_eval)
    hr10 = hit_sum / max(1, n_eval)
    out = dict(
        dataset=dataset,
        agent=agent_name,
        variant=variant,
        seed=seed,
        split=split,
        n_eval=n_eval,
        ndcg_at_10=ndcg10,
        hr_at_10=hr10,
    )
    print(f"[eval {dataset}/{agent_name}/{variant}/seed{seed}/{split}]  NDCG@10={ndcg10:.4f}  HR@10={hr10:.4f}  (n={n_eval})")
    # Save next to training run
    with (run_dir / f"simreal_{split}.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    evaluate_run(args.dataset, args.agent, args.variant, args.seed, split=args.split)


if __name__ == "__main__":
    main()
