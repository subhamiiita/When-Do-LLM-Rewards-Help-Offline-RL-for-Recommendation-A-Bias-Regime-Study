"""Quick smoke test for reward modules. Run:

    python -m src.reward._smoke --dataset ml1m
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.common.device import get_device
from src.data.splits import build_eval_pools
from src.reward.common import RewardEngine, RewardState
from src.reward.dispatch import VARIANT_REGISTRY, compute_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = get_device()
    engine = RewardEngine(args.dataset, device=device)
    engine.load_nli()

    splits = engine.splits
    rng = np.random.default_rng(0)

    # Build synthetic state: take the first `batch` users with ≥20 train items;
    # their state is the last 50 train items (right-aligned, pad left).
    users: list[int] = []
    for u in range(splits["n_users"]):
        if len(splits["train"][u]) >= 20:
            users.append(u)
        if len(users) >= args.batch:
            break

    max_len = engine.sas_max_len
    pad = engine.sas_pad
    B = len(users)
    hist = np.full((B, max_len), pad, dtype=np.int64)
    lab = np.full((B, max_len), 0, dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=bool)
    for i, u in enumerate(users):
        seq = splits["train"][u][-max_len:]
        take = len(seq)
        for j, (iid, y, _) in enumerate(seq):
            hist[i, max_len - take + j] = iid
            lab[i, max_len - take + j] = y
            mask[i, max_len - take + j] = True

    hist_t = torch.from_numpy(hist).to(device)
    lab_t = torch.from_numpy(lab).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    user_idx = torch.tensor(users, dtype=torch.long, device=device)
    t = torch.tensor([5] * B, dtype=torch.long, device=device)  # fake step
    recent = torch.full((B, engine.div_k), -1, dtype=torch.long, device=device)
    # Seed recent with the last action of each user
    for i, u in enumerate(users):
        last = splits["train"][u][-1][0]
        recent[i, 0] = last

    state = RewardState(hist_ids=hist_t, hist_labels=lab_t, hist_mask=mask_t, user_idx=user_idx, t=t, recent_actions=recent)

    # ---- Action sampling: half known positives, half random catalog items (mixed).
    actions = np.zeros(B, dtype=np.int64)
    for i, u in enumerate(users):
        if i % 2 == 0:
            # positive: a liked item from train
            liked = [it for (it, y, _) in splits["train"][u] if y == 1]
            actions[i] = liked[-1] if liked else splits["train"][u][-1][0]
        else:
            actions[i] = int(rng.integers(0, splits["n_items"]))
    action_t = torch.from_numpy(actions).to(device)

    print(f"[smoke {args.dataset}] batch={B} users, half positive-like actions, half random")
    print(f"{'variant':25s}  {'mean':>7s}  {'std':>6s}  {'min':>6s}  {'max':>6s}")
    print("-" * 58)
    for v in VARIANT_REGISTRY:
        r = compute_reward(v, engine, state, action_t, prev_state=state, gamma=0.9)
        r_np = r.detach().cpu().numpy()
        print(f"{v:25s}  {r_np.mean():7.3f}  {r_np.std():6.3f}  {r_np.min():6.3f}  {r_np.max():6.3f}")


if __name__ == "__main__":
    main()
