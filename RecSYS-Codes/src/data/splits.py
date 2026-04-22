"""Evaluation splits: neg-sampling utilities."""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def sample_eval_negatives(
    n_items: int,
    seen_items: set[int],
    n_negs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `n_negs` unseen item indices uniformly."""
    negs: List[int] = []
    attempts = 0
    while len(negs) < n_negs and attempts < n_negs * 20:
        cand = int(rng.integers(0, n_items))
        if cand not in seen_items:
            negs.append(cand)
        attempts += 1
    # Fallback: fill with random (may duplicate)
    while len(negs) < n_negs:
        negs.append(int(rng.integers(0, n_items)))
    return np.array(negs, dtype=np.int64)


def build_eval_pools(
    splits: dict,
    n_negs: int = 99,
    seed: int = 42,
    which: str = "test",
) -> torch.Tensor:
    """Return (n_users, 1 + n_negs) tensor: col 0 = positive, rest = negatives.
    Users with no target in `which` get -1 rows (skip during eval).
    """
    n_users = splits["n_users"]
    n_items = splits["n_items"]
    target = splits[which]
    rng = np.random.default_rng(seed)
    pool = -np.ones((n_users, 1 + n_negs), dtype=np.int64)
    for u in range(n_users):
        if target[u] is None:
            continue
        # target is (iid, y, ts). Consider only positive target for HR/NDCG.
        iid, y, _ = target[u]
        if y != 1:
            # still evaluate but with positive=iid flagged as "liked"? follow SASRec convention: use any last-item.
            pass
        seen = {it for (it, _, _) in splits["sequences"][u]}
        negs = sample_eval_negatives(n_items, seen, n_negs, rng)
        pool[u, 0] = iid
        pool[u, 1:] = negs
    return torch.from_numpy(pool)
