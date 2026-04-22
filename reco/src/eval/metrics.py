"""Standard leave-last-out ranking metrics.

For each user u with true test item i*, sample M negatives that the user
never interacted with (uniform from non-positives). Rank i* against the
negatives using the agent. Report HR@K, NDCG@K, MRR across the user base.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _sample_neg(num_items: int, seen: set, k: int) -> np.ndarray:
    out = np.empty(k, dtype=np.int64)
    filled = 0
    while filled < k:
        cand = np.random.randint(1, num_items, size=k - filled)
        for c in cand:
            if int(c) not in seen:
                out[filled] = c
                filled += 1
                if filled == k:
                    break
    return out


@torch.no_grad()
def evaluate_ranking(agent, enc, splits, hist, sim, profiles,
                     topks: List[int] = (1, 5, 10, 20),
                     num_negatives: int = 100,
                     device: str = "cuda",
                     max_users: int | None = None) -> Dict[str, float]:
    agent.eval()
    test = splits.test
    users = test["user_idx"].values
    items = test["item_idx"].values

    if max_users is not None and len(users) > max_users:
        idx = np.random.choice(len(users), size=max_users, replace=False)
        users, items = users[idx], items[idx]

    num_items = splits.num_items
    max_k = max(topks)

    ranks: List[int] = []
    for u, i_true in zip(users.tolist(), items.tolist()):
        seq_items = [it for (it, _r, _t) in hist.get(u, [])]
        # pad to max_seq_len (left-pad) using the agent's encoder.max_seq_len
        L = enc.max_seq_len
        seq_padded = [0] * max(0, L - len(seq_items)) + seq_items[-L:]
        seq = np.asarray(seq_padded[:L], dtype=np.int64)

        seen = set(seq_items)
        seen.add(i_true)
        negs = _sample_neg(num_items, seen, num_negatives)
        cand = np.concatenate([[i_true], negs])   # col 0 = true
        seq_t = torch.as_tensor(seq[None], device=device)
        cand_t = torch.as_tensor(cand[None], device=device)
        scores = agent.rank(seq_t, cand_t).cpu().numpy().reshape(-1)
        # rank of the true item (lower = better); 1 = top
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    out: Dict[str, float] = {}
    for k in topks:
        hit = (ranks_arr <= k).astype(np.float64)
        ndcg = np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1.0), 0.0)
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(ndcg.mean())
    out["MRR"] = float((1.0 / ranks_arr).mean())
    out["MeanRank"] = float(ranks_arr.mean())
    return out
