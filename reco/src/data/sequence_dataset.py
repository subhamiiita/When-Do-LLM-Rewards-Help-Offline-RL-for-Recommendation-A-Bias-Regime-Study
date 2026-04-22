"""Sequential training samples for SASRec-style encoders.

For each user history [i1, i2, ..., iN], produce rolling windows of length
L = max_seq_len. Each sample is:
    seq:    (L,) int32  — last L-1 items + current item (pad-left with 0)
    pos:    (L,) int32  — next-item targets (pad-left with 0)
    neg:    (L,) int32  — uniform / popularity negatives
    mask:   (L,) bool   — valid positions

We support two modes:
    mode="seq2seq": full rolling prediction (used for supervised warmup)
    mode="last":    final-step only (used for RL state encoding)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self,
                 user_hist: Dict[int, List[Tuple[int, float, int]]],
                 num_items: int,
                 max_seq_len: int = 50,
                 item_pop: np.ndarray | None = None,
                 neg_sampling: str = "popularity",
                 mode: str = "seq2seq"):
        self.users = sorted(user_hist.keys())
        self.user_hist = user_hist
        self.num_items = num_items          # includes PAD=0
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.neg_sampling = neg_sampling
        if item_pop is None:
            item_pop = np.ones(num_items, dtype=np.float64)
        item_pop = item_pop.astype(np.float64).copy()
        item_pop[0] = 0.0
        self.item_pop_probs = item_pop / max(item_pop.sum(), 1.0)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_neg(self, exclude: set) -> int:
        # simple rejection sampling
        for _ in range(10):
            if self.neg_sampling == "uniform":
                j = int(np.random.randint(1, self.num_items))
            else:
                j = int(np.random.choice(self.num_items, p=self.item_pop_probs))
            if j not in exclude:
                return j
        # fallback: return any non-pad
        return max(1, int(np.random.randint(1, self.num_items)))

    def __getitem__(self, idx: int):
        u = self.users[idx]
        hist = self.user_hist[u]
        items = [it for (it, _r, _t) in hist][-self.max_seq_len:]
        L = self.max_seq_len

        # left-pad
        seq = np.zeros(L, dtype=np.int64)
        pos = np.zeros(L, dtype=np.int64)
        neg = np.zeros(L, dtype=np.int64)
        mask = np.zeros(L, dtype=np.bool_)

        if len(items) < 2:
            # degenerate: just return zeros (dataset filter handles this but defensive)
            return {"user": u, "seq": seq, "pos": pos, "neg": neg, "mask": mask}

        # input seq is items[:-1], target seq is items[1:]
        inp = items[:-1]
        tgt = items[1:]
        n = min(len(inp), L)
        seq[-n:] = inp[-n:]
        pos[-n:] = tgt[-n:]
        mask[-n:] = True

        exclude = set(items)
        for j in range(L):
            if mask[j]:
                neg[j] = self._sample_neg(exclude)

        return {"user": u, "seq": seq, "pos": pos, "neg": neg, "mask": mask}


def compute_item_popularity(train_df, num_items: int) -> np.ndarray:
    pop = np.zeros(num_items, dtype=np.float64)
    vals = train_df["item_idx"].value_counts()
    for item_id, count in vals.items():
        if 0 <= int(item_id) < num_items:
            pop[int(item_id)] = float(count)
    return pop


def collate(batch):
    out = {}
    for k in batch[0].keys():
        if k == "user":
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.as_tensor(np.stack([b[k] for b in batch]))
    return out
