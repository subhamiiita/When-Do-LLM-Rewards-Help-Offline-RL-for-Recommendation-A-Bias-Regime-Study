"""Offline replay buffer built from user interaction sequences.

For each (user, t) we construct a transition:
    state    = hist[:t]   (padded to max_seq_len)
    action   = hist[t]    (next item the user actually consumed)
    next_st  = hist[:t+1]
    done     = (t == len(hist)-1)

Rating is kept alongside to serve as "real" reward (for sim-real-gap eval
and for conformal calibration on calibration users).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    seq:       np.ndarray  # (L,) int64
    action:    int
    rating:    float
    next_seq:  np.ndarray
    done:      float
    user:      int


def build_transitions(user_hist: Dict[int, List[Tuple[int, float, int]]],
                      max_seq_len: int) -> List[Transition]:
    out: List[Transition] = []
    for u, seq in user_hist.items():
        items = [it for (it, _r, _t) in seq]
        ratings = [r for (_it, r, _t) in seq]
        n = len(items)
        if n < 2:
            continue
        # window builder with left-padding
        padded = [0] * max_seq_len + items
        for t in range(1, n):
            state = np.asarray(padded[t:t + max_seq_len], dtype=np.int64)
            next_state = np.asarray(padded[t + 1:t + 1 + max_seq_len], dtype=np.int64)
            out.append(Transition(
                seq=state,
                action=items[t],
                rating=float(ratings[t]),
                next_seq=next_state,
                done=float(t == n - 1),
                user=int(u),
            ))
    return out


class ReplayBuffer:
    def __init__(self, transitions: List[Transition], num_items: int,
                 item_pop: np.ndarray, neg_k: int = 127,
                 neg_sampling: str = "popularity"):
        self.tr = transitions
        self.num_items = num_items
        self.neg_k = neg_k
        self.mode = neg_sampling

        pop = item_pop.astype(np.float64).copy()
        pop[0] = 0.0
        self.pop_probs = pop / max(pop.sum(), 1.0)

        # pre-stack for speed
        self.seq = np.stack([t.seq for t in transitions])           # (N, L)
        self.next_seq = np.stack([t.next_seq for t in transitions]) # (N, L)
        self.action = np.asarray([t.action for t in transitions], dtype=np.int64)
        self.rating = np.asarray([t.rating for t in transitions], dtype=np.float32)
        self.done = np.asarray([t.done for t in transitions], dtype=np.float32)
        self.user = np.asarray([t.user for t in transitions], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.tr)

    def _neg(self, k: int) -> np.ndarray:
        if self.mode == "uniform":
            return np.random.randint(1, self.num_items, size=k)
        return np.random.choice(self.num_items, size=k, p=self.pop_probs)

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        idx = np.random.randint(0, len(self.tr), size=batch_size)
        seq = self.seq[idx]
        next_seq = self.next_seq[idx]
        action = self.action[idx]
        rating = self.rating[idx]
        done = self.done[idx]
        user = self.user[idx]

        # candidate set: true action at col 0 + K negatives
        negs = np.stack([self._neg(self.neg_k) for _ in range(batch_size)])
        cand = np.concatenate([action[:, None], negs], axis=1)   # (B, K+1)

        return {
            "seq":       torch.as_tensor(seq,      device=device),
            "next_seq":  torch.as_tensor(next_seq, device=device),
            "action":    torch.as_tensor(action,   device=device),
            "rating":    torch.as_tensor(rating,   device=device),
            "done":      torch.as_tensor(done,     device=device),
            "user":      torch.as_tensor(user,     device=device),
            "cand":      torch.as_tensor(cand,     device=device),
        }
