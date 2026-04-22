"""Frozen LLM User Simulator.

A user's preference profile is built by aggregating the positive/negative
keywords of the items they have historically interacted with (weighted by
their real ratings, if available). A new item is scored by cosine similarity
between its own pos/neg keyword embeddings and the user profile — which
simulates what an LLM user-simulator would say without running an LLM at
train time.

This is the *raw semantic signal*. The reward functions in src/rewards/
decide how to translate this (and the uncertainty) into a scalar reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .cache import SimulatorCache


@dataclass
class UserProfile:
    pos_vec: torch.Tensor   # (D,) L2-normed
    neg_vec: torch.Tensor   # (D,)
    history: torch.Tensor   # (H,) item_idxs consumed


class FrozenSimulator:
    """Cache-based LLM user simulator — no LLM inference at train time."""

    def __init__(self, cache: SimulatorCache, device: str = "cuda"):
        self.cache = cache
        self.device = device
        self.item_emb = torch.as_tensor(cache.item_emb, device=device)
        self.kw_emb   = torch.as_tensor(cache.kw_emb,   device=device)
        self.pos_ids  = torch.as_tensor(cache.pos_kw_ids.astype(np.int64), device=device)
        self.pos_conf = torch.as_tensor(cache.pos_kw_conf, device=device)
        self.neg_ids  = torch.as_tensor(cache.neg_kw_ids.astype(np.int64), device=device)
        self.neg_conf = torch.as_tensor(cache.neg_kw_conf, device=device)
        self.u_jml = torch.as_tensor(cache.u_jml, device=device)
        self.u_sem = torch.as_tensor(cache.u_sem, device=device)
        self.u_nli = torch.as_tensor(cache.u_nli, device=device)
        self.u_llm = torch.as_tensor(cache.u_llm, device=device)

    # ------- profile building -------
    def _side_vector(self, items: torch.Tensor, weights: torch.Tensor,
                     side: str) -> torch.Tensor:
        """Aggregate user's pos/neg keyword vectors weighted by rating."""
        kids = (self.pos_ids if side == "pos" else self.neg_ids)[items]   # (H, K)
        kcnf = (self.pos_conf if side == "pos" else self.neg_conf)[items]  # (H, K)
        kv = self.kw_emb[kids]                                            # (H, K, D)
        w = (kcnf * weights.unsqueeze(-1)).unsqueeze(-1)                  # (H, K, 1)
        v = (kv * w).sum(dim=(0, 1))                                      # (D,)
        n = v.norm() + 1e-8
        return v / n

    def build_profile(self, items: List[int], ratings: List[float] | None = None,
                       positive_threshold: float = 4.0) -> UserProfile:
        items_t = torch.as_tensor([i for i in items if i > 0], device=self.device, dtype=torch.long)
        if items_t.numel() == 0:
            D = self.item_emb.size(1)
            return UserProfile(
                pos_vec=torch.zeros(D, device=self.device),
                neg_vec=torch.zeros(D, device=self.device),
                history=items_t,
            )
        if ratings is None:
            pos_w = torch.ones_like(items_t, dtype=torch.float32)
            neg_w = torch.zeros_like(items_t, dtype=torch.float32)
        else:
            r = torch.as_tensor([ratings[k] for k, i in enumerate(items) if i > 0],
                                device=self.device, dtype=torch.float32)
            pos_w = torch.clamp((r - 3.0) / 2.0, min=0.0)           # 0 at r<=3, 1 at r=5
            neg_w = torch.clamp((3.0 - r) / 2.0, min=0.0)           # 0 at r>=3, 1 at r=1

        pos_vec = self._side_vector(items_t, pos_w, "pos")
        neg_vec = self._side_vector(items_t, neg_w, "neg") if neg_w.sum() > 0 else torch.zeros_like(pos_vec)
        return UserProfile(pos_vec=pos_vec, neg_vec=neg_vec, history=items_t)

    # ------- scoring -------
    def item_pos_vec(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Aggregate the item's *own* pos-keyword vectors (confidence-weighted)."""
        kids = self.pos_ids[item_idx]   # (..., K)
        kcnf = self.pos_conf[item_idx]  # (..., K)
        kv = self.kw_emb[kids]          # (..., K, D)
        v = (kv * kcnf.unsqueeze(-1)).sum(dim=-2)
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return v / n

    def item_neg_vec(self, item_idx: torch.Tensor) -> torch.Tensor:
        kids = self.neg_ids[item_idx]
        kcnf = self.neg_conf[item_idx]
        kv = self.kw_emb[kids]
        v = (kv * kcnf.unsqueeze(-1)).sum(dim=-2)
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return v / n

    def semantic_score(self, user_pos: torch.Tensor, user_neg: torch.Tensor,
                       item_idx: torch.Tensor) -> torch.Tensor:
        """Raw semantic preference score ∈ approximately [-1, 1]."""
        ipv = self.item_pos_vec(item_idx)
        inv = self.item_neg_vec(item_idx)
        # user_pos: (D,) or (B, D); item vectors same last-dim
        if user_pos.dim() == 1:
            s_pos = (ipv * user_pos).sum(-1) - (inv * user_pos).sum(-1)
            s_neg = (ipv * user_neg).sum(-1) - (inv * user_neg).sum(-1)
        else:
            s_pos = (ipv * user_pos).sum(-1) - (inv * user_pos).sum(-1)
            s_neg = (ipv * user_neg).sum(-1) - (inv * user_neg).sum(-1)
        return s_pos - s_neg                # larger = item aligns with user's likes and opposes dislikes

    def item_similarity(self, item_a: torch.Tensor, item_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two items' semantic (pos) vectors."""
        va = self.item_pos_vec(item_a)
        vb = self.item_pos_vec(item_b)
        return (va * vb).sum(-1)
