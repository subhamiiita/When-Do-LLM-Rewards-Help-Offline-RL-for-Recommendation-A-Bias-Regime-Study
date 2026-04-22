"""Ranking + diagnostic metrics. All tensor-batch where it makes sense."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def ndcg_at_k(scores: torch.Tensor, target_idx: torch.Tensor, k: int = 10) -> torch.Tensor:
    """scores: (B, N) higher = better. target_idx: (B,) index of the ground-truth positive.
    Returns per-row NDCG@K.
    """
    B, N = scores.shape
    # ranks: 1-based position of the target in descending score order
    order = torch.argsort(scores, dim=1, descending=True)
    pos_rank = (order == target_idx.unsqueeze(1)).float().argmax(dim=1) + 1  # (B,)
    ndcg = torch.where(
        pos_rank <= k,
        1.0 / torch.log2(pos_rank.float() + 1.0),
        torch.zeros_like(pos_rank, dtype=torch.float32),
    )
    return ndcg


def hit_at_k(scores: torch.Tensor, target_idx: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Return per-row 1/0 indicating if target is in top-K."""
    order = torch.argsort(scores, dim=1, descending=True)
    pos_rank = (order == target_idx.unsqueeze(1)).float().argmax(dim=1) + 1
    return (pos_rank <= k).float()


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def reward_snr(rewards: np.ndarray, labels: np.ndarray) -> float:
    """SNR = |mu+ - mu-| / (sigma+ + sigma- + eps)."""
    pos = rewards[labels == 1]
    neg = rewards[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    mu_p, mu_n = pos.mean(), neg.mean()
    sd_p, sd_n = pos.std() + 1e-8, neg.std() + 1e-8
    return float(abs(mu_p - mu_n) / (sd_p + sd_n))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(conf - acc)
    return float(ece)
