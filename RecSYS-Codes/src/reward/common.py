"""Reward engine state: the shared objects all reward variants use.

We pack everything into a single class so rewards can be computed in big
batched GPU passes without rebuilding tensors per step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.data.cache import CacheBundle, build_cache, build_keyword_indicator_fast
from src.data.loaders import load_splits
from src.models.retention import RetentionConfig, RetentionLSTM
from src.models.sasrec import SASRec, SASRecConfig


@dataclass
class RewardState:
    """Per-env batched state used by reward computations.

    All tensors have leading batch dim B (= number of parallel envs).
    """

    # Sequence tensor of item ids (B, T_max) padded with pad_id
    hist_ids: torch.Tensor
    # Binary like/dislike per history step (B, T_max), same padding pattern; values in {0,1}; pad cells should be ignored via mask
    hist_labels: torch.Tensor
    # Mask (B, T_max) bool - True for real (non-pad) positions
    hist_mask: torch.Tensor
    # Per-env user idx (B,)
    user_idx: torch.Tensor
    # Time step within rollout (scalar, same for whole batch OR per-env)
    t: torch.Tensor
    # Recent-action ring buffer for diversity (B, k_div) padded with -1
    recent_actions: torch.Tensor


class RewardEngine:
    """Holds all stateful pieces needed to compute any reward variant.

    The engine is dataset-specific (loads per-dataset SASRec, retention LSTM, NLI cache,
    keyword indicator matrices, item/kw embeddings) but reward-variant-agnostic.
    """

    def __init__(self, dataset: str, device: Optional[torch.device] = None):
        self.dataset = dataset
        self.device = device or get_device()
        cfg = dataset_config(dataset)
        self.cfg = cfg

        # --- Splits and cache
        self.splits = load_splits(dataset)
        self.cache: CacheBundle = build_cache(dataset, self.device)
        self.n_items = self.cache.n_items
        self.n_users = self.splits["n_users"]

        # --- Keyword indicator matrices (n_items, n_kw) fp16 on device
        self.pos_ind, self.neg_ind = build_keyword_indicator_fast(self.cache)

        # --- Primary category ids per item (for h_C filter)
        cat2id: dict[str, int] = {}
        cat_ids_list: list[int] = []
        for c in self.cache.item_categories:
            key = c or ""
            if key not in cat2id:
                cat2id[key] = len(cat2id)
            cat_ids_list.append(cat2id[key])
        self.cat_ids = torch.tensor(cat_ids_list, dtype=torch.long, device=self.device)

        # --- SASRec (f_sta)
        sa = cfg["sasrec"]
        ck = torch.load(project_root() / cfg["paths"]["sasrec_ckpt"], map_location=self.device, weights_only=False)
        self.sasrec = SASRec(SASRecConfig(**ck["cfg"])).to(self.device).eval()
        self.sasrec.load_state_dict(ck["state_dict"])
        self.sas_max_len = sa["max_len"]
        self.sas_pad = self.sasrec.pad_id

        # --- Retention LSTM (q_leave)
        rc = cfg["retention"]
        rk = torch.load(project_root() / cfg["paths"]["retention_ckpt"], map_location=self.device, weights_only=False)
        self.retention = RetentionLSTM(RetentionConfig(**rk["cfg"])).to(self.device).eval()
        self.retention.load_state_dict(rk["state_dict"])

        # --- NLI cache (lazy)
        self._nli_mat: Optional[torch.Tensor] = None
        self._nli_mask: Optional[torch.Tensor] = None
        self._nli_mean: float = 0.0
        self._nli_std: float = 1.0

        # --- Reward hyperparams
        rw = cfg["reward"]
        self.alpha = float(rw["alpha"])
        self.beta = float(rw["beta"])
        self.tau_m = float(rw["tau_m"])
        self.tau_u = float(rw["tau_u"])
        self.eta_div = float(rw["eta_div"])
        self.eta_per = float(rw["eta_per"])
        self.eta_ret = float(rw["eta_ret"])
        self.div_k = int(rw["div_window_k"])
        self.T0 = int(rw["T0_schedule"])

    # ------------------------------------------------------------------
    # NLI cache
    # ------------------------------------------------------------------
    def load_nli(self) -> None:
        """Load NLI cache lazily. Computes per-dataset normalization (mean/std)
        from the TOUCHED (computed) cells only; untouched cells stay at raw 0.0.
        At consume time: r_per = clip((raw - mean) / std, -1, 1).
        """
        if self._nli_mat is not None:
            return
        path = project_root() / self.cfg["paths"]["nli_cache"]
        z = np.load(path)
        raw = z["nli"].astype(np.float32)  # (n_users, n_items)
        mask = z["mask"]
        touched = raw[mask]
        if touched.size == 0:
            mean, std = 0.0, 1.0
        else:
            mean = float(touched.mean())
            std = float(touched.std() + 1e-6)
        self._nli_mean = mean
        self._nli_std = std
        self._nli_mat = torch.from_numpy(raw).to(self.device)
        self._nli_mask = torch.from_numpy(mask).to(self.device)
        print(f"[reward {self.dataset}] NLI loaded: mean={mean:.3f} std={std:.3f} coverage={mask.mean()*100:.2f}%")

    def nli_lookup(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Return r_per contribution (B,) float. Normalized per dataset, clipped to [-1, 1].

        Cells outside the precomputed mask get 0 (neutral) AFTER normalization.
        """
        if self._nli_mat is None:
            self.load_nli()
        raw = self._nli_mat[user_idx, item_idx]            # (B,)
        mask = self._nli_mask[user_idx, item_idx]           # (B,)
        norm = (raw - self._nli_mean) / self._nli_std
        norm = norm.clamp(-1.0, 1.0)
        # Untouched cells → treat as 0 (no signal) regardless of raw value (which was 0 anyway).
        return torch.where(mask, norm, torch.zeros_like(norm))
