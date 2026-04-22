"""Keyword margin m(s, a) and similarity margin u(s, a), both fully batched on GPU.

Following the paper:
    pos = sum_{i in I_pos(h)} |D^{ic}_pos ∩ D^{i}_pos|
    neg = sum_{i in I_neg(h)} |D^{ic}_neg ∩ D^{i}_neg|
    m(s, a) = (pos - neg) / (pos + neg + eps)

    βpos = max_{i in I_pos(h)} cos(E^{ic}_pos, E^{i}_pos)
    βneg = max_{i in I_neg(h)} cos(E^{ic}_neg, E^{i}_neg)
    u(s, a) = βpos - βneg

`h` is the user's historical interaction set; I_pos(h) are items with y=1, I_neg(h)
items with y=0. We use the optionally-restricted category-matched history `h_C`
(see paper): items in h with the same primary category as the candidate. When
no same-category items exist, we fall back to full h.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.reward.common import RewardEngine, RewardState


@dataclass
class Margins:
    m: torch.Tensor   # (B,) in [-1, 1]
    u: torch.Tensor   # (B,) approx in [-1, 1]
    pos_overlap: torch.Tensor   # (B,) float  - total pos keyword overlap
    neg_overlap: torch.Tensor   # (B,) float  - total neg keyword overlap
    beta_pos: torch.Tensor      # (B,) max pos similarity
    beta_neg: torch.Tensor      # (B,) max neg similarity


def compute_margins(engine: RewardEngine, state: RewardState, action: torch.Tensor) -> Margins:
    """Fully batched. Complexity O(B · T_max · n_kw_or_d) via matmul/einsum."""
    B = action.size(0)
    device = engine.device
    n_kw = engine.cache.n_keywords

    # (B,) candidate item's kw indicator vectors
    cand_pos_ind = engine.pos_ind[action]   # (B, n_kw) fp16
    cand_neg_ind = engine.neg_ind[action]   # (B, n_kw) fp16

    # ---- keyword overlap per history item: (B, T_max) ------------------
    hist = state.hist_ids        # (B, T)
    hist_pad = engine.sas_pad
    # Clamp pad positions to a safe index (0) for gather; mask out later.
    safe_hist = torch.where(hist != hist_pad, hist, torch.zeros_like(hist))
    # Gather pos indicators for every history item: (B, T, n_kw)
    hist_pos_ind = engine.pos_ind[safe_hist]    # (B, T, n_kw)
    hist_neg_ind = engine.neg_ind[safe_hist]    # (B, T, n_kw)

    # Overlap per history step = <cand_ind, hist_step_ind> → (B, T)
    overlap_pos_per_step = torch.einsum("bk,btk->bt", cand_pos_ind.float(), hist_pos_ind.float())
    overlap_neg_per_step = torch.einsum("bk,btk->bt", cand_neg_ind.float(), hist_neg_ind.float())

    # Mask: only count history positions with mask True AND proper label
    valid = state.hist_mask                                # (B, T)
    pos_label = (state.hist_labels == 1) & valid           # (B, T)
    neg_label = (state.hist_labels == 0) & valid           # (B, T)

    # Apply category-matched filter h_C: item has same primary category as candidate.
    # When no matches, fallback to full h. We use torch string-compare via integer ids.
    cat_ids = engine.cat_ids                               # (n_items,) long; filled by env setup
    cand_cat = cat_ids[action]                             # (B,)
    hist_cat = cat_ids[safe_hist]                          # (B, T)
    same_cat = (hist_cat == cand_cat.unsqueeze(1)) & valid  # (B, T)
    # For envs with any same-cat, use same_cat; otherwise fall back to full valid
    has_same = same_cat.any(dim=1, keepdim=True)           # (B, 1)
    hC_mask = torch.where(has_same, same_cat, valid)       # (B, T)

    pos_sel = (pos_label & hC_mask).float()                # (B, T)
    neg_sel = (neg_label & hC_mask).float()                # (B, T)

    pos_overlap = (overlap_pos_per_step * pos_sel).sum(dim=1)   # (B,)
    neg_overlap = (overlap_neg_per_step * neg_sel).sum(dim=1)   # (B,)

    eps = 1.0
    m = (pos_overlap - neg_overlap) / (pos_overlap + neg_overlap + eps)
    m = m.clamp(-1.0, 1.0)

    # ---- similarity margin u(s, a) -------------------------------------
    # E_pos, E_neg are L2-normalized (n_items, d).
    E_pos = engine.cache.E_pos
    E_neg = engine.cache.E_neg

    cand_Ep = E_pos[action]      # (B, d)
    cand_En = E_neg[action]      # (B, d)
    hist_Ep = E_pos[safe_hist]   # (B, T, d)
    hist_En = E_neg[safe_hist]   # (B, T, d)

    cos_pos = torch.einsum("bd,btd->bt", cand_Ep, hist_Ep)  # (B, T)
    cos_neg = torch.einsum("bd,btd->bt", cand_En, hist_En)  # (B, T)

    # Where row has no valid pos entries, we set cos = 0 (neutral)
    cos_pos_m = cos_pos.masked_fill(~pos_sel.bool(), float("-inf"))
    cos_neg_m = cos_neg.masked_fill(~neg_sel.bool(), float("-inf"))

    has_pos = pos_sel.bool().any(dim=1)
    has_neg = neg_sel.bool().any(dim=1)
    beta_pos = torch.where(has_pos, cos_pos_m.max(dim=1).values, torch.zeros(B, device=device))
    beta_neg = torch.where(has_neg, cos_neg_m.max(dim=1).values, torch.zeros(B, device=device))

    u = (beta_pos - beta_neg).clamp(-1.0, 1.0)

    return Margins(m=m, u=u, pos_overlap=pos_overlap, neg_overlap=neg_overlap, beta_pos=beta_pos, beta_neg=beta_neg)
