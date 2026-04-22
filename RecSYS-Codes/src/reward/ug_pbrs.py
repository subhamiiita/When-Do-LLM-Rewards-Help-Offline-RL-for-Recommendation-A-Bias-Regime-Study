"""UG-PBRS: Potential-based shaping atop the baseline vote reward.

    R_ext(s, a, s') = 2 · yvote − 1 ∈ {−1, +1}
    Φ(s)  = η_div · Φ_div(s) + η_per · Φ_per(s)
        Φ_div(s) = − mean pairwise cos similarity among last k actions
        Φ_per(s) = mean NLI r_per of last k actions
    R_PBRS(s, a, s') = R_ext + γ · Φ(s') − Φ(s)

Policy invariance: under standard PBRS assumptions (Ng et al. 1999), adding γ·Φ(s')−Φ(s)
preserves the optimal policy wrt R_ext. We compute Φ on the *action window* state only
(not full history) for tractability.
"""
from __future__ import annotations

import torch

from src.reward.baseline_vote import compute_baseline_vote
from src.reward.common import RewardEngine, RewardState


def _phi_div(engine: RewardEngine, window: torch.Tensor) -> torch.Tensor:
    """window: (B, k) item ids with -1 padding. Returns (B,) negative of average pairwise cos."""
    B, k = window.shape
    if k < 2:
        return torch.zeros(B, device=window.device)
    safe = torch.where(window >= 0, window, torch.zeros_like(window))
    emb = engine.cache.item_emb[safe]                           # (B, k, d)
    pad_mask = (window < 0)                                     # True where pad
    sims = torch.einsum("bkd,bld->bkl", emb, emb)               # (B, k, k)
    # Mask out self-pairs and pads
    eye = torch.eye(k, dtype=torch.bool, device=window.device).unsqueeze(0).expand(B, k, k)
    pad_mat = pad_mask.unsqueeze(2) | pad_mask.unsqueeze(1)     # (B, k, k)
    invalid = eye | pad_mat
    sims = sims.masked_fill(invalid, 0.0)
    denom = (~invalid).float().sum(dim=(1, 2)).clamp(min=1.0)
    avg = sims.sum(dim=(1, 2)) / denom
    return -avg


def _phi_per(engine: RewardEngine, window: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
    """Mean r_per over last-k actions per env. Untouched NLI cells → 0."""
    B, k = window.shape
    if k < 1:
        return torch.zeros(B, device=window.device)
    safe = torch.where(window >= 0, window, torch.zeros_like(window))
    nli_vals = []
    for i in range(k):
        nli_vals.append(engine.nli_lookup(user_idx, safe[:, i]))
    stack = torch.stack(nli_vals, dim=1)                        # (B, k)
    mask = (window >= 0).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (stack * mask).sum(dim=1) / denom


def compute_potential(engine: RewardEngine, state: RewardState) -> torch.Tensor:
    """Φ(s) = η_div · Φ_div(s) + η_per · Φ_per(s). Returns (B,)."""
    phi_div = _phi_div(engine, state.recent_actions)
    phi_per = _phi_per(engine, state.recent_actions, state.user_idx)
    return engine.eta_div * phi_div + engine.eta_per * phi_per


def compute_ug_pbrs(
    engine: RewardEngine,
    state_prev: RewardState,
    state_next: RewardState,
    action: torch.Tensor,
    gamma: float = 0.9,
) -> dict:
    """Returns dict with R_ext, Φ(s), Φ(s'), R_pbrs."""
    votes = compute_baseline_vote(engine, state_prev, action)
    R_ext = 2.0 * votes["vote"].float() - 1.0                   # (B,) ∈ {−1, +1}
    phi_s = compute_potential(engine, state_prev)
    phi_sp = compute_potential(engine, state_next)
    R = R_ext + gamma * phi_sp - phi_s
    return dict(R=R.clamp(-1.0, 1.0), R_ext=R_ext, phi_s=phi_s, phi_sp=phi_sp, vote=votes["vote"])
