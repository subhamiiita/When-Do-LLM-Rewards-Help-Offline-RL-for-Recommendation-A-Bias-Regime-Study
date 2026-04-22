"""UG-MORS scalar reward: Uncertainty-Gated Multi-Objective Reward Shaping.

Paper Eqs. 2-13. Reward vector r = (r_rel, r_div, r_per, r_ret) scalarized by
dynamic weights conditioned on gate g = 1 - U_LLM(action) and schedule ξ(t).

    r_rel(s,a) = 2 · (α · p_sta + β · g · p_sem) − 1,   α + β = 1
    p_sem       = sigmoid(τ_m · m(s,a) + τ_u · u(s,a))
    r_div       = 1 − 2 · max_{i∈W_t} cos(e_a, e_i)
    r_per       = NLI_cache[u, a]   (normalized, clipped to [-1, 1])
    r_ret       = 1 − 2 · q_leave(s)
    w_rel=1,  w_div = η_div·g·ξ(t),  w_per = η_per·g,  w_ret = η_ret·ξ(t)
    R = clip(Σ w_j · r_j, -1, 1)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.reward.baseline_vote import compute_p_sta
from src.reward.common import RewardEngine, RewardState
from src.reward.margins import Margins, compute_margins


@dataclass
class UGMorsAblation:
    """Controls which components / gate are active. Default: full UG-MORS."""
    gate_enabled: bool = True           # False → naive continuous (no gate on p_sem)
    dynamic_weights: bool = True        # False → fixed weights (no ξ(t), g is just 1)
    use_rel: bool = True
    use_div: bool = True
    use_per: bool = True
    use_ret: bool = True


def _xi(t: torch.Tensor, T0: int) -> torch.Tensor:
    """Monotone schedule: min(1, t/T0), t is (B,) int or float."""
    return torch.clamp(t.float() / float(max(1, T0)), 0.0, 1.0)


def compute_ug_mors_components(
    engine: RewardEngine,
    state: RewardState,
    action: torch.Tensor,
    margins: Margins | None = None,
    ablation: UGMorsAblation | None = None,
) -> dict:
    """Returns dict with each component r_* and the gate/weights for diagnostics."""
    ab = ablation or UGMorsAblation()
    B = action.size(0)
    dev = action.device

    if margins is None:
        margins = compute_margins(engine, state, action)

    # ---- Gate (1 − U_LLM)
    u_llm = engine.cache.u_llm[action]     # (B,) in [0, 1]
    g = (1.0 - u_llm).clamp(0.0, 1.0)       # (B,)
    if not ab.gate_enabled:
        g = torch.ones_like(g)

    # ---- r_rel
    p_sta = compute_p_sta(engine, state, action)                               # (B,)
    p_sem = torch.sigmoid(engine.tau_m * margins.m + engine.tau_u * margins.u)  # (B,) in (0,1)
    alpha, beta = engine.alpha, engine.beta
    r_rel_gated = 2.0 * (alpha * p_sta + beta * g * p_sem) - 1.0                # (B,) in [-1, 1]
    r_rel = r_rel_gated if ab.use_rel else torch.zeros_like(r_rel_gated)

    # ---- r_div: 1 − 2 · max cos(e_a, e_window)
    e_a = engine.cache.item_emb[action]                                         # (B, d)
    win = state.recent_actions                                                  # (B, k) padded with -1
    if win.numel() == 0 or win.size(1) == 0:
        r_div = torch.zeros(B, device=dev)
    else:
        safe_win = torch.where(win >= 0, win, torch.zeros_like(win))
        win_emb = engine.cache.item_emb[safe_win]                               # (B, k, d)
        sims = torch.einsum("bd,bkd->bk", e_a, win_emb)                         # (B, k)
        sims = sims.masked_fill(win < 0, float("-inf"))
        has_any = (win >= 0).any(dim=1)
        max_sim = torch.where(has_any, sims.max(dim=1).values, torch.zeros(B, device=dev))
        r_div_val = (1.0 - 2.0 * max_sim).clamp(-1.0, 1.0)
        r_div = r_div_val if ab.use_div else torch.zeros_like(r_div_val)
    if not ab.use_div:
        r_div = torch.zeros(B, device=dev)

    # ---- r_per via NLI cache
    if ab.use_per:
        r_per = engine.nli_lookup(state.user_idx, action)                       # (B,) in [-1, 1]
    else:
        r_per = torch.zeros(B, device=dev)

    # ---- r_ret via retention LSTM
    if ab.use_ret:
        hist = state.hist_ids
        B_, T = hist.shape
        max_len = engine.sas_max_len
        if T < max_len:
            pad = torch.full((B_, max_len - T), engine.sas_pad, device=hist.device, dtype=hist.dtype)
            seq = torch.cat([pad, hist], dim=1)
        else:
            seq = hist[:, -max_len:]
        with torch.no_grad():
            q_leave = engine.retention(seq)                                     # (B,) in (0,1)
        r_ret = (1.0 - 2.0 * q_leave).clamp(-1.0, 1.0)
    else:
        r_ret = torch.zeros(B, device=dev)

    # ---- Dynamic weights
    t = state.t if state.t.numel() > 1 else state.t.expand(B)
    if ab.dynamic_weights:
        xi_t = _xi(t, engine.T0)                                                # (B,)
        w_rel = torch.ones(B, device=dev)
        w_div = engine.eta_div * g * xi_t
        w_per = engine.eta_per * g
        w_ret = engine.eta_ret * xi_t
    else:
        w_rel = torch.ones(B, device=dev)
        w_div = torch.full((B,), engine.eta_div, device=dev)
        w_per = torch.full((B,), engine.eta_per, device=dev)
        w_ret = torch.full((B,), engine.eta_ret, device=dev)

    R = (w_rel * r_rel + w_div * r_div + w_per * r_per + w_ret * r_ret).clamp(-1.0, 1.0)

    return dict(
        r_rel=r_rel,
        r_div=r_div,
        r_per=r_per,
        r_ret=r_ret,
        g=g,
        p_sta=p_sta,
        p_sem=p_sem,
        w_rel=w_rel,
        w_div=w_div,
        w_per=w_per,
        w_ret=w_ret,
        R=R,
    )


def compute_ug_mors(
    engine: RewardEngine,
    state: RewardState,
    action: torch.Tensor,
    margins: Margins | None = None,
    ablation: UGMorsAblation | None = None,
) -> torch.Tensor:
    """Scalar reward only (B,). Wrapper around compute_ug_mors_components."""
    return compute_ug_mors_components(engine, state, action, margins=margins, ablation=ablation)["R"]
