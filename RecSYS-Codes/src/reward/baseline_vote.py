"""Base-paper reward: majority vote among {f_mat, f_sim, f_sta}.

f_mat = 1[pos_overlap > neg_overlap] with random tiebreak
f_sim = 1[u(s,a) > 0]              with random tiebreak
f_sta = 1[p_sta(s,a) > 0.5]
R_vote = 1 if (f_mat + f_sim + f_sta) >= 2 else 0
"""
from __future__ import annotations

import torch

from src.reward.common import RewardEngine, RewardState
from src.reward.margins import Margins, compute_margins


def compute_p_sta(engine: RewardEngine, state: RewardState, action: torch.Tensor) -> torch.Tensor:
    """Sigmoid of SASRec score for the candidate given the current history.

    Input history tensor must be padded with sas_pad where inactive; we align to
    the SASRec's max_len window (right-aligned).
    """
    hist = state.hist_ids  # (B, T) — the env may already use the same T as sasrec
    B, T = hist.shape
    if T < engine.sas_max_len:
        pad = torch.full((B, engine.sas_max_len - T), engine.sas_pad, device=hist.device, dtype=hist.dtype)
        seq = torch.cat([pad, hist], dim=1)
    elif T > engine.sas_max_len:
        seq = hist[:, -engine.sas_max_len :]
    else:
        seq = hist
    # Replace any non-pad placeholders with pad for SASRec (should already be aligned)
    with torch.no_grad():
        h = engine.sasrec.last_hidden(seq)                   # (B, d)
        # SASRec item embedding dot product → logit
        emb = engine.sasrec.item_emb(action)                 # (B, d)
        logit = (h * emb).sum(dim=-1)                        # (B,)
        return torch.sigmoid(logit)                          # (B,) in (0,1)


def compute_baseline_vote(
    engine: RewardEngine,
    state: RewardState,
    action: torch.Tensor,
    margins: Margins | None = None,
    seed: int | None = None,
) -> dict:
    """Returns dict with keys: f_mat, f_sim, f_sta, vote, p_sta."""
    if margins is None:
        margins = compute_margins(engine, state, action)

    # f_mat: 1 if pos_overlap > neg_overlap; rand tiebreak if equal
    gt = margins.pos_overlap > margins.neg_overlap
    eq = margins.pos_overlap == margins.neg_overlap
    g = torch.Generator(device=action.device)
    if seed is not None:
        g.manual_seed(seed)
    tb = (torch.rand(action.size(0), device=action.device) > 0.5)
    f_mat = torch.where(gt, torch.ones_like(gt, dtype=torch.long), torch.where(eq, tb.long(), torch.zeros_like(gt, dtype=torch.long)))

    # f_sim: 1 if u > 0; rand tiebreak if == 0
    gts = margins.u > 0
    eqs = margins.u == 0
    tb2 = (torch.rand(action.size(0), device=action.device) > 0.5)
    f_sim = torch.where(gts, torch.ones_like(gts, dtype=torch.long), torch.where(eqs, tb2.long(), torch.zeros_like(gts, dtype=torch.long)))

    # f_sta: 1 if p_sta > 0.5
    p_sta = compute_p_sta(engine, state, action)
    f_sta = (p_sta > 0.5).long()

    votes = f_mat + f_sim + f_sta
    yvote = (votes >= 2).long()

    return dict(f_mat=f_mat, f_sim=f_sim, f_sta=f_sta, p_sta=p_sta, vote=yvote)
