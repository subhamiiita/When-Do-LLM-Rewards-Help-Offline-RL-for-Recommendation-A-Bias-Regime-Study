"""Naive continuous reward ablation: UG-MORS relevance r_rel WITHOUT the uncertainty gate.

    p_sem = sigmoid(τ_m · m + τ_u · u)
    R_naive = 2 · (α · p_sta + β · p_sem) − 1              (no g factor)

This directly tests whether LLM hallucination propagates when the agent fully trusts
the semantic signal. Paper predicts this variant will overfit the simulator and
degrade SimReal transfer.
"""
from __future__ import annotations

import torch

from src.reward.common import RewardEngine, RewardState
from src.reward.margins import Margins, compute_margins
from src.reward.ug_mors import UGMorsAblation, compute_ug_mors_components


def compute_naive_continuous(
    engine: RewardEngine,
    state: RewardState,
    action: torch.Tensor,
    margins: Margins | None = None,
) -> torch.Tensor:
    """Returns scalar reward (B,) - identical to UG-MORS's r_rel but with gate forced to 1
    AND the div/per/ret components zeroed out (pure relevance, ungated)."""
    ab = UGMorsAblation(
        gate_enabled=False,
        dynamic_weights=True,
        use_rel=True,
        use_div=False,
        use_per=False,
        use_ret=False,
    )
    comps = compute_ug_mors_components(engine, state, action, margins=margins, ablation=ab)
    return comps["R"]
