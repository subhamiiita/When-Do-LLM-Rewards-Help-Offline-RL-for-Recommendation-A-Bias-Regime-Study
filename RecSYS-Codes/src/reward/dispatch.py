"""Dispatch: given a reward-variant name, return the scalar reward for a batch.

Variants:
    baseline_vote        : majority vote ∈ {0,1}       (original AAAI-25 paper)
    naive_continuous     : UG-MORS's r_rel with gate=1 (hallucination propagation test)
    ug_mors              : full UG-MORS
    ug_mors_fixed        : UG-MORS with fixed (non-dynamic) weights
    ug_mors_no_div       : component ablation
    ug_mors_no_per       : component ablation
    ug_mors_no_ret       : component ablation
    ug_pbrs              : policy-invariant PBRS shaping atop baseline vote

`prev_state` is only used by ug_pbrs; other variants read only the current state.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.reward.baseline_vote import compute_baseline_vote
from src.reward.common import RewardEngine, RewardState
from src.reward.margins import Margins, compute_margins
from src.reward.naive_continuous import compute_naive_continuous
from src.reward.ug_mors import UGMorsAblation, compute_ug_mors
from src.reward.ug_pbrs import compute_ug_pbrs


VARIANT_REGISTRY = [
    "baseline_vote",
    "naive_continuous",
    "ug_mors",
    "ug_mors_fixed",
    "ug_mors_no_div",
    "ug_mors_no_per",
    "ug_mors_no_ret",
    "ug_pbrs",
]


def compute_reward(
    variant: str,
    engine: RewardEngine,
    state: RewardState,
    action: torch.Tensor,
    prev_state: Optional[RewardState] = None,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Returns scalar reward (B,) for the requested variant."""
    # We compute margins once because every variant needs them (cheap shared work)
    margins = compute_margins(engine, state, action)

    if variant == "baseline_vote":
        out = compute_baseline_vote(engine, state, action, margins=margins)
        return out["vote"].float()

    if variant == "naive_continuous":
        return compute_naive_continuous(engine, state, action, margins=margins)

    if variant.startswith("ug_mors"):
        ab = UGMorsAblation()
        if variant == "ug_mors_fixed":
            ab.dynamic_weights = False
        elif variant == "ug_mors_no_div":
            ab.use_div = False
        elif variant == "ug_mors_no_per":
            ab.use_per = False
        elif variant == "ug_mors_no_ret":
            ab.use_ret = False
        elif variant != "ug_mors":
            raise ValueError(f"Unknown UG-MORS variant: {variant}")
        return compute_ug_mors(engine, state, action, margins=margins, ablation=ab)

    if variant == "ug_pbrs":
        if prev_state is None:
            raise ValueError("ug_pbrs requires prev_state (for Φ(s))")
        out = compute_ug_pbrs(engine, prev_state, state, action, gamma=gamma)
        return out["R"]

    raise ValueError(f"Unknown variant: {variant!r}. Known: {VARIANT_REGISTRY}")
