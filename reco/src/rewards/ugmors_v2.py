"""UG-MORS v2 — uncertainty as a PER-SAMPLE LOSS WEIGHT, not a reward target.

Key reframing compared to v1:
    v1: r = r_sem * g(u_epi)            <-- gate scales the reward scalar
    v2: r = r_sem                        <-- reward is the raw semantic score
        L = g(u) * L_RL + (1-g(u)) * L_BC + lam * u * L_pessimism

Why this fixes the null-result in v1:
    * In v1, when the gate saturated at g~0.94 (as observed), UG-MORS
      reward was indistinguishable from naive_continuous reward.
    * In v2, even when g is close to 1 everywhere, the L_BC and L_pess
      terms still provide reward-independent training signal — so the
      uncertainty modulates WHICH LOSS DOMINATES per transition, which
      is a strictly stronger notion of control than scaling a scalar.

Interpretation:
    * Low uncertainty item  -> g~1 -> pure offline-RL (pursues shaped reward).
    * High uncertainty item -> g~0 -> pure behavior cloning (trusts data).
    * Pessimism term penalises Q on out-of-distribution actions in
      proportion to state uncertainty: dangerous extrapolation is
      suppressed precisely where the reward is least trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .conformal import ConformalGate
from .registry import RewardModule, _epistemic, _semantic_score


@dataclass
class UGMORSv2Output:
    """Returned by UGMORSv2Reward.compute — the trainer composes the loss."""
    r_sem: torch.Tensor        # (B,) dense semantic reward in [-1, 1]
    w_rl:  torch.Tensor        # (B,) loss weight for the RL terms
    w_bc:  torch.Tensor        # (B,) loss weight for BC (=1 - w_rl)
    u_epi: torch.Tensor        # (B,) epistemic uncertainty (for pessimism + logging)
    gate:  torch.Tensor        # (B,) identical to w_rl; kept for logging back-compat


class UGMORSv2Reward(RewardModule):
    name = "ug_mors_v2"

    def __init__(self, sim_temperature: float = 0.15,
                 crc_alpha: float = 0.1, gate_temperature: float = 0.1,
                 confidence_floor: float = 0.2,
                 epistemic_weights=(0.0, 0.5, 0.5), **_):
        self.temperature = sim_temperature
        self.weights = tuple(epistemic_weights)
        self.gate = ConformalGate(alpha=crc_alpha,
                                   temperature=gate_temperature,
                                   confidence_floor=confidence_floor)

    def set_calibration(self, gate):
        if gate is not None:
            self.gate = gate

    def compute(self, batch) -> Tuple[torch.Tensor, Dict]:
        sim = batch["sim"]
        r_sem = _semantic_score(sim, batch["user_pos"], batch["user_neg"],
                                batch["item_idx"], self.temperature)
        u_epi = _epistemic(sim, batch["item_idx"], self.weights)
        g = self.gate.gate(u_epi)                                # (B,) in [floor, 1]
        w_rl = g
        w_bc = 1.0 - g

        diag = {"r_sem": r_sem.detach(),
                "u_epi": u_epi.detach(),
                "w_rl":  w_rl.detach(),
                "w_bc":  w_bc.detach(),
                "gate":  g.detach()}
        # NB: return r_sem as the "reward" (for backward-compat with the old loop),
        # and expose w_rl/w_bc in the aux dict for the v2 trainer.
        return r_sem, diag
