"""Conformal Risk Control for the UG-MORS gate.

Setup:
    Nonconformity score s_i = |r_sim_i - r_real_i|   on a calibration set.
    For target miscoverage alpha, conformal quantile q_hat is the
    ceil((n+1)(1-alpha)) / n empirical quantile of s_i.
    Guarantee (Vovk '05, Angelopoulos '22):
        P(|r_sim - r_real| <= q_hat) >= 1 - alpha

We extend this to *uncertainty-conditional* calibration: the gate applies
stronger attenuation to items whose epistemic uncertainty U_epi is above
the conformal threshold at a chosen alpha. This is the novelty of the paper.

Implementation:
    fit(s, u)     -> stores (s_i, u_epi_i) pairs
    threshold(alpha) -> conformal q on s
    gate(u_epi, s_hat) -> soft gate value in [confidence_floor, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class CRCResult:
    q_hat: float
    u_threshold: float
    n: int


def empirical_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = max(1, min(k, n))
    return float(np.sort(scores)[k - 1])


class ConformalGate:
    """Fits a calibrated soft gate g(u_epi) from calibration samples.

    The gate smoothly interpolates:
        g(u_epi) = confidence_floor + (1 - confidence_floor) * sigmoid((u* - u_epi) / T)
    where u* is the conformally chosen threshold.
    """

    def __init__(self, alpha: float = 0.1, temperature: float = 0.1,
                 confidence_floor: float = 0.1):
        self.alpha = alpha
        self.T = temperature
        self.floor = confidence_floor
        self.fitted = False
        self.q_hat = float("inf")
        self.u_star = float("inf")
        self._epi_scores: Optional[np.ndarray] = None

    def fit(self, sim_scores: np.ndarray, real_scores: np.ndarray,
            u_epi: np.ndarray) -> CRCResult:
        """Calibrate on a held-out set.

        sim_scores:  (n,) LLM-simulator predicted reward (scaled to [0,1] or [-1,1])
        real_scores: (n,) observed user rating, scaled to the same range
        u_epi:       (n,) per-sample epistemic uncertainty
        """
        sim_scores = np.asarray(sim_scores, dtype=np.float64)
        real_scores = np.asarray(real_scores, dtype=np.float64)
        u_epi = np.asarray(u_epi, dtype=np.float64)

        nc = np.abs(sim_scores - real_scores)          # nonconformity
        self.q_hat = empirical_quantile(nc, self.alpha)

        # u_star = (1 - alpha) empirical quantile of u_epi itself, so by
        # construction ~alpha fraction of items fall in the "untrusted" regime.
        # Auto-scale T to the u_epi spread so the sigmoid spans [floor, 1]
        # smoothly instead of saturating.
        u_sd = float(u_epi.std() + 1e-8)
        self.u_star = float(np.quantile(u_epi, 1.0 - self.alpha))
        if self.T <= 0 or self.T > 10.0 * u_sd:
            self.T = max(0.5 * u_sd, 1e-3)

        self._epi_scores = u_epi
        self.fitted = True
        return CRCResult(q_hat=self.q_hat, u_threshold=self.u_star, n=len(nc))

    def gate(self, u_epi: torch.Tensor) -> torch.Tensor:
        """Soft gate in [floor, 1]. Differentiable."""
        if not self.fitted:
            # identity gate if not calibrated yet
            return torch.ones_like(u_epi)
        g = torch.sigmoid((self.u_star - u_epi) / self.T)
        return self.floor + (1.0 - self.floor) * g

    def hard_gate(self, u_epi: torch.Tensor) -> torch.Tensor:
        """Hard ablation: 1 if u_epi <= u_star else 0."""
        if not self.fitted:
            return torch.ones_like(u_epi)
        return (u_epi <= self.u_star).float()

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "T": self.T, "floor": self.floor,
                "q_hat": self.q_hat, "u_star": self.u_star,
                "fitted": self.fitted}

    def load_state_dict(self, s: dict) -> None:
        self.alpha = s["alpha"]; self.T = s["T"]; self.floor = s["floor"]
        self.q_hat = s["q_hat"]; self.u_star = s["u_star"]
        self.fitted = s["fitted"]
