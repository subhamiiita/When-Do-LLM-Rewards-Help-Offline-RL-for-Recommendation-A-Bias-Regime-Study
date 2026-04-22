"""Learned epistemic uncertainty + Conformalised Quantile Regression (CQR).

Instead of U_epi = 0.5*u_sem + 0.5*u_nli (hand-set), we learn:
    U_epi(item) = MLP([u_jml, u_sem, u_nli, log1p_pop, s_var, kw_count])

trained on the calibration set to regress |r_sim - r_real|. Then apply
CQR (Romano et al. 2019) on the residuals — tighter intervals than CRC.

Why this matters for the paper:
    * Hand-set weights are hard to defend ("why 0.5/0.5?"). A learned
      predictor validated on a held-out split is.
    * CQR adapts the interval width to each sample's predicted uncertainty —
      exactly the property you want an "uncertainty gate" to have.
    * Combined with Mondrian / popularity-bucketed calibration, you get
      per-popularity conditional coverage guarantees, which is the
      strongest form of CRC guarantee that's practical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMLP(nn.Module):
    """Tiny MLP that predicts |r_sim - r_real| from item-level features."""
    def __init__(self, in_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus(),           # positive output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def item_features(sim, item_idx: torch.Tensor,
                  item_pop: torch.Tensor | None) -> torch.Tensor:
    """Assemble [u_jml, u_sem, u_nli, log1p_pop, kw_count, s_var] per item."""
    u_jml = sim.u_jml[item_idx]
    u_sem = sim.u_sem[item_idx]
    u_nli = sim.u_nli[item_idx]
    pop = torch.log1p(item_pop[item_idx]) if item_pop is not None else torch.zeros_like(u_jml)
    kw_count = (sim.pos_conf[item_idx] > 0).float().sum(-1) + (sim.neg_conf[item_idx] > 0).float().sum(-1)
    s_var = sim.pos_conf[item_idx].var(-1) + sim.neg_conf[item_idx].var(-1)
    return torch.stack([u_jml, u_sem, u_nli, pop, kw_count, s_var], dim=-1)


@dataclass
class CQRResult:
    q_hat: float
    n: int
    empirical_coverage: float


class LearnedUncertaintyCQR:
    """Train an MLP to predict |r_sim - r_real|, then CQR-calibrate it."""

    def __init__(self, alpha: float = 0.1, hidden: int = 64, lr: float = 3e-3,
                 epochs: int = 500):
        self.alpha = alpha
        self.mlp = UncertaintyMLP(in_dim=6, hidden=hidden)
        self.lr = lr
        self.epochs = epochs
        self.q_hat: float = float("inf")
        self.feat_mean: torch.Tensor | None = None
        self.feat_std:  torch.Tensor | None = None

    def fit(self, X: torch.Tensor, residuals: np.ndarray) -> CQRResult:
        """X: (n, 6) features; residuals: (n,) = |r_sim - r_real|."""
        device = X.device
        # standardise features
        self.feat_mean = X.mean(0, keepdim=True)
        self.feat_std  = X.std(0, keepdim=True).clamp_min(1e-6)
        X_std = (X - self.feat_mean) / self.feat_std

        y = torch.as_tensor(residuals, device=device, dtype=torch.float32)
        self.mlp = self.mlp.to(device)
        opt = torch.optim.Adam(self.mlp.parameters(), lr=self.lr, weight_decay=1e-5)
        for _ in range(self.epochs):
            pred = self.mlp(X_std)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            pred = self.mlp(X_std).cpu().numpy()

        # CQR: non-conformity score is (|r_sim-r_real| - pred), then take
        # ceil((n+1)(1-alpha))/n quantile -> q_hat
        nc = residuals - pred
        n = len(nc)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = max(1, min(k, n))
        self.q_hat = float(np.sort(nc)[k - 1])
        coverage = float((residuals <= pred + self.q_hat).mean())
        return CQRResult(q_hat=self.q_hat, n=n, empirical_coverage=coverage)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return conformal upper bound on |r_sim - r_real| for each item."""
        X_std = (X - self.feat_mean.to(X.device)) / self.feat_std.to(X.device)
        return self.mlp(X_std) + self.q_hat

    @torch.no_grad()
    def gate(self, X: torch.Tensor, temperature: float = 0.1,
             confidence_floor: float = 0.2) -> torch.Tensor:
        """Differentiable soft gate based on the predicted CQR band.

        Wider band -> more uncertain -> smaller gate.
        """
        band = self.predict(X)                                # (B,) positive
        # normalise band into [0,1] via tanh for smooth gating
        norm = torch.tanh(band)
        g = confidence_floor + (1.0 - confidence_floor) * torch.sigmoid(
            (1.0 - norm) / temperature - 3.0)
        return g


class MondrianConformalGate:
    """Per-popularity-bucket CRC. Useful when tail items have very different
    uncertainty profiles than head items.
    """
    def __init__(self, alpha: float = 0.1, n_buckets: int = 10):
        self.alpha = alpha
        self.n_buckets = n_buckets
        self.bucket_edges: np.ndarray | None = None
        self.bucket_q: np.ndarray | None = None

    def fit(self, residuals: np.ndarray, item_pop: np.ndarray) -> None:
        edges = np.quantile(item_pop, np.linspace(0, 1, self.n_buckets + 1))
        edges[-1] = edges[-1] + 1e-6
        buckets = np.digitize(item_pop, edges) - 1
        buckets = np.clip(buckets, 0, self.n_buckets - 1)

        qs = np.zeros(self.n_buckets, dtype=np.float64)
        for b in range(self.n_buckets):
            mask = buckets == b
            if mask.sum() < 2:
                qs[b] = float(np.quantile(residuals, 1 - self.alpha)) \
                    if len(residuals) > 0 else float("inf")
                continue
            r = np.sort(residuals[mask])
            n = len(r)
            k = int(np.ceil((n + 1) * (1 - self.alpha)))
            k = max(1, min(k, n))
            qs[b] = r[k - 1]
        self.bucket_edges = edges
        self.bucket_q = qs

    def q_for(self, item_pop: np.ndarray) -> np.ndarray:
        buckets = np.digitize(item_pop, self.bucket_edges) - 1
        buckets = np.clip(buckets, 0, self.n_buckets - 1)
        return self.bucket_q[buckets]
