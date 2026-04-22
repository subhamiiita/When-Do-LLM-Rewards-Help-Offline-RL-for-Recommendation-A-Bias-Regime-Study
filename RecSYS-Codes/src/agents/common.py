"""Shared actor/critic heads for RL agents.

We use a standard MLP head that maps the frozen SASRec state vector to per-item
Q-values / policy logits. The head is randomly initialized and learns entirely
from the RL reward signal — this is the clean setting for comparing reward
variants (baseline vote vs UG-MORS), because any gains attributable to the
SASRec prior are eliminated.

We experimented with SASRec-initialized residual heads, but found that DQN
training on the simulator's (biased) rewards actively *destroys* the SASRec
prior through residual updates that favor training-selected items at the
expense of test items. The cleanest experimental setup is therefore agents
that learn from scratch, with the relative (UG-MORS − baseline) lift as the
paper-relevant signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class HeadConfig:
    d_state: int
    n_items: int
    hidden: int = 256


class QHead(nn.Module):
    """Q-values Q(s, a) for every action a ∈ [0, n_items)."""

    def __init__(self, cfg: HeadConfig, item_emb_init: Optional[torch.Tensor] = None):
        super().__init__()
        # item_emb_init is accepted for API compat but currently ignored.
        self.net = nn.Sequential(
            nn.Linear(cfg.d_state, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.n_items),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class ActorCriticHead(nn.Module):
    """Policy gradient shared trunk with actor (categorical) + critic (scalar)."""

    def __init__(self, cfg: HeadConfig, item_emb_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_state, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.actor = nn.Linear(cfg.hidden, cfg.n_items)
        self.critic = nn.Linear(cfg.hidden, 1)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        return self.actor(h), self.critic(h).squeeze(-1)


# Aliases so downstream imports still resolve (we renamed the classes during the
# ItemDot experiment; keep both names working).
ItemDotQHead = QHead
ItemDotActorCriticHead = ActorCriticHead
