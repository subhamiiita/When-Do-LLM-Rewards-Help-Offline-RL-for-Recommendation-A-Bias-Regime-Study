"""SASRec + PPO (policy-gradient RL backbone) with slate-level log-prob.

Policy: pi(a|s) = softmax(<h_t, e_a> / T) over a candidate slate
        (positive + K sampled negatives). A critic V(s) is a small head on
        top of the encoder.

Advantage = reward - V(s). Single-step (bandit-style) formulation is standard
for slate recommenders; we keep gamma=0 here and rely on the shaped reward
to encode sequence-level preference signal through the frozen simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOBatch:
    seq:      torch.Tensor  # (B, L)
    cand:     torch.Tensor  # (B, K)  — first column is the true action
    reward:   torch.Tensor  # (B,)


class PPOAgent(nn.Module):
    def __init__(self, encoder, lr: float = 1e-3,
                 clip: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, temperature: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.critic = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(encoder.hidden_dim, 1),
        )
        self.clip = clip
        self.vc = value_coef
        self.ec = entropy_coef
        self.T = temperature

    def _logits_and_value(self, seq, cand):
        h = self.encoder.encode(seq)
        last = h[:, -1, :]
        v = self.critic(last).squeeze(-1)
        e = self.encoder.item_emb(cand)
        logits = torch.einsum("bd,bkd->bk", last, e) / self.T
        return logits, v, last

    @torch.no_grad()
    def collect_old_logprob(self, b: PPOBatch) -> torch.Tensor:
        logits, _, _ = self._logits_and_value(b.seq, b.cand)
        logp = F.log_softmax(logits, dim=-1)
        return logp[:, 0]  # true action is col 0 in cand by convention

    def loss(self, b: PPOBatch, old_logp: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        logits, v, _ = self._logits_and_value(b.seq, b.cand)
        logp = F.log_softmax(logits, dim=-1)
        new_logp = logp[:, 0]
        adv = (b.reward - v).detach()
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.smooth_l1_loss(v, b.reward)
        entropy = -(logp.exp() * logp).sum(-1).mean()
        total = actor_loss + self.vc * value_loss - self.ec * entropy
        return total, {"actor": actor_loss.detach(),
                       "value": value_loss.detach(),
                       "entropy": entropy.detach(),
                       "adv_mean": adv.mean().detach()}

    @torch.no_grad()
    def rank(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self._logits_and_value(seq, candidates)
        return logits
