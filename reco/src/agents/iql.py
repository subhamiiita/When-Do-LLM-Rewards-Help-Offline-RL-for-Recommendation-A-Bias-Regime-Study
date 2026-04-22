"""IQL (Implicit Q-Learning) for offline sequential recommendation.

Kostrikov, Nair, Levine 2022. Key properties for us:

  * No out-of-distribution action evaluation (uses V, not max_a Q(s, a')).
    This matters because our action space = |items| (thousands), so the
    double-DQN max operator is noisy and amplifies reward-shaping error.

  * Decouples value learning from policy extraction. We learn V via
    expectile regression, Q via standard Bellman, and extract the policy
    by advantage-weighted regression (AWR) on the OFFLINE actions. That
    means we never invent out-of-distribution actions — essential when
    the simulator is an LLM derivative with unknown extrapolation.

  * The offline-RL loss is cleanly decomposable per-transition, so we can
    apply the UG-MORS gate as a PER-SAMPLE LOSS WEIGHT:
        L = g(u) * (L_V + L_Q + L_pi)  +  (1-g(u)) * L_BC  +  lam * L_pess

Policy form (we use the AWR form, not the Gaussian / softmax form):
    pi(a|s) ~ exp((Q(s,a) - V(s)) / beta)
so we train pi by cross-entropy on the OFFLINE action weighted by
exp((Q - V) / beta).

Supports UG-MORS v2 by returning per-sample raw loss terms (reduction='none')
so the trainer can weight them.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IQLBatch:
    seq:      torch.Tensor  # (B, L) state
    action:   torch.Tensor  # (B,)   offline action taken
    reward:   torch.Tensor  # (B,)   shaped reward r_sem
    next_seq: torch.Tensor  # (B, L)
    done:     torch.Tensor  # (B,)
    cand:     torch.Tensor  # (B, K) candidate set (col 0 = true) for BC + pessimism


def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2. tau=0.5 => mean; tau>0.5 biases toward upper quantile."""
    w = torch.where(diff >= 0, tau, 1.0 - tau)
    return w * diff.pow(2)


class IQLAgent(nn.Module):
    """SASRec-shared-encoder IQL. Three heads: V, Q, and the policy (implicit)."""

    def __init__(self, encoder, num_items: int,
                 iql_tau: float = 0.7, iql_beta: float = 3.0,
                 gamma: float = 0.99, ema_tau: float = 0.005):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        D = encoder.hidden_dim
        # V(s): scalar value of state
        self.v_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        # Q(s, a): <W h, e_a> + b_a
        self.q_proj = nn.Linear(D, D, bias=False)
        self.q_bias = nn.Embedding(num_items, 1, padding_idx=0)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(D))
            self.q_bias.weight.zero_()
        self.q_target_proj = copy.deepcopy(self.q_proj)
        self.q_target_bias = copy.deepcopy(self.q_bias)
        for p in list(self.q_target_proj.parameters()) + list(self.q_target_bias.parameters()):
            p.requires_grad_(False)

        self.num_items = num_items
        self.iql_tau = iql_tau
        self.iql_beta = iql_beta
        self.gamma = gamma
        self.ema_tau = ema_tau

    # ---------- forward helpers ----------

    def _encode_last(self, enc, seq):
        return enc.encode(seq)[:, -1, :]          # (B, D)

    def V(self, seq: torch.Tensor) -> torch.Tensor:
        h = self._encode_last(self.encoder, seq)
        return self.v_head(h).squeeze(-1)

    def Q(self, seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encode_last(self.encoder, seq)
        e = self.encoder.item_emb(action)
        return (self.q_proj(h) * e).sum(-1) + self.q_bias(action).squeeze(-1)

    def Q_target(self, seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_last(self.target_encoder, seq)
            e = self.target_encoder.item_emb(action)
            return (self.q_target_proj(h) * e).sum(-1) + self.q_target_bias(action).squeeze(-1)

    def Q_cand(self, seq: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Q-values over a candidate set (B, K). For BC logits + pessimism."""
        h = self._encode_last(self.encoder, seq)
        s = self.q_proj(h)                                # (B, D)
        e = self.encoder.item_emb(cand)                   # (B, K, D)
        return torch.einsum("bd,bkd->bk", s, e) + self.q_bias(cand).squeeze(-1)

    # ---------- losses (per-sample; mean reduction is done by trainer) ----------

    def loss_V(self, seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """V targets the tau-expectile of Q(s, a_data)."""
        q = self.Q_target(seq, action)
        v = self.V(seq)
        diff = q - v                                      # (B,)
        return _expectile_loss(diff, self.iql_tau)        # (B,)

    def loss_Q(self, seq: torch.Tensor, action: torch.Tensor,
               reward: torch.Tensor, next_seq: torch.Tensor,
               done: torch.Tensor) -> torch.Tensor:
        """Bellman with V(s') on the target encoder as the bootstrap (NOT max_a Q)."""
        q = self.Q(seq, action)
        with torch.no_grad():
            h_next = self._encode_last(self.target_encoder, next_seq)
            v_next = self.v_head(h_next).squeeze(-1)
            target = reward + self.gamma * (1.0 - done) * v_next
        return F.smooth_l1_loss(q, target, reduction="none")

    def loss_pi_awr(self, seq: torch.Tensor, action: torch.Tensor,
                    cand: torch.Tensor) -> torch.Tensor:
        """Advantage-weighted regression over the candidate slate.

        cand[:, 0] must be the true action; we treat this as a multi-class
        classification where the weight on the correct-class CE is exp(A/beta).
        """
        with torch.no_grad():
            q_a = self.Q(seq, action)
            v   = self.V(seq)
            adv = (q_a - v).clamp(-10.0, 10.0)
            w   = torch.exp(adv / self.iql_beta).clamp(max=100.0)

        logits = self.Q_cand(seq, cand)                    # (B, K)
        logp = F.log_softmax(logits, dim=-1)
        # target index is 0 (true action is col 0)
        nll = -logp[:, 0]                                  # (B,)
        return w * nll

    def loss_bc(self, seq: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Plain cross-entropy with target=0 — pure behaviour cloning. (B,)"""
        logits = self.Q_cand(seq, cand)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target, reduction="none")

    def loss_pessimism(self, seq: torch.Tensor, action_data: torch.Tensor,
                       num_random: int = 8) -> torch.Tensor:
        """CQL-lite: penalise Q on random OOD actions relative to the data action."""
        B = seq.size(0)
        a_rand = torch.randint(1, self.num_items, (B, num_random), device=seq.device)
        q_rand = self.Q_cand(seq, a_rand).logsumexp(dim=-1)        # (B,)
        q_data = self.Q(seq, action_data)                          # (B,)
        return (q_rand - q_data).clamp(min=0.0)

    # ---------- EMA update ----------

    @torch.no_grad()
    def ema_update(self) -> None:
        t = self.ema_tau
        for p, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(1 - t).add_(p.data, alpha=t)
        for p, pt in zip(self.q_proj.parameters(), self.q_target_proj.parameters()):
            pt.data.mul_(1 - t).add_(p.data, alpha=t)
        for p, pt in zip(self.q_bias.parameters(), self.q_target_bias.parameters()):
            pt.data.mul_(1 - t).add_(p.data, alpha=t)

    # ---------- eval ranking ----------

    @torch.no_grad()
    def rank(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """For eval: higher = better. We use Q(s,a) as the ranking score."""
        return self.Q_cand(seq, candidates)
