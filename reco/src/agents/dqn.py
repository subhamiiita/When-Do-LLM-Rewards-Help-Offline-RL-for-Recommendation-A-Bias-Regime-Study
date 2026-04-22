"""SASRec + negative-sampled DQN (value-based RL backbone).

Q(s, a) = <W h_t, e_a> + b(a)
where W is a learnable (D, D) state projection and b(a) is an item-specific bias.
Double-DQN targets with candidate-set argmax.

`freeze_encoder=True` detaches encoder outputs so RL only updates the Q-head and
embeddings are protected from reward-induced representation drift (a common
offline-RL pathology where the shaped reward pulls SASRec away from the
next-item optimum learned in supervised warmup).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNBatch:
    seq:      torch.Tensor  # (B, L)
    action:   torch.Tensor  # (B,) item_idx chosen (= next ground-truth item in offline data)
    reward:   torch.Tensor  # (B,) shaped reward from UG-MORS etc.
    next_seq: torch.Tensor  # (B, L)
    done:     torch.Tensor  # (B,) float
    cand:     torch.Tensor  # (B, K) candidate actions at next state (includes true)


class DQNHead(nn.Module):
    """Q(s, a) = <W h, e_a> + b(a). W initialised to identity so the
    post-warmup Q values match the plain inner-product baseline at RL step 0."""

    def __init__(self, dim: int, num_items: int):
        super().__init__()
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
        with torch.no_grad():
            self.state_proj.weight.copy_(torch.eye(dim))
            self.item_bias.weight.zero_()

    def q_value(self, h_last: torch.Tensor, item_emb_a: torch.Tensor,
                action_idx: torch.Tensor) -> torch.Tensor:
        s = self.state_proj(h_last)                                         # (B, D)
        return (s * item_emb_a).sum(-1) + self.item_bias(action_idx).squeeze(-1)

    def q_cand(self, h_last: torch.Tensor, item_emb_cand: torch.Tensor,
               cand_idx: torch.Tensor) -> torch.Tensor:
        s = self.state_proj(h_last)                                         # (B, D)
        return torch.einsum("bd,bkd->bk", s, item_emb_cand) + \
               self.item_bias(cand_idx).squeeze(-1)


class DQNAgent(nn.Module):
    def __init__(self, encoder, num_items: int, gamma: float = 0.9,
                 tau: float = 0.005, freeze_encoder: bool = False):
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.head_online = DQNHead(encoder.hidden_dim, num_items)
        self.head_target = copy.deepcopy(self.head_online)
        for p in self.head_target.parameters():
            p.requires_grad_(False)
        self.num_items = num_items
        self.gamma = gamma
        self.tau = tau
        self.freeze_encoder = freeze_encoder

    def set_freeze_encoder(self, freeze: bool) -> None:
        self.freeze_encoder = freeze
        for p in self.online.parameters():
            p.requires_grad_(not freeze)

    def rl_parameters(self):
        if self.freeze_encoder:
            return self.head_online.parameters()
        return list(self.online.parameters()) + list(self.head_online.parameters())

    @torch.no_grad()
    def soft_update(self):
        # encoder Polyak (no-op when online encoder is frozen — params are equal)
        for p, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
        for p, pt in zip(self.head_online.parameters(),
                         self.head_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

    def _encode_last(self, encoder, seq: torch.Tensor) -> torch.Tensor:
        h = encoder.encode(seq)                               # (B, L, D)
        last = h[:, -1, :]
        if encoder is self.online and self.freeze_encoder:
            last = last.detach()
        return last

    def td_loss(self, b: DQNBatch) -> Tuple[torch.Tensor, dict]:
        last = self._encode_last(self.online, b.seq)
        e_a = self.online.item_emb(b.action)
        if self.freeze_encoder:
            e_a = e_a.detach()
        q = self.head_online.q_value(last, e_a, b.action)

        with torch.no_grad():
            next_last_online = self._encode_last(self.online, b.next_seq)
            e_cand_online = self.online.item_emb(b.cand)
            q_next_online = self.head_online.q_cand(next_last_online,
                                                     e_cand_online, b.cand)
            a_star = q_next_online.argmax(dim=-1)
            a_star_item = b.cand.gather(1, a_star.unsqueeze(-1)).squeeze(-1)

            next_last_target = self.target.encode(b.next_seq)[:, -1, :]
            e_star_target = self.target.item_emb(a_star_item)
            q_next = self.head_target.q_value(next_last_target,
                                               e_star_target, a_star_item)
            target = b.reward + self.gamma * (1.0 - b.done) * q_next

        loss = F.smooth_l1_loss(q, target)
        return loss, {"q_mean": q.mean().detach(),
                      "target_mean": target.mean().detach(),
                      "td_err": (q - target).abs().mean().detach()}

    @torch.no_grad()
    def rank(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        last = self.online.encode(seq)[:, -1, :]
        e = self.online.item_emb(candidates)
        return self.head_online.q_cand(last, e, candidates)
