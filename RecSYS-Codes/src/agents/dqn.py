"""Double DQN for discrete full-catalog actions."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.agents.common import HeadConfig, ItemDotQHead


@dataclass
class DQNConfig:
    d_state: int
    n_items: int
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.9
    replay_size: int = 100_000
    batch: int = 256
    target_update: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000


class ReplayBuffer:
    def __init__(self, capacity: int, d_state: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.s = torch.zeros((capacity, d_state), dtype=torch.float32, device=device)
        self.a = torch.zeros(capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.sn = torch.zeros((capacity, d_state), dtype=torch.float32, device=device)
        self.d = torch.zeros(capacity, dtype=torch.float32, device=device)

    def add_batch(self, s, a, r, sn, d):
        n = s.size(0)
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.sn[idx] = sn
        self.d[idx] = d.float()
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx]


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: torch.device, item_emb_init: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.device = device
        head_cfg = HeadConfig(d_state=cfg.d_state, n_items=cfg.n_items, hidden=cfg.hidden)
        self.q = ItemDotQHead(head_cfg, item_emb_init=item_emb_init).to(device)
        self.q_target = ItemDotQHead(head_cfg, item_emb_init=item_emb_init).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size, cfg.d_state, device)
        self.global_step = 0

    def _epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, state: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        eps = self._epsilon()
        B = state.size(0)
        if torch.rand(1).item() < eps:
            if valid_mask is not None:
                # Sample uniformly over valid actions per env
                probs = valid_mask.float()
                probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
                return torch.multinomial(probs, num_samples=1).squeeze(1)
            return torch.randint(0, self.cfg.n_items, (B,), device=self.device)
        q = self.q(state)
        if valid_mask is not None:
            q = q.masked_fill(~valid_mask, float("-inf"))
        return q.argmax(dim=1)

    def update(self) -> dict:
        if self.replay.size < self.cfg.batch:
            return {}
        s, a, r, sn, d = self.replay.sample(self.cfg.batch)
        with torch.no_grad():
            # Double-DQN: argmax by online net, eval by target net
            next_q_online = self.q(sn)
            a_next = next_q_online.argmax(dim=1)
            next_q = self.q_target(sn).gather(1, a_next.unsqueeze(1)).squeeze(1)
            y = r + self.cfg.gamma * (1.0 - d) * next_q
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.item()), "eps": self._epsilon()}

    @torch.no_grad()
    def score_candidates(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """(B, d_state), (B, K) -> (B, K). Used for eval ranking."""
        q = self.q(state)
        return q.gather(1, candidates)
