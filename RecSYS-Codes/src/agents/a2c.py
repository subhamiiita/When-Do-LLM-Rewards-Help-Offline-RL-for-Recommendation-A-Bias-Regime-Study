"""Synchronous A2C — same head as PPO, single-step update per rollout slice."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import torch

from src.agents.common import HeadConfig, ItemDotActorCriticHead


@dataclass
class A2CConfig:
    d_state: int
    n_items: int
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.9
    rollout: int = 32
    entropy: float = 0.01
    vf_coef: float = 0.5


class A2CAgent:
    def __init__(self, cfg: A2CConfig, device: torch.device, item_emb_init: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.device = device
        head_cfg = HeadConfig(d_state=cfg.d_state, n_items=cfg.n_items, hidden=cfg.hidden)
        self.net = ItemDotActorCriticHead(head_cfg, item_emb_init=item_emb_init).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, state: torch.Tensor, valid_mask: torch.Tensor | None = None):
        logits, value = self.net(state)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        _, v = self.net(state)
        return v

    def update(self, batch: dict) -> dict:
        """Single-pass A2C."""
        s, a, adv, ret = batch["s"], batch["a"], batch["adv"], batch["ret"]
        logits, v = self.net(s)
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(a)
        pi_loss = -(lp * adv.detach()).mean()
        vf_loss = 0.5 * (v - ret).pow(2).mean()
        ent = dist.entropy().mean()
        loss = pi_loss + self.cfg.vf_coef * vf_loss - self.cfg.entropy * ent

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()
        return {"pi_loss": float(pi_loss.item()), "vf_loss": float(vf_loss.item()), "entropy": float(ent.item())}

    @torch.no_grad()
    def score_candidates(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits.gather(1, candidates)
