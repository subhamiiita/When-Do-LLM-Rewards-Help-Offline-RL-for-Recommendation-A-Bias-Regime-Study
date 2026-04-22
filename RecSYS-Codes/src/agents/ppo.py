"""Clipped PPO with GAE."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import torch
import torch.nn as nn

from src.agents.common import HeadConfig, ItemDotActorCriticHead


@dataclass
class PPOConfig:
    d_state: int
    n_items: int
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch: int = 256
    rollout: int = 2048
    entropy: float = 0.01
    vf_coef: float = 0.5


class PPOAgent:
    def __init__(self, cfg: PPOConfig, device: torch.device, item_emb_init: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.device = device
        head_cfg = HeadConfig(d_state=cfg.d_state, n_items=cfg.n_items, hidden=cfg.hidden)
        self.net = ItemDotActorCriticHead(head_cfg, item_emb_init=item_emb_init).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, state: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, value)."""
        logits, value = self.net(state)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)
        return a, lp, value

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        _, v = self.net(state)
        return v

    def update(self, batch: dict) -> dict:
        """batch: dict with keys s (N, d), a (N,), lp_old (N,), adv (N,), ret (N,), mask (N, n_items) or None.
        Performs cfg.epochs passes of PPO over minibatches of cfg.batch."""
        s, a, lp_old, adv, ret = batch["s"], batch["a"], batch["lp_old"], batch["adv"], batch["ret"]
        N = s.size(0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_pi, total_vf, total_ent, steps = 0.0, 0.0, 0.0, 0
        for _ in range(self.cfg.epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.cfg.batch):
                mb = idx[start : start + self.cfg.batch]
                s_b = s[mb]
                a_b = a[mb]
                lp_old_b = lp_old[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]
                logits, v = self.net(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a_b)
                ratio = torch.exp(lp - lp_old_b)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv_b
                pi_loss = -torch.min(s1, s2).mean()
                vf_loss = 0.5 * (v - ret_b).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + self.cfg.vf_coef * vf_loss - self.cfg.entropy * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                total_pi += float(pi_loss.item()); total_vf += float(vf_loss.item()); total_ent += float(ent.item()); steps += 1
        steps = max(1, steps)
        return {"pi_loss": total_pi / steps, "vf_loss": total_vf / steps, "entropy": total_ent / steps}

    @torch.no_grad()
    def score_candidates(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits.gather(1, candidates)
