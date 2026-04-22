"""TRPO via conjugate gradient.

Simplified but faithful: natural gradient on actor, critic trained with MSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from typing import Optional

import torch

from src.agents.common import HeadConfig, ItemDotActorCriticHead


@dataclass
class TRPOConfig:
    d_state: int
    n_items: int
    hidden: int = 256
    critic_lr: float = 3e-4
    gamma: float = 0.9
    max_kl: float = 0.01
    cg_iters: int = 10
    damping: float = 0.1
    rollout: int = 2048
    vf_epochs: int = 5
    vf_batch: int = 256
    entropy: float = 0.01


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in module.parameters()])


def _set_flat_params(module: torch.nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n


def _flat_grads(loss: torch.Tensor, module: torch.nn.Module, create_graph: bool = False) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(module.parameters()), create_graph=create_graph, retain_graph=True)
    return torch.cat([g.reshape(-1) for g in grads])


class TRPOAgent:
    def __init__(self, cfg: TRPOConfig, device: torch.device, item_emb_init: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.device = device
        head_cfg = HeadConfig(d_state=cfg.d_state, n_items=cfg.n_items, hidden=cfg.hidden)
        self.net = ItemDotActorCriticHead(head_cfg, item_emb_init=item_emb_init).to(device)
        # Train critic with Adam; actor updated via natural gradient.
        self.vf_opt = torch.optim.Adam(
            list(self.net.critic.parameters()) + list(self.net.trunk.parameters()),
            lr=cfg.critic_lr,
        )

    @torch.no_grad()
    def act(self, state: torch.Tensor, valid_mask=None):
        logits, v = self.net(state)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        _, v = self.net(state)
        return v

    def _kl(self, s: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
        new_logits, _ = self.net(s)
        p_old = torch.softmax(old_logits, dim=-1)
        logp_old = torch.log_softmax(old_logits, dim=-1)
        logp_new = torch.log_softmax(new_logits, dim=-1)
        kl = (p_old * (logp_old - logp_new)).sum(dim=-1)
        return kl.mean()

    def _conjugate_gradient(self, fvp: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = r.dot(r)
        for _ in range(self.cfg.cg_iters):
            Ap = fvp(p)
            alpha = rr / (p.dot(Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rr_new = r.dot(r)
            if float(rr_new) < 1e-10:
                break
            p = r + (rr_new / (rr + 1e-10)) * p
            rr = rr_new
        return x

    def update(self, batch: dict) -> dict:
        s, a, adv, ret, old_lp = batch["s"], batch["a"], batch["adv"], batch["ret"], batch["lp_old"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- Compute policy gradient g (surrogate) --------------------
        # Set actor params require_grad True, trunk shared
        logits, _ = self.net(s)
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(a)
        ratio = torch.exp(lp - old_lp.detach())
        surr = (ratio * adv.detach()).mean()
        ent = dist.entropy().mean()
        g_loss = -(surr + self.cfg.entropy * ent)

        # Policy params (actor + shared trunk). Critic excluded.
        policy_params = list(self.net.trunk.parameters()) + list(self.net.actor.parameters())
        g_flat = torch.autograd.grad(g_loss, policy_params, retain_graph=True)
        g_flat = -torch.cat([gg.reshape(-1) for gg in g_flat])  # note negative for ascent

        # ---- FVP for KL ---------------------------------------------
        old_logits = logits.detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            kl = self._kl(s, old_logits)
            gkl = torch.autograd.grad(kl, policy_params, create_graph=True)
            gkl_flat = torch.cat([gg.reshape(-1) for gg in gkl])
            gvp = (gkl_flat * v).sum()
            h = torch.autograd.grad(gvp, policy_params, retain_graph=True)
            h_flat = torch.cat([gg.reshape(-1) for gg in h])
            return h_flat + self.cfg.damping * v

        step_dir = self._conjugate_gradient(fvp, g_flat)

        shs = 0.5 * step_dir.dot(fvp(step_dir))
        lm = torch.sqrt(shs / self.cfg.max_kl + 1e-10)
        full_step = step_dir / (lm + 1e-10)

        # Line search
        old_params = torch.cat([p.detach().view(-1) for p in policy_params])
        success = False
        for frac in [0.5**i for i in range(10)]:
            new_params = old_params + frac * full_step
            idx = 0
            for p in policy_params:
                n = p.numel()
                p.data.copy_(new_params[idx : idx + n].view_as(p))
                idx += n
            with torch.no_grad():
                kl = self._kl(s, old_logits)
            if float(kl) < self.cfg.max_kl * 1.5:
                success = True
                break
        if not success:
            # revert
            idx = 0
            for p in policy_params:
                n = p.numel()
                p.data.copy_(old_params[idx : idx + n].view_as(p))
                idx += n

        # ---- Critic update (MSE) ------------------------------------
        N = s.size(0)
        for _ in range(self.cfg.vf_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.cfg.vf_batch):
                mb = perm[start : start + self.cfg.vf_batch]
                _, v = self.net(s[mb])
                vf_loss = 0.5 * (v - ret[mb]).pow(2).mean()
                self.vf_opt.zero_grad(set_to_none=True)
                vf_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.net.critic.parameters()) + list(self.net.trunk.parameters()), 0.5)
                self.vf_opt.step()
        return {"surr": float(surr.item()), "entropy": float(ent.item()), "kl": float(kl.item() if success else 0.0), "vf_loss": float(vf_loss.item())}

    @torch.no_grad()
    def score_candidates(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits.gather(1, candidates)
