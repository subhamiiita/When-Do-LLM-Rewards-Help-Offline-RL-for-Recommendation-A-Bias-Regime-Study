"""Reward functions for UG-MORS ablation.

All rewards share a signature:
    compute(batch) -> (reward: (B,), diag: dict)

batch contains:
    user_pos    (B, D)     — user positive-keyword profile
    user_neg    (B, D)     — user negative-keyword profile
    item_idx    (B,)
    sim         FrozenSimulator

The four rewards tested in the paper:
    1) BinaryVoteReward       — baseline (current SOTA per the abstract)
    2) NaiveContinuous        — raw semantic score, no uncertainty (ablation)
    3) HardGatedReward        — continuous but mask items with U>thr (ablation)
    4) UGMORS                 — continuous, confidence-modulated, soft CRC gate

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .conformal import ConformalGate


# ----------------- base -----------------

class RewardModule:
    name = "base"

    def set_calibration(self, gate: ConformalGate | None) -> None:
        self.gate = gate

    def compute(self, batch: dict) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError


def _epistemic(sim, item_idx, weights=(0.0, 0.5, 0.5)) -> torch.Tensor:
    """Epistemic uncertainty = w_jml*u_jml + w_sem*u_sem + w_nli*u_nli.

    Default weights: u_jml is treated as aleatoric (paraphrase noise, w=0),
    u_sem + u_nli are epistemic (meaning disagreement + contradictions).
    This decomposition is new to this paper.
    """
    w_jml, w_sem, w_nli = weights
    return (w_jml * sim.u_jml[item_idx]
            + w_sem * sim.u_sem[item_idx]
            + w_nli * sim.u_nli[item_idx])


def _semantic_score(sim, user_pos, user_neg, item_idx, temperature=0.15) -> torch.Tensor:
    """Raw continuous semantic score in [-1, 1] (approx)."""
    s = sim.semantic_score(user_pos, user_neg, item_idx)
    # tanh squash so reward is bounded; temperature controls sharpness
    return torch.tanh(s / temperature)


# ----------------- 1) binary vote (baseline) -----------------

class BinaryVoteReward(RewardModule):
    """Replicates the binary-voting SOTA in the abstract.

    We simulate an ensemble vote: each of the cached self-consistency runs
    "votes" whether this item aligns with user preferences. Here we
    approximate the vote by thresholding the continuous semantic score.
    The reward is in {-1, +1}. This yields sparse gradients — which is
    precisely what the paper argues against.
    """
    name = "binary"

    def __init__(self, sim_temperature=0.15, threshold=0.0, **_):
        self.temperature = sim_temperature
        self.threshold = threshold

    def compute(self, batch):
        sim = batch["sim"]
        r_cont = _semantic_score(sim, batch["user_pos"], batch["user_neg"],
                                 batch["item_idx"], self.temperature)
        r = torch.where(r_cont > self.threshold,
                        torch.ones_like(r_cont),
                        -torch.ones_like(r_cont))
        return r, {"r_cont": r_cont.detach()}


# ----------------- 2) naive continuous (ablation) -----------------

class NaiveContinuousReward(RewardModule):
    """The straw-man from the abstract: dense gradient, but zero trust check.

    Expect HIGH simulator scores and LOW real-world scores — this is the
    hallucination-propagation demonstration.
    """
    name = "naive_continuous"

    def __init__(self, sim_temperature=0.15, **_):
        self.temperature = sim_temperature

    def compute(self, batch):
        sim = batch["sim"]
        r = _semantic_score(sim, batch["user_pos"], batch["user_neg"],
                            batch["item_idx"], self.temperature)
        return r, {"r_cont": r.detach()}


# ----------------- 3) hard-gated (ablation) -----------------

class HardGatedReward(RewardModule):
    """Continuous reward, but multiplied by 0 when u_llm >= threshold.

    Ablates the 'softness' of UG-MORS — shows the gate should be
    continuous not a step function. Expect unstable training near the
    threshold boundary.
    """
    name = "hard_gate"

    def __init__(self, sim_temperature=0.15, hard_gate_threshold=0.55,
                 epistemic_weights=(0.0, 0.5, 0.5), **_):
        self.temperature = sim_temperature
        self.threshold = hard_gate_threshold
        self.weights = tuple(epistemic_weights)

    def compute(self, batch):
        sim = batch["sim"]
        r = _semantic_score(sim, batch["user_pos"], batch["user_neg"],
                            batch["item_idx"], self.temperature)
        u_epi = _epistemic(sim, batch["item_idx"], self.weights)
        mask = (u_epi < self.threshold).float()
        return r * mask, {"r_cont": r.detach(), "u_epi": u_epi.detach(),
                          "gate": mask.detach()}


# ----------------- 4) UG-MORS (ours) -----------------

class UGMORSReward(RewardModule):
    """The contribution.

    Three mechanisms compose:
      (a) Confidence-modulated continuous reward — per-keyword confidence
          is already baked into sim.item_pos_vec/item_neg_vec (low-conf
          keywords contribute less). This gives DENSE gradients everywhere.
      (b) Epistemic-only uncertainty:
          U_epi = 0*u_jml + 0.5*u_sem + 0.5*u_nli      (paraphrase ignored)
      (c) Soft CRC gate:  g(U_epi) ∈ [floor, 1] where the inflection is
          the conformally-calibrated u* that guarantees
          P(|r_sim - r_real| <= q_hat) >= 1 - alpha.

    Final reward:    r = r_sem * g(U_epi)

    Rationale: low-uncertainty items receive near-full reward signal; high-
    uncertainty items get attenuated but never zeroed (dense gradient
    preserved). The gate is calibrated, not heuristic.
    """
    name = "ug_mors"

    def __init__(self, sim_temperature=0.15,
                 crc_alpha=0.1, gate_temperature=0.1, confidence_floor=0.1,
                 epistemic_weights=(0.0, 0.5, 0.5), **_):
        self.temperature = sim_temperature
        self.crc_alpha = crc_alpha
        self.gate_T = gate_temperature
        self.floor = confidence_floor
        self.weights = tuple(epistemic_weights)
        self.gate = ConformalGate(alpha=crc_alpha,
                                   temperature=gate_temperature,
                                   confidence_floor=confidence_floor)

    def set_calibration(self, gate):
        if gate is not None:
            self.gate = gate

    def compute(self, batch):
        sim = batch["sim"]
        r = _semantic_score(sim, batch["user_pos"], batch["user_neg"],
                            batch["item_idx"], self.temperature)
        u_epi = _epistemic(sim, batch["item_idx"], self.weights)
        g = self.gate.gate(u_epi)
        return r * g, {"r_cont": r.detach(), "u_epi": u_epi.detach(),
                       "gate": g.detach()}


# ----------------- registry -----------------

REWARDS: Dict[str, type] = {
    "binary": BinaryVoteReward,
    "naive_continuous": NaiveContinuousReward,
    "hard_gate": HardGatedReward,
    "ug_mors": UGMORSReward,
}


def build_reward(name: str, reward_cfg: dict) -> RewardModule:
    if name not in REWARDS:
        raise KeyError(f"unknown reward '{name}'. options={list(REWARDS)}")
    return REWARDS[name](**reward_cfg)
