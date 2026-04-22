"""Simulator-vs-real reward gap — the KEY figure of the paper.

For each (user, test_item, real_rating) triple, compute the reward each of
the four reward functions would give. Compare to the real rating mapped to
[-1, 1]. Report mean, MAE, and correlation with u_llm.

The story the paper tells:
  * naive_continuous: sim-reward is high even when real-rating is low
    (hallucination propagation), and the gap is strongly correlated with u_llm.
  * binary_vote: gap is uncorrelated but signal is coarse.
  * hard_gate: reduces gap on low-U items but unstable on borderline items.
  * ug_mors: gap is small AND the residual gap is decorrelated from u_llm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..rewards import build_reward
from ..rewards.conformal import ConformalGate


def _compute_rewards(reward, sim, profiles, users, items, device):
    pos = torch.stack([profiles[int(u)][0] for u in users]).to(device)
    neg = torch.stack([profiles[int(u)][1] for u in users]).to(device)
    items_t = torch.as_tensor(items, device=device, dtype=torch.long)
    r, _ = reward.compute({"sim": sim, "user_pos": pos, "user_neg": neg,
                            "item_idx": items_t})
    return r.cpu().numpy()


def sim_real_gap(sim, profiles, splits, reward_cfg: dict, device: str,
                 out_path: Path | None = None,
                 gate: ConformalGate | None = None) -> Dict[str, Dict[str, float]]:
    test = splits.test
    users = test["user_idx"].values
    items = test["item_idx"].values
    ratings = test["rating"].values.astype(np.float32)
    r_real = (ratings - 3.0) / 2.0                           # [-1, 1]

    # epistemic uncertainty for each test item
    w = tuple(reward_cfg.get("epistemic_weights", [0.0, 0.5, 0.5]))
    u_epi = (w[0] * sim.u_jml.cpu().numpy()
             + w[1] * sim.u_sem.cpu().numpy()
             + w[2] * sim.u_nli.cpu().numpy())[items]
    u_llm = sim.u_llm.cpu().numpy()[items]

    # fit ug_mors gate for this run (same protocol as training) unless caller
    # passed a pre-fit one (so we measure the exact gate the policy saw).
    if gate is None:
        from ..train.loop import _fit_calibration_gate
        gate = _fit_calibration_gate(sim, profiles, splits.val, splits.test,
                                      splits.calib_users, reward_cfg, device)

    results: Dict[str, Dict[str, float]] = {}
    for name in ("binary", "naive_continuous", "hard_gate", "ug_mors"):
        reward = build_reward(name, reward_cfg)
        if name == "ug_mors":
            reward.set_calibration(gate)
        r_sim = _compute_rewards(reward, sim, profiles, users, items, device)

        gap = r_sim - r_real
        abs_gap = np.abs(gap)
        # pearson r with u_llm
        if abs_gap.std() > 1e-6 and u_llm.std() > 1e-6:
            corr_u = float(np.corrcoef(abs_gap, u_llm)[0, 1])
            corr_epi = float(np.corrcoef(abs_gap, u_epi)[0, 1])
        else:
            corr_u = float("nan"); corr_epi = float("nan")

        # bias-corrected MAE: remove the dataset-level mean shift so we
        # measure pure calibration shape rather than where the sim sits
        # relative to the real-rating mean (which is reward-independent).
        gap_centered = gap - gap.mean()
        mae_unbiased = float(np.abs(gap_centered).mean())

        # stratified by epistemic uncertainty — UG-MORS's gate only fires
        # on high-u_epi items, so the aggregate MAE dilutes its effect.
        # Report MAE on the top-20% and bottom-80% u_epi buckets separately.
        thr = np.quantile(u_epi, 0.80)
        hi_mask = u_epi >= thr
        lo_mask = ~hi_mask
        mae_hi = float(abs_gap[hi_mask].mean()) if hi_mask.any() else float("nan")
        mae_lo = float(abs_gap[lo_mask].mean()) if lo_mask.any() else float("nan")
        mse_hi = float(np.mean(gap[hi_mask] ** 2)) if hi_mask.any() else float("nan")
        mse_lo = float(np.mean(gap[lo_mask] ** 2)) if lo_mask.any() else float("nan")

        results[name] = {
            "mean_r_sim": float(r_sim.mean()),
            "mean_r_real": float(r_real.mean()),
            "MAE": float(abs_gap.mean()),
            "MSE": float(np.mean(gap ** 2)),
            "MAE_unbiased": mae_unbiased,
            "MAE_highU": mae_hi,
            "MAE_lowU": mae_lo,
            "MSE_highU": mse_hi,
            "MSE_lowU": mse_lo,
            "u_epi_threshold_p80": float(thr),
            "corr_|gap|_u_llm": corr_u,
            "corr_|gap|_u_epi": corr_epi,
            "n": int(len(gap)),
            "n_highU": int(hi_mask.sum()),
            "n_lowU": int(lo_mask.sum()),
        }

    # record calibration artifacts so we can compare u_star/q_hat across seeds
    results["_gate"] = {
        "alpha": gate.alpha, "T": gate.T, "floor": gate.floor,
        "q_hat": gate.q_hat, "u_star": gate.u_star,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results
