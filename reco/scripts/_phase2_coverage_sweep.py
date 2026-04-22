"""Phase 2: post-hoc coverage validation sweep at alpha in {0.05, 0.10, 0.20}
on ML-1M IQL (seed 42). No retraining — reuses existing sim cache & splits.

For each alpha:
  - Fit ConformalGate on calib_users (val_df slice) using same protocol as training.
  - Evaluate empirical coverage = fraction of test-set samples (from NON-calib users)
    with |r_sim - r_real| <= q_hat_alpha.
  - Proposition 1 claim: empirical coverage >= 1 - alpha (finite-sample correction
    gives ceil((n+1)(1-alpha))/n; Barber 2023 noted deflation under non-exch.).
  - Hard abort: if coverage < 1 - alpha - 0.05 on any alpha, flag it.

Outputs:
  paper_rewrite/data/coverage_validation.json
  paper_rewrite/data/coverage_table.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import make_splits, user_sequences
from src.simulator.cache import build_cache
from src.simulator.frozen_sim import FrozenSimulator
from src.rewards.conformal import ConformalGate
from src.train.loop import _build_user_profiles


def fit_gate_for_alpha(sim, profiles, val_df, test_df, calib_users,
                        reward_cfg, device, alpha):
    """Replicate _fit_calibration_gate but at a specified alpha."""
    mask = val_df["user_idx"].isin(calib_users)
    cal_val = val_df[mask]
    if len(cal_val) == 0:
        mask = test_df["user_idx"].isin(calib_users)
        cal_val = test_df[mask]
    users = cal_val["user_idx"].values
    items = torch.as_tensor(cal_val["item_idx"].values, device=device, dtype=torch.long)
    ratings = cal_val["rating"].values.astype(np.float32)
    r_real = (ratings - 3.0) / 2.0  # [-1, 1]
    pos = torch.stack([profiles[int(u)][0] for u in users]).to(device)
    neg = torch.stack([profiles[int(u)][1] for u in users]).to(device)
    with torch.no_grad():
        s = sim.semantic_score(pos, neg, items)
        r_sim = torch.tanh(s / reward_cfg["sim_temperature"]).cpu().numpy()

    w = tuple(reward_cfg.get("epistemic_weights", [0.0, 0.5, 0.5]))
    u_epi = (w[0] * sim.u_jml[items].cpu().numpy()
             + w[1] * sim.u_sem[items].cpu().numpy()
             + w[2] * sim.u_nli[items].cpu().numpy())

    gate = ConformalGate(alpha=alpha,
                         temperature=reward_cfg["gate_temperature"],
                         confidence_floor=reward_cfg["confidence_floor"])
    gate.fit(r_sim, r_real, u_epi)
    return gate, r_sim, r_real, u_epi


def eval_coverage_on_test(sim, profiles, test_df, calib_users, reward_cfg,
                           device, q_hat):
    """Empirical coverage: fraction of test users (non-calib) where
    |r_sim - r_real| <= q_hat."""
    mask = ~test_df["user_idx"].isin(calib_users)
    eval_df = test_df[mask]
    users = eval_df["user_idx"].values
    items = torch.as_tensor(eval_df["item_idx"].values, device=device, dtype=torch.long)
    ratings = eval_df["rating"].values.astype(np.float32)
    r_real = (ratings - 3.0) / 2.0
    pos = torch.stack([profiles[int(u)][0] for u in users]).to(device)
    neg = torch.stack([profiles[int(u)][1] for u in users]).to(device)
    with torch.no_grad():
        s = sim.semantic_score(pos, neg, items)
        r_sim = torch.tanh(s / reward_cfg["sim_temperature"]).cpu().numpy()
    nc = np.abs(r_sim - r_real)
    cov = float((nc <= q_hat).mean())
    n = int(len(nc))
    return cov, n, nc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = "movielens-1m"
    seed = 42
    print(f"[cov-sweep] dataset={ds}  seed={seed}  device={device}")

    reward_cfg = {
        "topk_keywords": 16,
        "sim_temperature": 0.15,
        "binary_vote_threshold": 0.5,
        "hard_gate_threshold": 0.55,
        "crc_alpha": 0.1,
        "epistemic_weights": [0.0, 0.5, 0.5],
        "gate_temperature": 0.1,
        "confidence_floor": 0.2,
        "calibration": {"mode": "mondrian", "groups": "popularity_decile",
                         "conformal_method": "cqr"},
    }

    proc = ROOT / "processed" / ds / "interactions.parquet"
    splits = make_splits(proc, val_frac=0.1, seed=seed)
    print(f"[data] users={splits.num_users}  items={splits.num_items}  "
          f"calib_users={len(splits.calib_users)}")

    cache = build_cache(ds, reward_cfg["topk_keywords"])
    sim = FrozenSimulator(cache, device=device)
    hist = user_sequences(splits.train)
    profiles = _build_user_profiles(sim, hist)

    alphas = [0.05, 0.10, 0.20]
    rows = []
    for a in alphas:
        gate, _, _, _ = fit_gate_for_alpha(sim, profiles, splits.val,
                                            splits.test, splits.calib_users,
                                            reward_cfg, device, a)
        cov, n, _ = eval_coverage_on_test(sim, profiles, splits.test,
                                           splits.calib_users, reward_cfg,
                                           device, gate.q_hat)
        target = 1.0 - a
        lower = 1.0 - a - 0.05
        status = "OK" if cov >= lower else "ABORT"
        print(f"[alpha={a:.2f}]  q_hat={gate.q_hat:.4f}  u*={gate.u_star:.4f}  "
              f"n_test={n}  empirical_coverage={cov:.4f}  target>={target:.2f}  "
              f"lower_bound={lower:.2f}  status={status}")
        rows.append({"alpha": a, "q_hat": gate.q_hat, "u_star": gate.u_star,
                     "n_test": n, "coverage": cov, "target": target,
                     "lower_bound": lower, "status": status})

    out = {
        "dataset": ds,
        "seed": seed,
        "n_calib": int(len(splits.calib_users)),
        "rows": rows,
        "note": "Coverage = Pr(|r_sim - r_real| <= q_hat). Fit on calib-user "
                "val interactions; evaluated on test-set interactions from "
                "non-calib users (held-out). Proposition 1 predicts "
                "coverage >= 1 - alpha under exchangeability.",
    }
    out_json = ROOT / "paper_rewrite" / "data" / "coverage_validation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {out_json}")

    # LaTeX table
    out_tex = ROOT / "paper_rewrite" / "data" / "coverage_table.tex"
    lines = [
        "% Generated by scripts/_phase2_coverage_sweep.py — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Coverage validation of the conformal gate on ML-1M "
        "(IQL, seed 42). Calibration set: 10\\% of users held out at split time; "
        "calibration samples taken from their validation interactions. "
        "Test set: held-out last-interaction of non-calibration users. "
        "Proposition~\\ref{prop:cov} predicts empirical coverage "
        "$\\Pr(|r_{\\mathrm{sim}}-r_{\\mathrm{real}}| \\leq \\hat q_\\alpha) \\geq 1-\\alpha$.}",
        "\\label{tab:coverage}",
        "\\small",
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "$\\alpha$ & target $\\geq 1{-}\\alpha$ & empirical coverage & $\\hat q_\\alpha$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        a = r["alpha"]; cov = r["coverage"]; q = r["q_hat"]; t = r["target"]
        lines.append(f"{a:.2f} & {t:.2f} & {cov:.3f} & {q:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {out_tex}")

    aborts = [r for r in rows if r["status"] == "ABORT"]
    if aborts:
        print("\n[ABORT] the following alpha(s) had coverage below 1-alpha-0.05:")
        for r in aborts:
            print(f"  alpha={r['alpha']}  cov={r['coverage']:.4f}  "
                  f"lower={r['lower_bound']:.4f}")
        sys.exit(2)

    print("\n[OK] all alpha values meet empirical coverage >= 1 - alpha - 0.05.")


if __name__ == "__main__":
    main()
