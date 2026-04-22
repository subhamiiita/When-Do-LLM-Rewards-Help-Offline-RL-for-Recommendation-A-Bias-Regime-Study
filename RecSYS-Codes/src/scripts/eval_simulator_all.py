"""Run simulator_metrics eval across the existing 48 trained runs.

Writes results/tables/simulator_metrics.csv so we can compare to the base paper's
Table 2 (A. Rwd / T. Rwd / Liking%).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.common.config import project_root
from src.eval.simulator_metrics import evaluate_simulator_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n_eval_episodes", type=int, default=200)
    args = ap.parse_args()

    runs_dir = project_root() / "results/runs"
    rows = []
    t0 = time.time()
    # Collect existing (dataset, agent, variant, seed) tuples that have model.pt
    candidates = []
    for ds_dir in sorted(runs_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        for ag_dir in sorted(ds_dir.iterdir()):
            if not ag_dir.is_dir():
                continue
            ag = ag_dir.name
            for var_dir in sorted(ag_dir.iterdir()):
                if not var_dir.is_dir():
                    continue
                var = var_dir.name
                for seed_dir in sorted(var_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    seed = int(seed_dir.name.split("_")[1])
                    if (seed_dir / "model.pt").exists():
                        candidates.append((ds, ag, var, seed))

    print(f"Found {len(candidates)} trained runs")
    for i, (ds, ag, var, seed) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {ds}/{ag}/{var}/seed{seed}")
        try:
            out = evaluate_simulator_metrics(
                ds, ag, var, seed,
                horizon=args.horizon,
                n_eval_episodes=args.n_eval_episodes,
            )
            rows.append(out)
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append(dict(dataset=ds, agent=ag, variant=var, seed=seed, error=str(e)))

    out_csv = project_root() / "results/tables/simulator_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "agent", "variant", "seed", "horizon", "n_eval_episodes", "n_eval_steps", "A_Rwd", "T_Rwd", "Liking_pct", "error"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
