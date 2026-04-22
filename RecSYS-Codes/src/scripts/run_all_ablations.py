"""Orchestrator: runs the full experimental grid.

First pass (fast iteration):
  3 datasets × 4 agents × 4 key reward variants × 1 seed × 50k steps
  = 48 RL runs, then SimReal eval on each.

Writes per-run JSONL metrics + a summary CSV at results/tables/simreal.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.common.config import project_root
from src.eval.simreal import evaluate_run
from src.train.train_rl import AGENTS, train


DATASETS = ("ml1m", "videogames", "yelp")

# Key variants for first pass. Add ablation variants later once the big-four results look right.
KEY_VARIANTS = ("baseline_vote", "naive_continuous", "ug_mors", "ug_pbrs")
ABLATION_VARIANTS = ("ug_mors_fixed", "ug_mors_no_div", "ug_mors_no_per", "ug_mors_no_ret")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--agents", nargs="+", default=list(AGENTS))
    ap.add_argument("--variants", nargs="+", default=list(KEY_VARIANTS))
    ap.add_argument("--ablations", action="store_true", help="Also run component ablations")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    variants = list(args.variants)
    if args.ablations:
        variants = list(variants) + list(ABLATION_VARIANTS)

    rows = []
    total = len(args.datasets) * len(args.agents) * len(variants) * len(args.seeds)
    print(f"[run_all] scheduling {total} runs...")
    i = 0
    t0 = time.time()
    for ds in args.datasets:
        for ag in args.agents:
            for var in variants:
                for seed in args.seeds:
                    i += 1
                    run_dir = project_root() / f"results/runs/{ds}/{ag}/{var}/seed_{seed}"
                    sim_file = run_dir / "simreal_test.json"
                    if args.skip_existing and sim_file.exists():
                        print(f"[{i}/{total}] skip existing {sim_file}")
                        with sim_file.open() as f:
                            rows.append(json.load(f))
                        continue
                    print(f"[{i}/{total}] train {ds}/{ag}/{var}/seed{seed} steps={args.steps}")
                    try:
                        train(ds, ag, var, seed, args.steps, run_dir)
                        out = evaluate_run(ds, ag, var, seed, split="test")
                        rows.append(out)
                    except Exception as e:
                        print(f"[{i}/{total}] FAILED: {e}")
                        rows.append(dict(dataset=ds, agent=ag, variant=var, seed=seed, error=str(e)))

    dt = (time.time() - t0) / 60
    csv_path = project_root() / "results/tables/simreal.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "agent", "variant", "seed", "ndcg_at_10", "hr_at_10", "n_eval", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[run_all] total wall {dt:.1f} min → {csv_path}")


if __name__ == "__main__":
    main()
