"""Run one v2 experiment (IQL + UG-MORS v2 + full-rank eval).

Usage (bash):
    py -3.12 scripts/run_experiment_v2.py \\
      --config configs/v2.yaml \\
      --override dataset.name=movielens-1m reward.name=ug_mors_v2 \\
      --out runs_v2/ml1m-iql-ugmorsv2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make src/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.train.loop_v2 import train_v2
from src.utils.config import deep_update


def _parse_override(overrides):
    """Parse CLI overrides of the form section.key=value into a nested dict."""
    out: dict = {}
    for o in overrides:
        k, v = o.split("=", 1)
        try:
            v_parsed = json.loads(v)
        except json.JSONDecodeError:
            v_parsed = v
        keys = k.split(".")
        cur = out
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v_parsed
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg = deep_update(cfg, _parse_override(args.override))

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    res = train_v2(cfg, args.out)
    with open(args.out / "final.json", "w") as f:
        json.dump({"best_ndcg": res["best_ndcg"],
                   "last_metrics": res["history"][-1] if res["history"] else {},
                   "sim_real_gap": res["gap"]}, f, indent=2)


if __name__ == "__main__":
    main()
