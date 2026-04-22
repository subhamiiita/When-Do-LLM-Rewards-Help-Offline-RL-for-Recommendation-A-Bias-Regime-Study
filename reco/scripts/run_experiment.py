"""Run a single (dataset, agent, reward) experiment.

    py -3.12 scripts/run_experiment.py \
        --config configs/base.yaml \
        --override dataset.name=movielens-1m agent.name=dqn reward.name=ug_mors \
        --out runs/ml1m-dqn-ug_mors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.loop import train
from src.utils.config import load_yaml


def _parse_override(overrides: list[str]) -> dict:
    out: dict = {}
    for o in overrides:
        k, v = o.split("=", 1)
        # coerce
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


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    cfg = _deep_merge(cfg, _parse_override(args.override))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    res = train(cfg, out_dir)
    with open(out_dir / "final.json", "w") as f:
        json.dump({"best_ndcg": res["best_ndcg"],
                   "last_metrics": res["history"][-1] if res["history"] else {},
                   "sim_real_gap": res["gap"]}, f, indent=2)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
