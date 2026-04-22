"""DIAGNOSTIC: prove that the v1 RL phase has near-zero effect on the policy.

Runs three configurations back-to-back on MovieLens-1M / DQN:
    (A) warmup-only       — 0 RL epochs, just the supervised warmup
    (B) UG-MORS + freeze  — the current published config (10 RL epochs)
    (C) naive + freeze    — same as (B) but with a different reward

Expected: A, B, and C report NDCG@10 within 0.002 of each other. This
confirms that the reward does nothing and reshaping it cannot possibly
produce a publishable contrast in the current architecture.

Usage:
    py -3.12 scripts/verify_v1_diagnosis.py --out runs/diag
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.train.loop import train


BASE = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())


def _override(base: dict, **kw) -> dict:
    out = copy.deepcopy(base)
    for k, v in kw.items():
        section, sub = k.split(".")
        out[section][sub] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("runs/diag"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # (A) warmup only
    cfg_A = _override(BASE,
                      **{"agent.epochs": 0, "agent.warmup_epochs": 3,
                         "reward.name": "binary", "dataset.name": "movielens-1m"})
    outA = args.out / "A_warmup_only"
    outA.mkdir(parents=True, exist_ok=True)
    resA = train(cfg_A, outA)

    # (B) UG-MORS + frozen encoder
    cfg_B = _override(BASE,
                      **{"agent.epochs": 10, "agent.freeze_encoder": True,
                         "reward.name": "ug_mors", "dataset.name": "movielens-1m"})
    outB = args.out / "B_ugmors_frozen"
    outB.mkdir(parents=True, exist_ok=True)
    resB = train(cfg_B, outB)

    # (C) naive + frozen encoder
    cfg_C = _override(BASE,
                      **{"agent.epochs": 10, "agent.freeze_encoder": True,
                         "reward.name": "naive_continuous", "dataset.name": "movielens-1m"})
    outC = args.out / "C_naive_frozen"
    outC.mkdir(parents=True, exist_ok=True)
    resC = train(cfg_C, outC)

    summary = {
        "A_warmup_only_NDCG@10":   resA["history"][-1]["NDCG@10"] if resA["history"] else None,
        "B_ugmors_frozen_NDCG@10": resB["history"][-1]["NDCG@10"],
        "C_naive_frozen_NDCG@10":  resC["history"][-1]["NDCG@10"],
    }
    with open(args.out / "diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== DIAGNOSIS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nIf all three numbers are within 0.002 of each other, the v1 RL phase")
    print("is a no-op and you must switch to the v2 architecture in src/train/loop_v2.py")


if __name__ == "__main__":
    main()
