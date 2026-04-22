"""Re-evaluate sim-real gap with stratified / bias-corrected metrics.

The gap itself depends only on (sim, reward_cfg, splits), not on the trained
Q-head — so we rebuild the simulator from cache and recompute without any RL.

    py -3.12 scripts/rerun_sim_real_gap.py --dataset movielens-1m
    py -3.12 scripts/rerun_sim_real_gap.py --dataset amazon-videogames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import make_splits, user_sequences
from src.eval.sim_real_gap import sim_real_gap
from src.simulator.cache import build_cache
from src.simulator.frozen_sim import FrozenSimulator
from src.train.loop import _build_user_profiles
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None,
                    help="output path (default: runs/sim_real_gap_<dataset>.json)")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    proc = ROOT / "processed" / args.dataset / "interactions.parquet"
    splits = make_splits(proc, val_frac=cfg["dataset"]["val_frac"],
                          seed=cfg["seed"])
    hist = user_sequences(splits.train)

    cache = build_cache(args.dataset, cfg["reward"]["topk_keywords"])
    sim = FrozenSimulator(cache, device=device)
    profiles = _build_user_profiles(sim, hist)

    out_path = ROOT / (args.out or f"runs/sim_real_gap_{args.dataset}.json")
    gap = sim_real_gap(sim=sim, profiles=profiles, splits=splits,
                       reward_cfg=cfg["reward"], device=device,
                       out_path=out_path)

    print(f"\n[done] wrote {out_path}")
    print("\n--- summary (lower = better) ---")
    for name, r in gap.items():
        print(f"  {name:18s}  MAE={r['MAE']:.4f}  MAE_highU={r['MAE_highU']:.4f}  "
              f"MAE_unbiased={r['MAE_unbiased']:.4f}  "
              f"corr|gap|_u_epi={r['corr_|gap|_u_epi']:+.4f}")


if __name__ == "__main__":
    main()
