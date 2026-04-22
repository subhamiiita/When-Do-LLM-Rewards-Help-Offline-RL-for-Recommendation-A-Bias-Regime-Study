"""Compute sim-real gap for a seed WITHOUT training.

Useful for:
  * seed-replication sanity checks on MAE_highU / corr(|gap|, u_epi) without
    a 70-min retrain — the gap is a function of (split_seed, reward_cfg), not
    the trained agent's weights.
  * post-hoc verification during the 45-run grid: if one grid cell reports a
    suspicious MAE_highU, re-run this standalone in ~2 min.

Usage:
    py -3.12 scripts/gap_only.py --config configs/v2.yaml --seed 42 \\
        --out runs_v2/gap-only/ml1m-seed42

If --gate-pt is provided, loads the fitted ConformalGate from that path instead
of refitting — so the measurement uses the exact gate the trained policy saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data.splits import make_splits, user_sequences
from src.eval.sim_real_gap import sim_real_gap
from src.rewards.conformal import ConformalGate
from src.simulator.cache import build_cache
from src.simulator.frozen_sim import FrozenSimulator
from src.train.loop import _build_user_profiles
from src.utils.config import deep_update
from src.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gate-pt", type=Path, default=None,
                   help="path to a saved ConformalGate state_dict; if omitted, gate is refit")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["seed"] = args.seed
    # apply overrides (same format as run_experiment_v2)
    for o in args.override:
        k, v = o.split("=", 1)
        try:
            v_parsed = json.loads(v)
        except json.JSONDecodeError:
            v_parsed = v
        keys = k.split(".")
        cur = cfg
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v_parsed

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    ds = cfg["dataset"]["name"]
    proc = ROOT / "processed" / ds / "interactions.parquet"
    splits = make_splits(proc, val_frac=cfg["dataset"]["val_frac"], seed=cfg["seed"])
    hist = user_sequences(splits.train)

    cache = build_cache(ds, cfg["reward"]["topk_keywords"])
    sim = FrozenSimulator(cache, device=device)
    profiles = _build_user_profiles(sim, hist)

    gate = None
    if args.gate_pt is not None:
        state = torch.load(args.gate_pt, map_location="cpu")
        gate = ConformalGate(alpha=cfg["reward"]["crc_alpha"],
                             temperature=cfg["reward"]["gate_temperature"],
                             confidence_floor=cfg["reward"]["confidence_floor"])
        gate.load_state_dict(state)
        print(f"[gate] loaded from {args.gate_pt}: u*={gate.u_star:.4f} q_hat={gate.q_hat:.4f} T={gate.T:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = sim_real_gap(sim=sim, profiles=profiles, splits=splits,
                           reward_cfg=cfg["reward"], device=device,
                           out_path=args.out / "sim_real_gap.json",
                           gate=gate)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
