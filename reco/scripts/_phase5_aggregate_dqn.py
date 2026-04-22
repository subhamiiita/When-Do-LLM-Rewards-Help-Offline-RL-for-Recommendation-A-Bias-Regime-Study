"""Phase 5 companion: aggregate multi-seed DQN results from RecSYS-Codes.

Reads RecSYS-Codes/results/runs/{dataset}/dqn/{variant}/seed_{0,1,2}/
and reports NDCG@10 mean +/- std + lift under both 99-neg (simreal_test.json)
and full-rank (simreal_test_fullrank.json) protocols.

Outputs:
  paper_rewrite/data/multiseed_dqn.json
  paper_rewrite/data/multiseed_dqn_summary.md
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "RecSYS-Codes"
RUNS = REC / "results" / "runs"
OUT_DIR = ROOT / "paper_rewrite" / "data"


def read_ndcg(p: Path, fullrank: bool) -> float | None:
    name = "simreal_test_fullrank.json" if fullrank else "simreal_test.json"
    fp = p / name
    if not fp.exists():
        return None
    with open(fp) as f:
        d = json.load(f)
    return d.get("ndcg_at_10")


def gather_ndcg(paths: list[Path], fullrank: bool) -> dict:
    vals = [read_ndcg(p, fullrank) for p in paths]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0}
    out = {"n": len(vals), "values": vals, "mean": statistics.mean(vals)}
    out["std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    seeds = [0, 1, 2]
    datasets = ("ml1m", "yelp", "videogames")
    variants = ("baseline_vote", "ug_mors")

    summary = {}
    for ds in datasets:
        summary[ds] = {}
        for var in variants:
            paths = [RUNS / ds / "dqn" / var / f"seed_{s}" for s in seeds]
            summary[ds][var] = {
                "99neg": gather_ndcg(paths, fullrank=False),
                "fullrank": gather_ndcg(paths, fullrank=True),
            }
        # lift
        for protocol in ("99neg", "fullrank"):
            bv = summary[ds]["baseline_vote"][protocol]
            um = summary[ds]["ug_mors"][protocol]
            if bv.get("n") and um.get("n") and bv["mean"] > 1e-9:
                lift = (um["mean"] - bv["mean"]) / bv["mean"] * 100
                summary[ds][f"lift_{protocol}_pct"] = lift

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "multiseed_dqn.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["# Multi-seed DQN aggregate", ""]
    for ds in datasets:
        lines.append(f"## {ds}")
        for var in variants:
            for protocol in ("99neg", "fullrank"):
                v = summary[ds][var][protocol]
                if v.get("n"):
                    lines.append(f"- **{var} / {protocol}**: "
                                  f"{v['mean']:.4f} +/- {v['std']:.4f} "
                                  f"(n={v['n']}, vals={[round(x, 4) for x in v['values']]})")
                else:
                    lines.append(f"- **{var} / {protocol}**: missing")
        for protocol in ("99neg", "fullrank"):
            key = f"lift_{protocol}_pct"
            if key in summary[ds]:
                lines.append(f"- **lift ({protocol})**: {summary[ds][key]:+.1f}%")
        lines.append("")
    (OUT_DIR / "multiseed_dqn_summary.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    print(f"[write] {OUT_DIR / 'multiseed_dqn.json'}")
    print(f"[write] {OUT_DIR / 'multiseed_dqn_summary.md'}")


if __name__ == "__main__":
    main()
