"""Phase 5: aggregate multi-seed IQL results into paper tables.

Reads all runs_v2/grid/.../seed*/final.json + runs_v2/R_llm_init_*/final.json
and computes mean +/- std per cell. Also computes the Yelp-lift and
representation-share numbers that need refreshing.

Outputs:
  paper_rewrite/data/multiseed_iql.json  -- raw aggregate
  paper_rewrite/data/multiseed_iql_summary.md -- human-readable

Does NOT edit results_tables.tex directly; those edits happen in a second
pass once this aggregate is reviewed.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_v2"
OUT_DIR = ROOT / "paper_rewrite" / "data"


def best_ndcg(p: Path) -> float | None:
    fp = p / "final.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        d = json.load(f)
    return float(d.get("best_ndcg", float("nan")))


def mae_highu(p: Path) -> float | None:
    fp = p / "final.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        d = json.load(f)
    g = d.get("sim_real_gap", {}).get("ug_mors", {})
    return g.get("MAE_highU")


def mae_highu_naive(p: Path) -> float | None:
    fp = p / "final.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        d = json.load(f)
    g = d.get("sim_real_gap", {}).get("naive_continuous", {})
    return g.get("MAE_highU")


def gather(paths: list[Path]) -> dict:
    vals = [best_ndcg(p) for p in paths]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0}
    out = {"n": len(vals), "values": vals, "mean": statistics.mean(vals)}
    if len(vals) > 1:
        out["std"] = statistics.stdev(vals)
    else:
        out["std"] = 0.0
    return out


def gather_mae(paths: list[Path], kind: str = "ug_mors") -> dict:
    fn = mae_highu if kind == "ug_mors" else mae_highu_naive
    vals = [fn(p) for p in paths]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    out = {"n": len(vals), "values": vals, "mean": statistics.mean(vals)}
    if len(vals) > 1:
        out["std"] = statistics.stdev(vals)
    else:
        out["std"] = 0.0
    return out


def cell_paths(kind: str, ds: str, seeds: list[int]) -> list[Path]:
    if kind == "R_warm":
        return [RUNS / "grid" / ds / "R_warm" / f"seed{s}" for s in seeds]
    if kind == "R4":
        return [RUNS / "grid" / ds / "R4_ugmv2_BC" / f"seed{s}" for s in seeds]
    if kind == "R0":
        return [RUNS / "grid" / ds / "R0_naive" / f"seed{s}" for s in seeds]
    if kind == "R3":
        return [RUNS / "grid" / ds / "R3_ugmv2_noBC" / f"seed{s}" for s in seeds]
    if kind == "R_llm_init":
        short = {"movielens-1m": "ml1m", "yelp": "yelp"}[ds]
        return [RUNS / f"R_llm_init_{short}_seed{s}" for s in seeds]
    raise ValueError(kind)


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p-value for the sign test (null p=0.5)."""
    from math import comb
    k = max(wins, n - wins)
    p = 0.0
    for i in range(k, n + 1):
        p += comb(n, i) * (0.5 ** n)
    return 2 * p if k > n - k else 1.0


def main():
    # Pre-existing ML-1M R_warm, R4, R0, R3 seeds from the original grid.
    ml1m_grid_seeds = [42, 7, 123]
    # Yelp and Amazon VG grid cells: seed 42 pre-existing, 43 and 44 from Phase 1.
    # ML-1M R_llm_init: seed 42 pre-existing, 43 and 44 from Phase 1.
    p1_seeds = [42, 43, 44]

    summary: dict = {}
    for ds in ("movielens-1m", "yelp", "amazon-videogames"):
        ds_key = {"movielens-1m": "ml1m", "yelp": "yelp",
                   "amazon-videogames": "videogames"}[ds]
        r4_seeds = ml1m_grid_seeds if ds == "movielens-1m" else p1_seeds
        summary[ds_key] = {"seeds_grid": r4_seeds, "seeds_llm_init": p1_seeds}
        summary[ds_key]["R_warm"] = gather(cell_paths("R_warm", ds, r4_seeds))
        summary[ds_key]["R4_ugmv2_BC"] = gather(cell_paths("R4", ds, r4_seeds))
        if ds == "movielens-1m":
            summary[ds_key]["R0_naive"] = gather(cell_paths("R0", ds, ml1m_grid_seeds))
            summary[ds_key]["R3_ugmv2_noBC"] = gather(cell_paths("R3", ds, ml1m_grid_seeds))
        else:
            if ds == "amazon-videogames":
                summary[ds_key]["R0_naive"] = gather(cell_paths("R0", ds, [42]))
        if ds in ("movielens-1m", "yelp"):
            summary[ds_key]["R_llm_init"] = gather(cell_paths("R_llm_init", ds, p1_seeds))
        summary[ds_key]["MAE_highU_ug_mors"] = gather_mae(
            cell_paths("R4", ds, r4_seeds), "ug_mors")
        summary[ds_key]["MAE_highU_naive"] = gather_mae(
            cell_paths("R4", ds, r4_seeds), "naive_continuous")

    # --- Derived quantities the paper quotes and may need updating ---
    derived = {}
    # Yelp RL-over-Supervised lift (RL-gated-BC vs R_warm)
    y = summary["yelp"]
    if y["R4_ugmv2_BC"]["n"] and y["R_warm"]["n"]:
        lift = (y["R4_ugmv2_BC"]["mean"] - y["R_warm"]["mean"]) / y["R_warm"]["mean"]
        derived["yelp_R4_over_R_warm_lift_pct"] = lift * 100
    # Representation share on Yelp (LLM-init over Supervised, as fraction of R4 - Supervised)
    if y.get("R_llm_init", {}).get("n") and y["R4_ugmv2_BC"]["n"] and y["R_warm"]["n"]:
        rep = y["R_llm_init"]["mean"] - y["R_warm"]["mean"]
        tot = y["R4_ugmv2_BC"]["mean"] - y["R_warm"]["mean"]
        if tot > 1e-9:
            derived["yelp_rep_share_pct"] = rep / tot * 100
    # ML-1M representation share
    m = summary["ml1m"]
    if m.get("R_llm_init", {}).get("n") and m["R4_ugmv2_BC"]["n"] and m["R_warm"]["n"]:
        rep = m["R_llm_init"]["mean"] - m["R_warm"]["mean"]
        tot = m["R4_ugmv2_BC"]["mean"] - m["R_warm"]["mean"]
        if tot > 1e-9:
            derived["ml1m_rep_share_pct"] = rep / tot * 100
    # Amazon VG MAE_highU relative change (R4 ug_mors vs naive)
    a = summary["videogames"]
    if a["MAE_highU_ug_mors"]["n"] and a["MAE_highU_naive"]["n"]:
        rel = (a["MAE_highU_ug_mors"]["mean"] - a["MAE_highU_naive"]["mean"]) \
              / a["MAE_highU_naive"]["mean"]
        derived["amazon_vg_mae_highu_rel_change_pct"] = rel * 100
    # ML-1M MAE_highU relative change (same formula)
    if m["MAE_highU_ug_mors"]["n"] and m["MAE_highU_naive"]["n"]:
        rel = (m["MAE_highU_ug_mors"]["mean"] - m["MAE_highU_naive"]["mean"]) \
              / m["MAE_highU_naive"]["mean"]
        derived["ml1m_mae_highu_rel_change_pct"] = rel * 100
    if y["MAE_highU_ug_mors"]["n"] and y["MAE_highU_naive"]["n"]:
        rel = (y["MAE_highU_ug_mors"]["mean"] - y["MAE_highU_naive"]["mean"]) \
              / y["MAE_highU_naive"]["mean"]
        derived["yelp_mae_highu_rel_change_pct"] = rel * 100

    # --- Sign-test for R4 > R_warm across seeds (ML-1M) ---
    tests = {}
    for dsk, cell in (("ml1m", summary["ml1m"]), ("yelp", summary["yelp"]),
                     ("videogames", summary["videogames"])):
        if cell["R4_ugmv2_BC"].get("n", 0) >= 2 and cell["R_warm"].get("n", 0) >= 2:
            n = min(cell["R4_ugmv2_BC"]["n"], cell["R_warm"]["n"])
            pairs = list(zip(cell["R4_ugmv2_BC"]["values"][:n],
                              cell["R_warm"]["values"][:n]))
            wins = sum(1 for a, b in pairs if a > b)
            tests[dsk] = {"n_pairs": n, "wins_R4_over_Rwarm": wins,
                           "sign_test_two_sided_p": sign_test_p(wins, n)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {"summary": summary, "derived": derived, "sign_tests": tests}
    with open(OUT_DIR / "multiseed_iql.json", "w") as f:
        json.dump(raw, f, indent=2)

    lines = ["# Multi-seed IQL aggregate", ""]
    for dsk in ("ml1m", "yelp", "videogames"):
        lines.append(f"## {dsk}")
        for k, v in summary[dsk].items():
            if k in ("seeds_grid", "seeds_llm_init"):
                lines.append(f"- {k}: {v}")
                continue
            if not isinstance(v, dict) or v.get("n", 0) == 0:
                lines.append(f"- **{k}**: missing")
                continue
            if k.startswith("MAE_"):
                lines.append(f"- **{k}**: "
                              f"{v['mean']:.4f} +/- {v['std']:.4f} (n={v['n']})")
            else:
                lines.append(f"- **{k}**: "
                              f"{v['mean']:.4f} +/- {v['std']:.4f} (n={v['n']}, "
                              f"vals={[round(x, 4) for x in v['values']]})")
        lines.append("")
    lines.append("## Derived")
    for k, v in derived.items():
        lines.append(f"- {k}: {v:.2f}")
    lines.append("")
    lines.append("## Sign tests (R4 > R_warm)")
    for dsk, t in tests.items():
        lines.append(f"- {dsk}: {t['wins_R4_over_Rwarm']}/{t['n_pairs']} wins, "
                      f"two-sided p={t['sign_test_two_sided_p']:.4f}")
    (OUT_DIR / "multiseed_iql_summary.md").write_text("\n".join(lines) + "\n",
                                                         encoding="utf-8")
    print(f"[write] {OUT_DIR / 'multiseed_iql.json'}")
    print(f"[write] {OUT_DIR / 'multiseed_iql_summary.md'}")


if __name__ == "__main__":
    main()
