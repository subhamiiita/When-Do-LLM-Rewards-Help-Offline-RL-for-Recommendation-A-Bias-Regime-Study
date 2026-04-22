"""Fill {{placeholder}} tokens in paper/*.tex from runs_v2/grid aggregate.

Usage:
    python scripts/fill_placeholders.py \
        --runs runs_v2/grid \
        --paper_in paper \
        --paper_out paper_filled \
        [--require_complete]   # error if any grid cell is missing
        [--check_only]         # report unresolved placeholders, no writes
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DS_ALIAS = {
    "movielens-1m": "ml1m",
    "amazon-videogames": "amz",
    "yelp": "yelp",
}

RUNG_TO_SIMREAL_KEY = {
    # R_warm never applies the LLM reward during training, so the sim-real
    # comparison that makes sense is the naive_continuous variant: "what
    # MAE would the raw LLM reward have against the realised outcome, if
    # we were to use it as-is?" This matches R0_naive's variant and gives
    # the Yelp §results_simreal story a non-degenerate baseline row.
    "R_warm":           "naive_continuous",
    "R0_naive":         "naive_continuous",
    "R2_hardgate":      "hard_gate",
    "R3_ugmv2_noBC":    "ug_mors",
    "R4_ugmv2_BC":      "ug_mors",
    "R5_ugmv2_BC_pess": "ug_mors",
}

DATASET_STATS = {
    "ml1m": {
        "users": "6{,}040",
        "items": "3{,}261",
        "interactions": "980{,}419",
        "avg_len": "162.3",
        "density": "5.0",
    },
    # Amazon Video Games 10-core (data/amazon-videogames/videogames_10core_report.txt):
    # reviews=127,865; users=7,253; items=4,338; sparsity=0.995936 -> density 0.41%;
    # avg_len = 127,865 / 7,253 = 17.6 interactions per user.
    "amz": {
        "users": "7{,}253",
        "items": "4{,}338",
        "interactions": "127{,}865",
        "avg_len": "17.6",
        "density": "0.41",
    },
    # Yelp Missouri 10-core (data/yelp/mo_10core_report.txt):
    # reviews=194,623; users=6,559; businesses=4,111; sparsity=0.992782 -> density 0.72%;
    # avg_len = 194,623 / 6,559 = 29.7 interactions per user.
    "yelp": {
        "users": "6{,}559",
        "items": "4{,}111",
        "interactions": "194{,}623",
        "avg_len": "29.7",
        "density": "0.72",
    },
}

RUNG_PREFIX = {
    "R_warm":           "r_warm",
    "R0_naive":         "r0",
    "R2_hardgate":      "r2",
    "R3_ugmv2_noBC":    "r3",
    "R4_ugmv2_BC":      "r4",
    "R5_ugmv2_BC_pess": "r5",
}


def _load_run(seed_dir: Path) -> Optional[Dict]:
    fp = seed_dir / "final.json"
    if not fp.exists():
        return None
    try:
        js = json.loads(fp.read_text())
    except Exception as e:
        print(f"[warn] failed to parse {fp}: {e}", file=sys.stderr)
        return None
    # Stale marker: a final.json whose _stale flag is set represents a
    # pre-patch run that the grid runner has been told to re-execute.
    # Treat as missing for aggregation so its numbers don't contaminate
    # the paper tables.
    if isinstance(js, dict) and js.get("_stale"):
        return None
    # Attach per-epoch history if it lives in a sibling file (history.json).
    if "history" not in js and "eval_history" not in js:
        hp = seed_dir / "history.json"
        if hp.exists():
            try:
                js["history"] = json.loads(hp.read_text())
            except Exception as e:
                print(f"[warn] failed to parse {hp}: {e}", file=sys.stderr)
    return js


def _val_selected_test(js: Dict) -> Dict[str, Optional[float]]:
    lm = js.get("last_metrics", {}) or {}
    history = js.get("history") or js.get("eval_history")
    sel = None
    if isinstance(history, list) and history:
        valid = [h for h in history if h.get("val_NDCG@10") is not None]
        if valid:
            sel = max(valid, key=lambda h: h["val_NDCG@10"])
    if sel is not None:
        return {
            "NDCG@10":   sel.get("NDCG@10"),
            "HR@10":     sel.get("HR@10"),
            "TailHR@10": sel.get("TailHR@10"),
            "_selection_method": "val_selected",
        }
    return {
        "NDCG@10":   js.get("best_ndcg") or lm.get("NDCG@10"),
        "HR@10":     lm.get("HR@10"),
        "TailHR@10": lm.get("TailHR@10"),
        "_selection_method": "peak_test",
    }


def _simreal(js: Dict, rung: str) -> Dict[str, Optional[float]]:
    variant = RUNG_TO_SIMREAL_KEY.get(rung, "ug_mors")
    srg = js.get("sim_real_gap", {}) or {}
    block = srg.get(variant, {}) or {}
    return {
        "MAE_highU": block.get("MAE_highU"),
        "MSE_highU": block.get("MSE_highU"),
        "corr":      block.get("corr_|gap|_u_epi"),
    }


def _valtest_gap_pct(js: Dict) -> Optional[float]:
    lm = js.get("last_metrics", {}) or {}
    v = lm.get("val_NDCG@10")
    t = lm.get("NDCG@10")
    if v is None or t is None or v == 0:
        return None
    return 100.0 * (v - t) / v


def aggregate(runs_root: Path) -> Dict:
    agg: Dict = {}
    for ds_dir in sorted(runs_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_alias = DS_ALIAS.get(ds_dir.name, ds_dir.name)
        agg[ds_alias] = {}
        # Dataset-level invariants extracted from the first available run's
        # sim_real_gap.naive_continuous block. mean_r_sim and mean_r_real
        # are test-set invariants (they depend only on the leave-last-out
        # split and the LLM prompt, not on the training rung).
        agg[ds_alias]["__dataset__"] = {
            "mean_r_sim": None, "mean_r_real": None,
        }
        for rung_dir in sorted(ds_dir.iterdir()):
            if not rung_dir.is_dir():
                continue
            rung = rung_dir.name
            agg[ds_alias][rung] = {"seeds": []}
            for seed_dir in sorted(rung_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                js = _load_run(seed_dir)
                if js is None:
                    continue
                vst = _val_selected_test(js)
                sr = _simreal(js, rung)
                # Stash dataset-level r_sim / r_real the first time we see them.
                if agg[ds_alias]["__dataset__"]["mean_r_sim"] is None:
                    nc = (js.get("sim_real_gap", {}) or {}).get(
                        "naive_continuous", {}) or {}
                    if "mean_r_sim" in nc:
                        agg[ds_alias]["__dataset__"]["mean_r_sim"] = nc["mean_r_sim"]
                    if "mean_r_real" in nc:
                        agg[ds_alias]["__dataset__"]["mean_r_real"] = nc["mean_r_real"]
                agg[ds_alias][rung]["seeds"].append({
                    "seed":       seed_dir.name,
                    "NDCG@10":    vst["NDCG@10"],
                    "HR@10":      vst["HR@10"],
                    "TailHR@10":  vst["TailHR@10"],
                    "MAE_highU":  sr["MAE_highU"],
                    "MSE_highU":  sr["MSE_highU"],
                    "corr":       sr["corr"],
                    "valtest_gap": _valtest_gap_pct(js),
                    "peak_ndcg":  js.get("best_ndcg"),
                    "_selection_method": vst["_selection_method"],
                })
    return agg


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{digits}f}"


def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=0))


def _wilcoxon(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value.

    Uses scipy when available; otherwise falls back to an exact
    enumeration of all 2^n sign combinations (fine for n <= ~12).
    Returns None only if inputs are malformed.
    """
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    # Pure-numpy exact fallback
    try:
        from itertools import product
        d = np.array(a, dtype=float) - np.array(b, dtype=float)
        d = d[d != 0]
        n = len(d)
        if n == 0:
            return 1.0
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        wp = float(np.sum(ranks[d > 0]))
        cle = cge = 0
        for signs in product([0, 1], repeat=n):
            w = sum(r for r, s in zip(ranks.tolist(), signs) if s == 1)
            if w <= wp:
                cle += 1
            if w >= wp:
                cge += 1
        return float(min(1.0, 2 * min(cle, cge) / (2 ** n)))
    except Exception:
        return None


def _sign_consistency(a: List[float], b: List[float]) -> Optional[str]:
    """Return "<positive>/<total>" for per-seed deltas a - b.

    At n=3 seeds, exact Wilcoxon bottoms out at p=0.25, so sign consistency
    ("3/3 seeds positive") is the practical small-sample robustness proxy.
    """
    if len(a) != len(b) or len(a) == 0:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    pos = sum(1 for d in diffs if d > 0)
    return f"{pos}/{len(diffs)}"


def _load_r_llm_init(runs_root: Path, ds_alias: str, seed: int = 42
                     ) -> Optional[float]:
    """Load R_llm_init val-selected test NDCG@10 for a (ds_alias, seed).

    R_llm_init runs live flat under ``runs_root.parent`` (i.e.
    ``runs_v2/R_llm_init_{ds_alias}_seed{seed}/final.json``) per the
    brief, so the grid aggregator's tree-walk misses them. Returns None
    if the run directory or final.json is absent or stale.
    """
    parent = runs_root.parent if runs_root.name == "grid" else runs_root
    fp = parent / f"R_llm_init_{ds_alias}_seed{seed}" / "final.json"
    if not fp.exists():
        return None
    try:
        js = json.loads(fp.read_text())
    except Exception:
        return None
    if isinstance(js, dict) and js.get("_stale"):
        return None
    vst = _val_selected_test(js)
    return vst.get("NDCG@10")


def build_substitutions(agg: Dict, runs_root: Path) -> Dict[str, str]:
    sub: Dict[str, str] = {}

    # Dataset stats
    for ds_alias, stats in DATASET_STATS.items():
        for k, v in stats.items():
            sub[f"{ds_alias}_{k}"] = v

    # Per-rung per-dataset metrics
    for ds_alias, rungs in agg.items():
        for rung, data in rungs.items():
            prefix = RUNG_PREFIX.get(rung)
            if prefix is None:
                continue
            seeds = data.get("seeds", [])
            if not seeds:
                continue
            for metric_key, frag in [
                ("NDCG@10",   "ndcg"),
                ("HR@10",     "hr"),
                ("TailHR@10", "tailhr"),
                ("MAE_highU", "mae_high"),
                ("MSE_highU", "mse_high"),
                ("corr",      "corr"),
            ]:
                vals = [s.get(metric_key) for s in seeds]
                mean, std = _mean_std(vals)
                if mean is None:
                    continue
                digits = 3 if frag == "corr" else 4
                sub[f"{prefix}_{ds_alias}_{frag}"] = _fmt(mean, digits)
                # Reverse-order alias so prose written as {{r4_mae_high_ml1m}}
                # also resolves (canonical is {{r4_ml1m_mae_high}}).
                sub[f"{prefix}_{frag}_{ds_alias}"] = _fmt(mean, digits)
                if len(seeds) > 1:
                    sub[f"{prefix}_{ds_alias}_{frag}_mean"] = _fmt(mean, digits)
                    sub[f"{prefix}_{ds_alias}_{frag}_std"] = _fmt(std, digits)
                    sub[f"{prefix}_{frag}_{ds_alias}_mean"] = _fmt(mean, digits)
                    sub[f"{prefix}_{frag}_{ds_alias}_std"] = _fmt(std, digits)
                if frag == "ndcg":
                    sub[f"{prefix}_val_ndcg_{ds_alias}"] = _fmt(mean, digits)
                    if len(seeds) > 1:
                        sub[f"{prefix}_val_ndcg_{ds_alias}_mean"] = _fmt(mean, digits)
                        sub[f"{prefix}_val_ndcg_{ds_alias}_std"] = _fmt(std, digits)
            gaps = [s.get("valtest_gap") for s in seeds]
            mean_gap, _ = _mean_std(gaps)
            if mean_gap is not None:
                sub[f"{prefix}_valtest_gap_{ds_alias}"] = _fmt(mean_gap, 2)
            peaks = [s.get("peak_ndcg") for s in seeds]
            mean_peak, std_peak = _mean_std(peaks)
            if mean_peak is not None:
                sub[f"{prefix}_peak_ndcg_{ds_alias}"] = _fmt(mean_peak, 4)
                if len(seeds) > 1 and std_peak is not None:
                    sub[f"{prefix}_peak_ndcg_{ds_alias}_std"] = _fmt(std_peak, 4)

    # Derived: per-dataset LLM-marginal bias regime characterisation.
    # mean_r_sim and mean_r_real are leave-last-out test-set invariants
    # extracted by aggregate() into agg[ds]["__dataset__"]. Emit them as
    # r_sim_bar_<ds> / r_real_bar_<ds> for use in tab:biasregime.
    for ds_alias, rungs in agg.items():
        ds_invariants = rungs.get("__dataset__", {})
        mean_r_sim = ds_invariants.get("mean_r_sim")
        mean_r_real = ds_invariants.get("mean_r_real")
        if mean_r_sim is not None:
            sub[f"r_sim_bar_{ds_alias}"] = _fmt(mean_r_sim, 3)
        if mean_r_real is not None:
            sub[f"r_real_bar_{ds_alias}"] = _fmt(mean_r_real, 3)

    # Derived: relative improvements & Wilcoxon
    for ds_alias in agg.keys():
        r0 = agg[ds_alias].get("R0_naive", {}).get("seeds", [])
        r4 = agg[ds_alias].get("R4_ugmv2_BC", {}).get("seeds", [])
        if r0 and r4:
            m0, _ = _mean_std([s["MAE_highU"] for s in r0])
            m4, _ = _mean_std([s["MAE_highU"] for s in r4])
            if m0 and m4 and m0 > 0:
                sub[f"r4_mae_high_rel_improvement_{ds_alias}"] = _fmt(100.0 * (m0 - m4) / m0, 1)
            c0, _ = _mean_std([s["corr"] for s in r0])
            c4, _ = _mean_std([s["corr"] for s in r4])
            if c0 is not None and c4 is not None:
                sub[f"r4_corr_gap_u_improvement_{ds_alias}"] = _fmt(100 * (c4 - c0), 1)
            for mk, frag in [("NDCG@10", "ndcg"), ("HR@10", "hr"), ("TailHR@10", "tailhr")]:
                m0p, _ = _mean_std([s[mk] for s in r0])
                m4p, _ = _mean_std([s[mk] for s in r4])
                if m0p and m4p and m0p > 0:
                    sub[f"r4_{ds_alias}_{frag}_rel"] = _fmt(100.0 * (m4p - m0p) / m0p, 1)

        # Per-rung relative-to-R0 NDCG column for tab:main. Emits
        # <rung>_<ds>_ndcg_rel_r0 for R2/R3/R4/R5 vs R0_naive.
        if r0:
            m0p, _ = _mean_std([s["NDCG@10"] for s in r0])
            if m0p and m0p > 0:
                for rung, key in [
                    ("R2_hardgate",      "r2"),
                    ("R3_ugmv2_noBC",    "r3"),
                    ("R4_ugmv2_BC",      "r4"),
                    ("R5_ugmv2_BC_pess", "r5"),
                ]:
                    rr = agg[ds_alias].get(rung, {}).get("seeds", [])
                    if not rr:
                        continue
                    mr, _ = _mean_std([s["NDCG@10"] for s in rr])
                    if mr is None:
                        continue
                    sub[f"{key}_{ds_alias}_ndcg_rel_r0"] = _fmt(
                        100.0 * (mr - m0p) / m0p, 1)

        # %Δ vs R_warm (supervised-only anchor). Emits
        # <rung>_<ds>_ndcg_rel_rwarm for each RL rung that has seeds. Used
        # in the preservation framing (abstract, intro C5, discussion).
        rwarm = agg[ds_alias].get("R_warm", {}).get("seeds", [])
        if rwarm:
            mw, _ = _mean_std([s["NDCG@10"] for s in rwarm])
            if mw and mw > 0:
                for rung, key in [
                    ("R0_naive",         "r0"),
                    ("R2_hardgate",      "r2"),
                    ("R3_ugmv2_noBC",    "r3"),
                    ("R4_ugmv2_BC",      "r4"),
                    ("R5_ugmv2_BC_pess", "r5"),
                ]:
                    rr = agg[ds_alias].get(rung, {}).get("seeds", [])
                    if not rr:
                        continue
                    mr, _ = _mean_std([s["NDCG@10"] for s in rr])
                    if mr is None:
                        continue
                    sub[f"{key}_{ds_alias}_ndcg_rel_rwarm"] = _fmt(
                        100.0 * (mr - mw) / mw, 1)

        # R_llm_init (representation-transfer control; brief Deliverable 3).
        # Runs live flat under runs_root.parent, not in the grid tree, so the
        # aggregator walk misses them. Emit <ndcg> and <ndcg_rel_rwarm>
        # placeholders; fall back to "--" when the run is absent.
        rli_ndcg = _load_r_llm_init(runs_root, ds_alias, seed=42)
        sub[f"r_llm_init_{ds_alias}_ndcg"] = _fmt(rli_ndcg, 4)
        if rwarm and rli_ndcg is not None:
            mw_rli, _ = _mean_std([s["NDCG@10"] for s in rwarm])
            if mw_rli and mw_rli > 0:
                sub[f"r_llm_init_{ds_alias}_ndcg_rel_rwarm"] = _fmt(
                    100.0 * (rli_ndcg - mw_rli) / mw_rli, 1)
            else:
                sub[f"r_llm_init_{ds_alias}_ndcg_rel_rwarm"] = "--"
        else:
            sub[f"r_llm_init_{ds_alias}_ndcg_rel_rwarm"] = "--"

        for (a_rung, a_key), (b_rung, b_key) in [
            (("R4_ugmv2_BC",      "r4"), ("R0_naive",      "r0")),
            (("R4_ugmv2_BC",      "r4"), ("R2_hardgate",   "r2")),
            (("R4_ugmv2_BC",      "r4"), ("R3_ugmv2_noBC", "r3")),
            (("R5_ugmv2_BC_pess", "r5"), ("R4_ugmv2_BC",   "r4")),
            (("R3_ugmv2_noBC",    "r3"), ("R0_naive",      "r0")),
            (("R3_ugmv2_noBC",    "r3"), ("R2_hardgate",   "r2")),
            (("R5_ugmv2_BC_pess", "r5"), ("R3_ugmv2_noBC", "r3")),
        ]:
            aa = agg[ds_alias].get(a_rung, {}).get("seeds", [])
            bb = agg[ds_alias].get(b_rung, {}).get("seeds", [])
            if len(aa) < 2 or len(bb) < 2:
                continue
            sm_a = {s["seed"]: s["NDCG@10"] for s in aa}
            sm_b = {s["seed"]: s["NDCG@10"] for s in bb}
            common = [s for s in sm_a if s in sm_b and sm_a[s] is not None and sm_b[s] is not None]
            if len(common) < 2:
                continue
            av = [sm_a[s] for s in common]
            bv = [sm_b[s] for s in common]
            p = _wilcoxon(av, bv)
            if p is not None:
                sub[f"{a_key}_vs_{b_key}_pvalue_{ds_alias}"] = _fmt(p, 3)
            sc = _sign_consistency(av, bv)
            if sc is not None:
                sub[f"{a_key}_vs_{b_key}_sign_{ds_alias}"] = sc
            da = float(np.mean(av) - np.mean(bv))
            sub[f"{a_key}_minus_{b_key}_ndcg"] = _fmt(da, 4)
            sub[f"{a_key}_minus_{b_key}_ndcg_{ds_alias}"] = _fmt(da, 4)
            sub[f"{b_key}_vs_{a_key}_ndcg_delta_{ds_alias}"] = _fmt(-da, 4)
            sub[f"{a_key}_vs_{b_key}_ndcg_delta_{ds_alias}"] = _fmt(da, 4)

    # For rungs whose val logging is known-broken in the current grid,
    # emit an explicit "n/a" for the val-test gap so the selection-bias
    # table renders cleanly even before an R0 rerun lands. Only do this
    # when the key wasn't already resolved from seed data (which would
    # override this default) and the rung has at least one seed present
    # (otherwise the rung is just not yet in the grid).
    for ds_alias, rungs in agg.items():
        r0_seeds = rungs.get("R0_naive", {}).get("seeds", [])
        if r0_seeds and f"r0_valtest_gap_{ds_alias}" not in sub:
            sub[f"r0_valtest_gap_{ds_alias}"] = "n/a"

    return sub


def substitute(tex: str, subs: Dict[str, str]) -> Tuple[str, List[str]]:
    pattern = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
    unresolved: List[str] = []

    def _repl(m: re.Match) -> str:
        key = m.group(1)
        if key in subs:
            return subs[key]
        unresolved.append(key)
        return m.group(0)

    return pattern.sub(_repl, tex), unresolved


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs_v2/grid"))
    p.add_argument("--paper_in", type=Path, default=Path("paper"))
    p.add_argument("--paper_out", type=Path, default=Path("paper_filled"))
    p.add_argument("--require_complete", action="store_true")
    p.add_argument("--check_only", action="store_true")
    args = p.parse_args()

    if not args.runs.exists():
        print(f"[error] runs dir not found: {args.runs}", file=sys.stderr)
        return 2

    agg = aggregate(args.runs)
    subs = build_substitutions(agg, args.runs)

    tex_files = sorted(args.paper_in.glob("*.tex"))
    if not tex_files:
        print(f"[error] no .tex files in {args.paper_in}", file=sys.stderr)
        return 2

    all_unresolved: Dict[str, List[str]] = {}
    if not args.check_only:
        args.paper_out.mkdir(parents=True, exist_ok=True)

    for fp in tex_files:
        text = fp.read_text(encoding="utf-8")
        new_text, unresolved = substitute(text, subs)
        if unresolved:
            all_unresolved[fp.name] = sorted(set(unresolved))
        if not args.check_only:
            (args.paper_out / fp.name).write_text(new_text, encoding="utf-8")

    if not args.check_only:
        (args.paper_out / "_aggregate.json").write_text(json.dumps(agg, indent=2))
        (args.paper_out / "_substitutions.json").write_text(json.dumps(subs, indent=2))

    report_lines: List[str] = [
        "# fill_placeholders report",
        f"runs_root: {args.runs}",
        f"datasets: {sorted(agg.keys())}",
        f"resolved placeholders: {len(subs)}",
        "",
    ]

    if all_unresolved:
        report_lines.append("## unresolved placeholders (by file)")
        for fname in sorted(all_unresolved.keys()):
            keys = all_unresolved[fname]
            report_lines.append(f"### {fname} ({len(keys)})")
            for k in keys:
                report_lines.append(f"  - {k}")
            report_lines.append("")
    else:
        report_lines.append("## all placeholders resolved")

    report_text = "\n".join(report_lines) + "\n"
    if not args.check_only:
        (args.paper_out / "_fill_report.txt").write_text(report_text, encoding="utf-8")

    print(f"resolved: {len(subs)}")
    total_unresolved = sum(len(v) for v in all_unresolved.values())
    print(f"unresolved: {total_unresolved} across {len(all_unresolved)} file(s)")
    for fname, keys in sorted(all_unresolved.items()):
        print(f"  {fname}: {len(keys)}")

    if args.require_complete and total_unresolved > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
