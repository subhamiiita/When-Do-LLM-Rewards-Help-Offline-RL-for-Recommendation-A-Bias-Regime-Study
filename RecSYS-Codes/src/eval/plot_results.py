"""Generate all result visualizations.

Produces the following PNG files in results/figures/:
    01_simreal_ndcg_grid.png      - NDCG@10 per (dataset × agent × variant)
    02_simreal_hr_grid.png        - HR@10 same structure
    03_ug_mors_lift.png           - UG-MORS vs baseline lift per cell (sorted)
    04_liking_grid.png            - Liking% per (dataset × agent × variant)
    05_base_paper_vs_ours.png     - Liking% side-by-side on overlapping datasets
    06_ablation_components.png    - UG-MORS vs component-dropped variants
    07_reward_snr.png             - SNR diagnostic per variant per dataset
    08_sim_vs_real.png            - Scatter: simulator Liking% vs SimReal NDCG@10
    09_uncertainty_hist.png       - u_llm distribution per dataset
    10_nli_score_hist.png         - NLI soft score distribution per dataset
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.config import project_root


# Color palette — consistent across all charts
COLORS = {
    "baseline_vote":    "#8c8c8c",   # gray
    "naive_continuous": "#f08c00",   # orange
    "ug_mors":          "#1f77b4",   # blue (primary)
    "ug_pbrs":          "#2ca02c",   # green
    "ug_mors_fixed":    "#9ecae1",   # light blue
    "ug_mors_no_div":   "#6baed6",   # mid blue
    "ug_mors_no_per":   "#3182bd",   # deeper blue
    "ug_mors_no_ret":   "#08519c",   # dark blue
}
VARIANT_LABEL = {
    "baseline_vote":    "Baseline Vote",
    "naive_continuous": "Naive Continuous",
    "ug_mors":          "UG-MORS (full)",
    "ug_pbrs":          "UG-PBRS",
    "ug_mors_fixed":    "UG-MORS fixed-w",
    "ug_mors_no_div":   "UG-MORS −div",
    "ug_mors_no_per":   "UG-MORS −per",
    "ug_mors_no_ret":   "UG-MORS −ret",
}


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by(rows: list[dict], key_fields: list[str], value_field: str) -> dict:
    out = {}
    for r in rows:
        try:
            val = float(r[value_field])
        except (ValueError, KeyError, TypeError):
            continue
        key = tuple(r[k] for k in key_fields)
        out[key] = val
    return out


def _figdir() -> Path:
    d = project_root() / "results/figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_simreal_grid(sim_rows: list[dict], metric: str, ylabel: str, filename: str, title: str) -> None:
    """Grouped bar chart: 3 datasets × 4 agents × 4 variants."""
    data = _group_by(sim_rows, ["dataset", "agent", "variant"], metric)
    datasets = ["ml1m", "videogames", "yelp"]
    agents = ["dqn", "ppo", "a2c", "trpo"]
    variants = ["baseline_vote", "naive_continuous", "ug_mors", "ug_pbrs"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    x = np.arange(len(agents))
    width = 0.2
    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, v in enumerate(variants):
            ys = [data.get((ds, a, v), 0.0) for a in agents]
            bars = ax.bar(x + (j - 1.5) * width, ys, width, color=COLORS[v], label=VARIANT_LABEL[v])
            for b, y in zip(bars, ys):
                if y > 0:
                    ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in agents])
        ax.set_xlabel("RL Agent")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out = _figdir() / filename
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_lift_summary(sim_rows: list[dict]) -> None:
    """Sorted horizontal bar chart of UG-MORS lift over baseline per (dataset, agent)."""
    ndcg = _group_by(sim_rows, ["dataset", "agent", "variant"], "ndcg_at_10")
    cells = []
    for ds in ["ml1m", "videogames", "yelp"]:
        for ag in ["dqn", "ppo", "a2c", "trpo"]:
            b = ndcg.get((ds, ag, "baseline_vote"))
            u = ndcg.get((ds, ag, "ug_mors"))
            if b is not None and u is not None:
                cells.append((f"{ds} / {ag}", b, u))
    cells.sort(key=lambda r: (r[2] - r[1]) / max(r[1], 1e-6), reverse=True)
    labels = [c[0] for c in cells]
    baseline = np.array([c[1] for c in cells])
    ugmors = np.array([c[2] for c in cells])
    rel = (ugmors - baseline) / np.maximum(baseline, 1e-6) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    y = np.arange(len(cells))
    ax1.barh(y - 0.2, baseline, 0.4, color=COLORS["baseline_vote"], label="Baseline Vote")
    ax1.barh(y + 0.2, ugmors, 0.4, color=COLORS["ug_mors"], label="UG-MORS")
    for i, (b, u) in enumerate(zip(baseline, ugmors)):
        ax1.text(b + 0.005, i - 0.2, f"{b:.3f}", va="center", fontsize=8)
        ax1.text(u + 0.005, i + 0.2, f"{u:.3f}", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("SimReal NDCG@10")
    ax1.set_title("Absolute NDCG@10 — Baseline vs UG-MORS")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    colors = ["#2ca02c" if r > 0 else "#d62728" for r in rel]
    ax2.barh(y, rel, 0.6, color=colors)
    for i, r in enumerate(rel):
        ax2.text(r + (2 if r > 0 else -2), i, f"{r:+.0f}%", va="center", fontsize=9,
                 ha="left" if r > 0 else "right")
    ax2.axvline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Relative lift (%)")
    ax2.set_title("UG-MORS lift over Baseline (sorted)")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Figure: UG-MORS vs Baseline Vote — SimReal NDCG@10", fontsize=14)
    fig.tight_layout()
    out = _figdir() / "03_ug_mors_lift.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_base_paper_vs_ours(sim_rows: list[dict]) -> None:
    """Side-by-side: base paper Liking% vs ours (baseline & ug_mors) on overlapping datasets."""
    liking = _group_by(sim_rows, ["dataset", "agent", "variant"], "Liking_pct")
    # Base paper Liking% from Zhang et al. 2025 Table 2
    base = {
        ("yelp", "ppo"): 34.59, ("yelp", "trpo"): 40.07, ("yelp", "a2c"): 48.35, ("yelp", "dqn"): 49.43,
        ("videogames", "ppo"): 29.30, ("videogames", "trpo"): 32.46, ("videogames", "a2c"): 29.54, ("videogames", "dqn"): 33.18,
    }
    agents = ["dqn", "a2c", "trpo", "ppo"]
    datasets = ["yelp", "videogames"]
    dataset_labels = {"yelp": "Yelp\n(ours: MO 10-core; paper: full-US)",
                      "videogames": "Amazon Games/VG\n(ours: 10-core; paper: 5-core)"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        x = np.arange(len(agents))
        width = 0.27
        paper_vals = [base.get((ds, a), 0) for a in agents]
        ours_base = [liking.get((ds, a, "baseline_vote"), 0) for a in agents]
        ours_ug = [liking.get((ds, a, "ug_mors"), 0) for a in agents]
        ax.bar(x - width, paper_vals, width, color="#8c564b", label="Base paper (Zhang et al. 2025)")
        ax.bar(x, ours_base, width, color=COLORS["baseline_vote"], label="Ours — Baseline Vote")
        ax.bar(x + width, ours_ug, width, color=COLORS["ug_mors"], label="Ours — UG-MORS")
        for i, (p, b, u) in enumerate(zip(paper_vals, ours_base, ours_ug)):
            if p > 0: ax.text(i - width, p, f"{p:.1f}", ha="center", va="bottom", fontsize=8)
            if b > 0: ax.text(i,          b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
            if u > 0: ax.text(i + width,  u, f"{u:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in agents])
        ax.set_xlabel("RL Agent")
        ax.set_ylabel("Liking% (top-10 simulator vote rate)")
        ax.set_ylim(0, 105)
        ax.set_title(dataset_labels[ds])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Figure: Simulator Liking% — Base Paper vs Ours\n(Denser preprocessing + smaller catalog → higher absolute Liking%)", fontsize=13)
    fig.tight_layout()
    out = _figdir() / "05_base_paper_vs_ours.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ablation(sim_rows: list[dict]) -> None:
    """Component ablations: ug_mors vs no_div/no_per/no_ret/fixed per (dataset, agent)."""
    ndcg = _group_by(sim_rows, ["dataset", "agent", "variant"], "ndcg_at_10")
    datasets = ["ml1m", "videogames", "yelp"]
    agents = ["dqn", "ppo", "a2c", "trpo"]
    variants = ["ug_mors", "ug_mors_fixed", "ug_mors_no_div", "ug_mors_no_per", "ug_mors_no_ret"]

    # Check if any ablation variants exist in the CSV
    has_ablations = any((ds, ag, v) in ndcg for ds in datasets for ag in agents for v in variants if v != "ug_mors")
    if not has_ablations:
        print("Skipping ablation plot — no ablation variants in simreal.csv. Run:")
        print("  python -m src.scripts.run_all_ablations --ablations --skip_existing")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    x = np.arange(len(agents))
    width = 0.15
    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, v in enumerate(variants):
            ys = [ndcg.get((ds, a, v), 0.0) for a in agents]
            ax.bar(x + (j - 2) * width, ys, width, color=COLORS[v], label=VARIANT_LABEL[v])
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in agents])
        ax.set_xlabel("RL Agent")
        if i == 0:
            ax.set_ylabel("SimReal NDCG@10")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure: Component Ablations of UG-MORS — SimReal NDCG@10", fontsize=14)
    fig.tight_layout()
    out = _figdir() / "06_ablation_components.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_reward_snr() -> None:
    """SNR per variant per dataset, from diagnostics_*.json."""
    variants = ["baseline_vote", "naive_continuous", "ug_mors", "ug_pbrs",
                "ug_mors_fixed", "ug_mors_no_div", "ug_mors_no_per", "ug_mors_no_ret"]
    datasets = ["ml1m", "videogames", "yelp"]
    snrs = {}
    for ds in datasets:
        path = project_root() / f"results/tables/diagnostics_{ds}.json"
        if not path.exists():
            continue
        with path.open() as f:
            d = json.load(f)
        snrs[ds] = {v: d.get(v, {}).get("snr", 0.0) for v in variants}
    if not snrs:
        print("Skipping SNR plot — no diagnostics JSON found")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(variants))
    width = 0.27
    for i, ds in enumerate(datasets):
        if ds not in snrs:
            continue
        ys = [snrs[ds][v] for v in variants]
        ax.bar(x + (i - 1) * width, ys, width, label=ds)
        for j, y in enumerate(ys):
            if y > 0:
                ax.text(x[j] + (i - 1) * width, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Reward SNR")
    ax.set_title("Figure: Reward SNR per variant on labeled (s, a) pairs\n(Note: SNR metric biased against continuous rewards — see writeup)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = _figdir() / "07_reward_snr.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_sim_vs_real(sim_rows: list[dict], simulator_rows: list[dict]) -> None:
    """Scatter: simulator Liking% vs SimReal NDCG@10, one point per (dataset, agent, variant).

    The idea: points that are HIGH on sim Liking% but LOW on SimReal NDCG@10
    are simulator-overfit. UG-MORS should sit in the upper-right (high both).
    """
    ndcg = _group_by(sim_rows, ["dataset", "agent", "variant"], "ndcg_at_10")
    liking = _group_by(simulator_rows, ["dataset", "agent", "variant"], "Liking_pct")
    variants = ["baseline_vote", "naive_continuous", "ug_mors", "ug_pbrs"]
    datasets = ["ml1m", "videogames", "yelp"]
    markers = {"ml1m": "o", "videogames": "s", "yelp": "^"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for v in variants:
        for ds in datasets:
            xs, ys = [], []
            for ag in ["dqn", "ppo", "a2c", "trpo"]:
                l = liking.get((ds, ag, v))
                n = ndcg.get((ds, ag, v))
                if l is not None and n is not None:
                    xs.append(l)
                    ys.append(n)
            ax.scatter(xs, ys, c=COLORS[v], marker=markers[ds], s=120, edgecolors="black",
                       linewidth=0.7, alpha=0.85, label=f"{VARIANT_LABEL[v]} / {ds}")
    # Legend: two legends - variants (color) and datasets (marker)
    from matplotlib.lines import Line2D
    variant_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[v], markersize=12,
                              label=VARIANT_LABEL[v], markeredgecolor="black") for v in variants]
    dataset_handles = [Line2D([0], [0], marker=markers[ds], color="w", markerfacecolor="gray", markersize=12,
                              label=ds, markeredgecolor="black") for ds in datasets]
    leg1 = ax.legend(handles=variant_handles, loc="upper left", title="Variant")
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, loc="lower right", title="Dataset")
    ax.set_xlabel("Simulator Liking% (higher = better in simulator)")
    ax.set_ylabel("SimReal NDCG@10 (higher = better on real logs)")
    ax.set_title("Figure: Simulator vs Real-Log Performance\nEach dot = (dataset, agent, variant). UG-MORS (blue) sits in the upper-right.")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _figdir() / "08_sim_vs_real.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_uncertainty_hist() -> None:
    """u_llm distribution per dataset — where the gate is active vs silent."""
    from src.data.cache import build_cache
    import torch
    datasets = ["ml1m", "videogames", "yelp"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, ds in zip(axes, datasets):
        cache = build_cache(ds, torch.device("cpu"))
        u = cache.u_llm.cpu().numpy()
        ax.hist(u, bins=30, color="#1f77b4", alpha=0.85, edgecolor="black")
        ax.axvline(u.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={u.mean():.3f}")
        ax.set_title(f"{ds}  (n={len(u)} items)")
        ax.set_xlabel("u_llm (LLM uncertainty)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("# items")
    fig.suptitle("Figure: Distribution of LLM uncertainty u_llm per dataset\nGate g = 1 − u_llm; items with high u_llm get suppressed semantic contribution", fontsize=12)
    fig.tight_layout()
    out = _figdir() / "09_uncertainty_hist.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_nli_hist() -> None:
    """NLI soft score distribution per dataset (touched cells only)."""
    import numpy as np
    datasets = ["ml1m", "videogames", "yelp"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, ds in zip(axes, datasets):
        path = project_root() / f"results/nli/{ds}.npz"
        if not path.exists():
            ax.text(0.5, 0.5, f"{ds}: NLI cache not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds)
            continue
        z = np.load(path)
        mask = z["mask"]
        scores = z["nli"][mask].astype(np.float32)
        ax.hist(scores, bins=50, color="#f08c00", alpha=0.85, edgecolor="black")
        ax.axvline(0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(scores.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={scores.mean():.3f}\nstd={scores.std():.3f}")
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{ds} (n={mask.sum():,} pairs)")
        ax.set_xlabel("NLI soft score = p(entail) − p(contradict)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("# (user, item) pairs")
    fig.suptitle("Figure: Persona NLI soft-score distribution per dataset (raw, pre-normalization)", fontsize=12)
    fig.tight_layout()
    out = _figdir() / "10_nli_score_hist.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    sim_rows = _load_csv(project_root() / "results/tables/simreal.csv")
    simulator_rows = _load_csv(project_root() / "results/tables/simulator_metrics.csv")

    # 01-02: NDCG and HR grids
    plot_simreal_grid(sim_rows, "ndcg_at_10", "SimReal NDCG@10", "01_simreal_ndcg_grid.png",
                      "Figure: SimReal NDCG@10 per (dataset × agent × reward variant)")
    plot_simreal_grid(sim_rows, "hr_at_10", "SimReal HR@10", "02_simreal_hr_grid.png",
                      "Figure: SimReal HR@10 per (dataset × agent × reward variant)")

    # 03: lift summary
    plot_lift_summary(sim_rows)

    # 04: Liking% grid
    plot_simreal_grid(simulator_rows, "Liking_pct", "Liking% (top-10 sim vote rate)",
                      "04_liking_grid.png",
                      "Figure: Simulator Liking% per (dataset × agent × reward variant)")

    # 05: base paper vs ours
    plot_base_paper_vs_ours(simulator_rows)

    # 06: ablations
    plot_ablation(sim_rows)

    # 07: reward SNR
    plot_reward_snr()

    # 08: sim vs real scatter
    plot_sim_vs_real(sim_rows, simulator_rows)

    # 09-10: diagnostics distributions
    try:
        plot_uncertainty_hist()
    except Exception as e:
        print(f"Skipping uncertainty histogram: {e}")
    try:
        plot_nli_hist()
    except Exception as e:
        print(f"Skipping NLI histogram: {e}")

    print(f"\nAll figures written to {_figdir()}")


if __name__ == "__main__":
    main()
