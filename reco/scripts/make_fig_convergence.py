"""Generate fig_convergence.pdf for paper_expanded/.

Plots per-epoch val-NDCG@10 on MovieLens-1M seed 42 for four rungs:
  R_warm (supervised-only anchor), R0_naive (uncalibrated LLM reward),
  R3_ugmv2_noBC (soft conformal gate, reward-side), R4_ugmv2_BC (R3 + BC).

Input:  runs_v2/grid/movielens-1m/<rung>/seed42/history.json
Output: paper_expanded/figures/fig_convergence.pdf

Per-step granularity is not logged in the current grid, so this uses
per-epoch granularity. Re-runnable from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "runs_v2" / "grid" / "movielens-1m"
OUT = ROOT / "paper_expanded" / "figures" / "fig_convergence.pdf"

RUNGS = [
    ("R_warm",           "R_warm (supervised)",          "#888888", "--"),
    ("R0_naive",         "R0_naive (LLM reward, no gate)", "#d62728", "-"),
    ("R3_ugmv2_noBC",    "R3_ugmv2_noBC (soft gate)",    "#1f77b4", "-"),
    ("R4_ugmv2_BC",      "R4_ugmv2_BC (gate + BC)",      "#2ca02c", "-"),
]


def load_val_ndcg_by_epoch(history_path: Path) -> tuple[list[int], list[float]]:
    if not history_path.exists():
        return [], []
    hist = json.loads(history_path.read_text())
    epochs, vals = [], []
    for ep in hist:
        if "val_NDCG@10" in ep and "epoch" in ep:
            epochs.append(ep["epoch"])
            vals.append(ep["val_NDCG@10"])
    return epochs, vals


def load_train_ndcg_by_epoch(history_path: Path) -> tuple[list[int], list[float]]:
    if not history_path.exists():
        return [], []
    hist = json.loads(history_path.read_text())
    epochs, vals = [], []
    for ep in hist:
        if "NDCG@10" in ep and "epoch" in ep:
            epochs.append(ep["epoch"])
            vals.append(ep["NDCG@10"])
    return epochs, vals


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    any_plotted = False
    for rung, label, color, ls in RUNGS:
        hp = GRID / rung / "seed42" / "history.json"
        epochs, vals = load_val_ndcg_by_epoch(hp)
        if not epochs:
            epochs, vals = load_train_ndcg_by_epoch(hp)
            label = label + " (test*)"
        if not epochs:
            print(f"  [skip] no history for {rung}")
            continue
        ax.plot(epochs, vals, color=color, linestyle=ls, linewidth=1.4,
                marker="o", markersize=3, label=label)
        any_plotted = True

    if not any_plotted:
        raise SystemExit("No run histories found under runs_v2/grid/movielens-1m/")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val-NDCG@10")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("MovieLens-1M, seed 42")
    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
