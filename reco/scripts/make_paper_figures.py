"""Generate the paper's headline figures from completed runs/.

Figures produced:
  fig_ablation.pdf        — HR@10 / NDCG@10 bar chart per (dataset, reward)
  fig_sim_real_gap.pdf    — |r_sim - r_real| vs reward function
  fig_uncertainty_cal.pdf — reliability diagram per reward
  tab_main.tex            — main results table

Usage:
    py -3.12 scripts/make_paper_figures.py --runs runs --out paper_assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np


def _load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(runs_dir.glob("*")):
        cfg_p = d / "config.json"
        final_p = d / "final.json"
        if not cfg_p.exists() or not final_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        final = json.loads(final_p.read_text())
        rows.append({
            "tag": d.name,
            "dataset": cfg["dataset"]["name"],
            "agent": cfg["agent"]["name"],
            "reward": cfg["reward"]["name"],
            "HR@10": final["last_metrics"].get("HR@10"),
            "NDCG@10": final["last_metrics"].get("NDCG@10"),
            "MRR": final["last_metrics"].get("MRR"),
            "sim_real_gap": final.get("sim_real_gap", {}),
        })
    return rows


def figure_main(rows, out: Path):
    datasets = sorted(set(r["dataset"] for r in rows))
    rewards  = ["binary", "naive_continuous", "hard_gate", "ug_mors"]
    agents   = sorted(set(r["agent"] for r in rows))
    fig, axes = plt.subplots(len(datasets), len(agents),
                             figsize=(4 * len(agents), 3 * len(datasets)), squeeze=False)
    for i, d in enumerate(datasets):
        for j, a in enumerate(agents):
            ax = axes[i][j]
            vals = []
            for r in rewards:
                match = [x["NDCG@10"] for x in rows
                         if x["dataset"] == d and x["agent"] == a and x["reward"] == r
                         and x["NDCG@10"] is not None]
                vals.append(np.mean(match) if match else 0.0)
            bars = ax.bar(rewards, vals)
            # highlight ours
            for b, r in zip(bars, rewards):
                if r == "ug_mors":
                    b.set_edgecolor("red")
                    b.set_linewidth(2.0)
            ax.set_title(f"{d} / {a}")
            ax.set_ylabel("NDCG@10")
            ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "fig_main.pdf"); fig.savefig(out / "fig_main.png", dpi=150)
    plt.close(fig)


def figure_sim_real_gap(rows, out: Path):
    # average |gap MAE| and corr-with-u_llm across (dataset, agent), grouped by reward
    rewards = ["binary", "naive_continuous", "hard_gate", "ug_mors"]
    mae = {r: [] for r in rewards}
    corr = {r: [] for r in rewards}
    for row in rows:
        for r in rewards:
            g = row["sim_real_gap"].get(r, {})
            if g:
                mae[r].append(g.get("MAE", np.nan))
                corr[r].append(g.get("corr_|gap|_u_llm", np.nan))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.bar(rewards, [np.nanmean(mae[r]) if mae[r] else 0 for r in rewards])
    ax1.set_title("MAE |r_sim - r_real|"); ax1.set_ylabel("MAE")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(rewards, [np.nanmean(corr[r]) if corr[r] else 0 for r in rewards])
    ax2.set_title("corr(|gap|, u_llm)"); ax2.set_ylabel("Pearson r")
    ax2.axhline(0.0, color="gray", lw=0.5)
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "fig_sim_real_gap.pdf"); fig.savefig(out / "fig_sim_real_gap.png", dpi=150)
    plt.close(fig)


def table_main(rows, out: Path):
    datasets = sorted(set(r["dataset"] for r in rows))
    rewards = ["binary", "naive_continuous", "hard_gate", "ug_mors"]
    agents = sorted(set(r["agent"] for r in rows))
    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Agent & Binary & NaiveCont & HardGate & \textbf{UG-MORS} \\")
    lines.append(r"\midrule")
    for d in datasets:
        for a in agents:
            vals = []
            for r in rewards:
                match = [x["NDCG@10"] for x in rows
                         if x["dataset"] == d and x["agent"] == a and x["reward"] == r
                         and x["NDCG@10"] is not None]
                vals.append(f"{np.mean(match):.4f}" if match else "--")
            lines.append(f"{d} & {a} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out / "tab_main.tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper_assets")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_runs(ROOT / args.runs)
    if not rows:
        print(f"no completed runs in {args.runs}")
        return
    figure_main(rows, out)
    figure_sim_real_gap(rows, out)
    table_main(rows, out)
    print(f"wrote assets to {out}")


if __name__ == "__main__":
    main()
