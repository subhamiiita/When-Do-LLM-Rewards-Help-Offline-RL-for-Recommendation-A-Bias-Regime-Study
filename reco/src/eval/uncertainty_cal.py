"""Reliability diagram for u_llm.

Bins items by u_llm, measures mean |r_sim - r_real| in each bin, and
reports Pearson r between (u_llm, abs_gap). The abstract claims r > 0.85;
this script verifies that claim empirically and produces the figure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def reliability(u_llm: np.ndarray, gap: np.ndarray, n_bins: int = 10) -> Dict:
    u = np.asarray(u_llm)
    g = np.abs(np.asarray(gap))
    edges = np.linspace(u.min(), u.max() + 1e-9, n_bins + 1)
    bin_ids = np.digitize(u, edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    mean_gap = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_ids == b
        count[b] = int(mask.sum())
        mean_gap[b] = float(g[mask].mean()) if count[b] > 0 else float("nan")

    if u.std() > 1e-6 and g.std() > 1e-6:
        pearson = float(np.corrcoef(u, g)[0, 1])
    else:
        pearson = float("nan")

    return {
        "bin_centres": centres.tolist(),
        "mean_abs_gap": mean_gap.tolist(),
        "count": count.tolist(),
        "pearson_r": pearson,
    }


def save_plot(result: Dict, out_png: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(result["bin_centres"], result["mean_abs_gap"], "o-")
    ax.set_xlabel("u_llm")
    ax.set_ylabel("mean |r_sim - r_real|")
    ax.set_title(f"{title}\nPearson r = {result['pearson_r']:.3f}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_json(result: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
