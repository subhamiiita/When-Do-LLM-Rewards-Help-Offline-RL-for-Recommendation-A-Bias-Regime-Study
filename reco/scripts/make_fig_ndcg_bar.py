"""Figure: grouped NDCG@10 bar chart by variant x dataset.

ML-1M bars are three-seed means with one-std error bars; Yelp and
Amazon VG are single-seed. Missing cells (variants not evaluated on
a dataset) are omitted rather than shown as zeros. Output:
paper_rewrite/figures/fig_ndcg_bar.pdf.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (variant label, ndcg dict {dataset: (mean, std)})
# std is None where the value is single-seed.
DATA = [
    ("Supervised",
        {"ML-1M": (0.0692, 0.0003), "Yelp": (0.0106, None), "Amazon VG": (0.0241, None)}),
    ("RL-raw",
        {"ML-1M": (0.0751, 0.0023), "Yelp": None,           "Amazon VG": (0.0268, None)}),
    ("RL-gated",
        {"ML-1M": (0.0784, 0.0017), "Yelp": None,           "Amazon VG": (0.0265, None)}),
    ("RL-gated-BC",
        {"ML-1M": (0.0762, 0.0016), "Yelp": (0.0137, None), "Amazon VG": (0.0268, None)}),
]

DATASETS = ["ML-1M", "Yelp", "Amazon VG"]

OUT = Path(__file__).resolve().parent.parent / "paper_rewrite" / "figures" / "fig_ndcg_bar.pdf"


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), sharey=False)
    colors = ["#bdbdbd", "#9ecae1", "#4292c6", "#08519c"]

    for ax, dataset in zip(axes, DATASETS):
        means, errs, labels, clrs = [], [], [], []
        for (variant, d), col in zip(DATA, colors):
            cell = d.get(dataset)
            if cell is None:
                continue
            mean, std = cell
            means.append(mean)
            errs.append(std if std is not None else 0.0)
            labels.append(variant)
            clrs.append(col)

        xs = np.arange(len(means))
        ax.bar(xs, means, yerr=errs, color=clrs, edgecolor="black", linewidth=0.5,
               capsize=2.5, error_kw=dict(linewidth=0.7))
        ax.set_title(dataset, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("NDCG@10")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
