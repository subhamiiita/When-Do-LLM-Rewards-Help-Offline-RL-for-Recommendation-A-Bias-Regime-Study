"""Figure: relative change in MAE_highU vs LLM-real bias Delta.

Hardcoded three-dataset summary (numbers pulled from
paper_filled_v2/_substitutions.json at the three-seed-mean convention
for ML-1M; single-seed for Yelp and Amazon). Output:
paper_rewrite/figures/fig_bias_regime.pdf.
"""

from pathlib import Path

import matplotlib.pyplot as plt

POINTS = [
    # (label, bias Delta, rel MAE_highU change in %, annotation offset)
    ("ML-1M",     +0.32, -11.3, (+0.02, +1.2)),
    ("Yelp",      +0.13,  +2.9, (+0.02, +1.2)),
    ("Amazon VG", -0.23,  +8.1, (+0.02, +1.2)),
]

OUT = Path(__file__).resolve().parent.parent / "paper_rewrite" / "figures" / "fig_bias_regime.pdf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    xs = [p[1] for p in POINTS]
    ys = [p[2] for p in POINTS]

    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.5", zorder=0)
    ax.axvline(0.0, linestyle=":", linewidth=0.6, color="0.7", zorder=0)

    ax.scatter(xs, ys, s=80, zorder=2, color="#1f4e79", edgecolor="black", linewidth=0.6)
    for label, dx, dy, (ox, oy) in POINTS:
        ax.annotate(label, xy=(dx, dy), xytext=(dx + ox, dy + oy),
                    fontsize=9, ha="left", va="bottom")

    ax.set_xlabel(r"LLM-real bias $\Delta = \bar r_{\mathrm{sim}} - \bar r_{\mathrm{real}}$")
    ax.set_ylabel(r"$\Delta\,\mathrm{MAE}_{\mathrm{highU}}$  (%)")
    ax.set_xlim(-0.35, 0.45)
    ax.set_ylim(-15, 12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
