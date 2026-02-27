#!/usr/bin/env python3
"""
Generate Figure: Commercial LLM comparison (YES rate and accuracy).

Reproduces the data from Table 7 as a two-panel bar chart:
  (1) YES rate by method, (2) Accuracy by method.
Dashed lines show 50% YES rate (left) and dataset base rates (right).

Note: The paper's figure was manually polished for layout. This script
generates a reproducible version from the same underlying data. The
visual layout may differ slightly from the published figure.

Data source: Table 7 in the paper (controlled commercial LLM comparison).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ── Style (matches other paper figures) ────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "text.usetex": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_FIG = os.path.join(REPO_ROOT, "figures", "figure_commercial_comparison.pdf")

# ── Data from Table 7 (controlled commercial LLM comparison) ──────────
# Format: (model, method, nlsy97_acc, nlsy97_yes%, compas_acc, compas_yes%)
TABLE_DATA = [
    ("Grok 4.1 Fast", "Zero-shot", 0.43, 88.4, 0.70, 63.7),
    ("Grok 4.1 Fast", "CoT",       0.40, 88.8, 0.72, 65.2),
    ("Grok 4.1 Fast", "Debate",    0.43, 78.6, 0.75, 45.0),
    ("GPT-4o Mini",   "Zero-shot", 0.51, 73.4, 0.72, 34.5),
    ("GPT-4o Mini",   "CoT",       0.37, 93.8, 0.85, 57.2),
    ("GPT-4o Mini",   "Debate",    0.59, 53.8, 0.71, 43.3),
]

BASE_RATES = {"NLSY97": 31, "COMPAS": 47}  # YES% base rates

# ── Colours ────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "Zero-shot": "#a6cee3",  # light blue
    "CoT":       "#4a86c8",  # medium blue
    "Debate":    "#1a2744",  # dark navy
}

MODELS = ["Grok 4.1 Fast", "GPT-4o Mini"]
METHODS = ["Zero-shot", "CoT", "Debate"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18, left=0.08, right=0.97)

    # Build data structures
    data = {}
    for model, method, n_acc, n_yes, c_acc, c_yes in TABLE_DATA:
        data[(model, method)] = {
            "NLSY97": {"acc": n_acc, "yes": n_yes},
            "COMPAS": {"acc": c_acc, "yes": c_yes},
        }

    # Panel 1: YES rate
    ax = axes[0]
    x = np.arange(len(MODELS))
    w = 0.22
    offsets = [-w, 0, w]

    for i, method in enumerate(METHODS):
        nlsy97_vals = [data[(m, method)]["NLSY97"]["yes"] for m in MODELS]
        compas_vals = [data[(m, method)]["COMPAS"]["yes"] for m in MODELS]

        # Interleave: for each model, show NLSY97 and COMPAS side by side
        positions_n = x * 2.5 + offsets[i]
        positions_c = x * 2.5 + 1 + offsets[i]

        bars_n = ax.bar(positions_n, nlsy97_vals, w, color=METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.5, alpha=0.9)
        bars_c = ax.bar(positions_c, compas_vals, w, color=METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.5,
                        hatch="///" if True else "", alpha=0.7)

        for bar in list(bars_n) + list(bars_c):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("YES Rate (%)", fontsize=10)
    ax.set_title("YES Rate by Method", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(MODELS, fontsize=9)

    # Add dataset labels
    for xi in [0, 2.5]:
        ax.text(xi, -8, "NLSY97", ha="center", fontsize=7, fontstyle="italic",
                color="grey", transform=ax.get_xaxis_transform())
        ax.text(xi + 1, -8, "COMPAS", ha="center", fontsize=7, fontstyle="italic",
                color="grey", transform=ax.get_xaxis_transform())

    # Panel 2: Accuracy
    ax = axes[1]
    for i, method in enumerate(METHODS):
        nlsy97_vals = [data[(m, method)]["NLSY97"]["acc"] for m in MODELS]
        compas_vals = [data[(m, method)]["COMPAS"]["acc"] for m in MODELS]

        positions_n = x * 2.5 + offsets[i]
        positions_c = x * 2.5 + 1 + offsets[i]

        bars_n = ax.bar(positions_n, nlsy97_vals, w, color=METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.5, alpha=0.9)
        bars_c = ax.bar(positions_c, compas_vals, w, color=METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.5, alpha=0.7)

        for bar in list(bars_n) + list(bars_c):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    # Base rate lines
    for dataset, rate in BASE_RATES.items():
        ax.axhline(rate / 100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Accuracy by Method", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(MODELS, fontsize=9)

    for xi in [0, 2.5]:
        ax.text(xi, -8, "NLSY97", ha="center", fontsize=7, fontstyle="italic",
                color="grey", transform=ax.get_xaxis_transform())
        ax.text(xi + 1, -8, "COMPAS", ha="center", fontsize=7, fontstyle="italic",
                color="grey", transform=ax.get_xaxis_transform())

    # Shared legend
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS[m]) for m in METHODS],
        METHODS,
        loc="upper center", ncol=3, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")
    plt.close()


if __name__ == "__main__":
    main()
