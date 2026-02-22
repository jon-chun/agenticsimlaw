#!/usr/bin/env python3
"""
Random-Flip Monte Carlo Ablation (Critique #1 control).

Tests whether debate's calibration (low BRD) could arise from simply adding
random noise to single-shot predictions.  For each (model, dataset) pair:
  1. Take repeat_id=0 from the SC data as the single-shot baseline.
  2. For flip_rate in 1%..50%, run 10,000 Monte Carlo trials that randomly
     flip that fraction of predictions, recording YES%, BRD, and accuracy.
  3. Compare the random-flip BRD envelope against observed debate BRD.

Key insight: on NLSY97 (base rate 36%), random flipping pushes YES% toward
50%, so min achievable random-flip BRD ≈ |50% - 36%| = 14%.  Debate BRD
can be much lower (e.g., 2.3% for GPT-4o Mini), proving debate is not
merely adding noise.

Outputs
-------
- results/ablation_random_flip.csv
- papers/tmlr2026/images/figure_random_flip_ablation.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Paths ──────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SC_PATHS = {
    "nlsy97": os.path.join(REPO_ROOT, "results", "standardllm", "sc_nlsy97_merged.csv"),
    "compas": os.path.join(REPO_ROOT, "results", "standardllm", "sc_compas_merged.csv"),
}
DEBATE_PATHS = {
    "nlsy97": os.path.join(REPO_ROOT, "results", "debate_aggregate_nlsy97_200.csv"),
    "compas": os.path.join(REPO_ROOT, "results", "debate_aggregate_compas_200.csv"),
}
OUT_CSV = os.path.join(REPO_ROOT, "results", "ablation_random_flip.csv")
OUT_FIG = os.path.join(
    REPO_ROOT, "papers", "tmlr2026", "images", "figure_random_flip_ablation.pdf"
)

BASE_RATES = {"nlsy97": 0.36, "compas": 0.45}
OVERLAPPING_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gemini/gemini-2.5-flash"]
MODEL_DISPLAY = {
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
}

N_TRIALS = 10_000
FLIP_RATES = np.arange(0.01, 0.51, 0.01)
RNG_SEED = 42

# ── Style (matches existing paper figures) ─────────────────────────────
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

# ── Colours (one per model, consistent across panels) ──────────────────
MODEL_COLORS = {
    "gpt-4o-mini": "#1E3A8A",          # dark blue
    "gpt-4.1-mini": "#3B82F6",         # medium blue
    "gemini/gemini-2.5-flash": "#93C5FD",  # light blue
}


# ── Helpers ────────────────────────────────────────────────────────────
def load_single_shot(dataset: str) -> dict[str, np.ndarray]:
    """Return {model: array of 1/0 predictions} from SC repeat_id=0."""
    df = pd.read_csv(SC_PATHS[dataset])
    df = df[df["repeat_id"] == 0].copy()
    out = {}
    for model in OVERLAPPING_MODELS:
        sub = df[df["model_name"] == model].sort_values("case_id")
        preds = (sub["prediction"].str.upper() == "YES").astype(int).values
        out[model] = preds
    return out


def load_ground_truth(dataset: str) -> np.ndarray:
    """Return boolean ground-truth array (aligned by case_id) from SC data."""
    df = pd.read_csv(SC_PATHS[dataset])
    df = df[df["repeat_id"] == 0].copy()
    # Use first model to get ground truth (same for all)
    model = OVERLAPPING_MODELS[0]
    sub = df[df["model_name"] == model].sort_values("case_id")
    return sub["ground_truth"].values.astype(bool)


def load_debate_brd(dataset: str) -> dict[str, float]:
    """Return {model: debate BRD} using majority vote over repeats."""
    df = pd.read_csv(DEBATE_PATHS[dataset])
    df = df[df["prediction"].isin(["YES", "NO"])].copy()
    base_rate = BASE_RATES[dataset]
    out = {}
    for model in OVERLAPPING_MODELS:
        sub = df[df["model_name"] == model]
        # majority vote per case
        votes = (
            sub.groupby("case_id")["prediction"]
            .apply(lambda s: (s == "YES").sum() > len(s) / 2)
        )
        yes_pct = votes.mean()
        out[model] = abs(yes_pct - base_rate)
    return out


def monte_carlo_flip(
    preds: np.ndarray, flip_rate: float, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """Return array of YES fractions from n_trials random-flip experiments."""
    n = len(preds)
    n_flip = max(1, int(round(flip_rate * n)))
    yes_fracs = np.empty(n_trials)
    for t in range(n_trials):
        flipped = preds.copy()
        idx = rng.choice(n, size=n_flip, replace=False)
        flipped[idx] = 1 - flipped[idx]
        yes_fracs[t] = flipped.mean()
    return yes_fracs


def monte_carlo_accuracy(
    preds: np.ndarray, gt: np.ndarray, flip_rate: float,
    n_trials: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return array of accuracy values from n_trials random-flip experiments."""
    n = len(preds)
    n_flip = max(1, int(round(flip_rate * n)))
    accs = np.empty(n_trials)
    for t in range(n_trials):
        flipped = preds.copy()
        idx = rng.choice(n, size=n_flip, replace=False)
        flipped[idx] = 1 - flipped[idx]
        accs[t] = (flipped == gt.astype(int)).mean()
    return accs


# ── Main ───────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for dataset in ["nlsy97", "compas"]:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}  (base rate = {BASE_RATES[dataset]:.0%})")
        print(f"{'='*60}")

        single_shots = load_single_shot(dataset)
        gt = load_ground_truth(dataset)
        debate_brds = load_debate_brd(dataset)
        base_rate = BASE_RATES[dataset]

        for model in OVERLAPPING_MODELS:
            preds = single_shots[model]
            baseline_yes = preds.mean()
            baseline_brd = abs(baseline_yes - base_rate)
            d_brd = debate_brds[model]
            print(
                f"\n  {MODEL_DISPLAY[model]}: baseline YES%={baseline_yes:.1%}, "
                f"BRD={baseline_brd:.3f}, debate BRD={d_brd:.3f}"
            )

            for flip_rate in FLIP_RATES:
                yes_fracs = monte_carlo_flip(preds, flip_rate, N_TRIALS, rng)
                brds = np.abs(yes_fracs - base_rate)
                accs = monte_carlo_accuracy(preds, gt, flip_rate, N_TRIALS, rng)

                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "flip_rate": round(flip_rate, 2),
                    "mean_yes_pct": yes_fracs.mean(),
                    "mean_brd": brds.mean(),
                    "brd_ci_lo": np.percentile(brds, 2.5),
                    "brd_ci_hi": np.percentile(brds, 97.5),
                    "min_brd": brds.min(),
                    "mean_acc": accs.mean(),
                    "acc_ci_lo": np.percentile(accs, 2.5),
                    "acc_ci_hi": np.percentile(accs, 97.5),
                    "debate_brd": d_brd,
                    "baseline_yes_pct": baseline_yes,
                    "baseline_brd": baseline_brd,
                    "base_rate": base_rate,
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV → {OUT_CSV}")

    # ── Verification: debate BRD vs. theoretical random-flip floor ──
    print("\n" + "=" * 60)
    print("VERIFICATION: debate BRD vs. random-flip floor on NLSY97")
    print("  Theoretical floor = |50% - base_rate| = "
          f"|50% - 36%| = {abs(0.50 - 0.36):.3f}")
    print("=" * 60)
    n_pass = 0
    for model in OVERLAPPING_MODELS:
        sub = df[(df["model"] == model) & (df["dataset"] == "nlsy97")]
        # Best mean BRD achievable by random flipping (at optimal flip rate)
        best_mean_brd = sub["mean_brd"].min()
        d_brd = sub["debate_brd"].iloc[0]
        below_floor = d_brd < best_mean_brd
        if below_floor:
            n_pass += 1
        status = "PASS" if below_floor else "—"
        print(
            f"  {MODEL_DISPLAY[model]:20s}: debate BRD={d_brd:.3f}  "
            f"best random-flip mean BRD={best_mean_brd:.3f}  [{status}]"
        )
    print(f"\n  {n_pass}/{len(OVERLAPPING_MODELS)} models have debate BRD "
          f"below random-flip floor on NLSY97")

    # ── Figure ─────────────────────────────────────────────────────
    generate_figure(df)


def generate_figure(df: pd.DataFrame):
    """Two-panel line plot: BRD vs. flip rate with debate BRD reference."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    datasets = ["nlsy97", "compas"]
    titles = ["NLSY97", "COMPAS"]

    for ax, dataset, title in zip(axes, datasets, titles):
        base_rate = BASE_RATES[dataset]
        theoretical_min = abs(0.50 - base_rate)

        for model in OVERLAPPING_MODELS:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            color = MODEL_COLORS[model]
            label = MODEL_DISPLAY[model]

            # BRD line with CI band
            ax.plot(
                sub["flip_rate"], sub["mean_brd"],
                color=color, linewidth=1.5, label=label, zorder=3,
            )
            ax.fill_between(
                sub["flip_rate"], sub["brd_ci_lo"], sub["brd_ci_hi"],
                color=color, alpha=0.15, zorder=2,
            )
            # Debate BRD horizontal line
            d_brd = sub["debate_brd"].iloc[0]
            ax.axhline(
                d_brd, color=color, linestyle="--", linewidth=1.0, alpha=0.8, zorder=2,
            )

        # Theoretical minimum BRD from random flipping
        ax.axhline(
            theoretical_min, color="#888888", linestyle=":", linewidth=1.0,
            label=f"Random-flip floor ({theoretical_min:.0%})", zorder=1,
        )

        ax.set_xlabel("Flip rate", fontsize=9)
        ax.set_title(
            f"{title} (base rate {base_rate:.0%})",
            fontsize=10, fontweight="bold", pad=8,
        )
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_xlim(0, 0.50)
        ax.set_ylim(0, None)

    axes[0].set_ylabel("Base-Rate Deviation (BRD)", fontsize=9)

    # Legend: model lines + debate dashes + theoretical floor
    from matplotlib.lines import Line2D
    handles = []
    for model in OVERLAPPING_MODELS:
        handles.append(
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=1.5,
                   label=MODEL_DISPLAY[model])
        )
    handles.append(
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.0,
               label="Debate BRD (per model)")
    )
    handles.append(
        Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.0,
               label="Random-flip floor")
    )
    fig.legend(
        handles=handles, loc="upper center", ncol=5, fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=1.5,
        handletextpad=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    print(f"Saved figure → {OUT_FIG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
