#!/usr/bin/env python3
"""
Compare AgenticSimLaw debate results vs Self-Consistency (SC) majority-vote results.

For each model × dataset, computes:
  - Debate: accuracy, F1, YES%, BRD (from step2 aggregate CSV)
  - SC-K=59: accuracy, F1, YES%, BRD (from sc_majority_vote.py output)
  - Zero-shot (K=1): accuracy, F1, YES%, BRD (first rep from SC raw results)
  - Mann-Whitney U test on per-case accuracy (debate vs SC, n=100 cases)

Usage:
    python src/compare_debate_vs_sc.py \
        --debate-csv results/debate_aggregate.csv \
        --sc-per-case results/sc_majority_vote_per_case.csv \
        --sc-raw results/sc_raw_results.csv \
        --dataset nlsy97 \
        --output results/comparison_nlsy97.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Dataset target column mapping
# ---------------------------------------------------------------------------

DATASET_TARGET = {
    "nlsy97": "y_arrestedafter2002",
    "compas": "two_year_recid",
    "compas_nodecile": "two_year_recid",
}


def compute_brd(yes_pct: float, base_rate: float) -> float:
    """Base Rate Deviation: |YES% - base_rate|."""
    return abs(yes_pct - base_rate)


def metrics_from_predictions(predictions: pd.Series, ground_truth: pd.Series) -> dict:
    """Compute accuracy, F1, YES%, BRD from prediction and ground truth Series."""
    valid_mask = predictions.isin(["YES", "NO"])
    pred = predictions[valid_mask]
    truth = ground_truth[valid_mask].astype(bool)

    if len(pred) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}

    tp = ((pred == "YES") & truth).sum()
    tn = ((pred == "NO") & ~truth).sum()
    fp = ((pred == "YES") & ~truth).sum()
    fn = ((pred == "NO") & truth).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_pct = (pred == "YES").mean()
    base_rate = truth.mean()
    brd = compute_brd(yes_pct, base_rate)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "yes_pct": yes_pct,
        "base_rate": base_rate,
        "brd": brd,
        "n": int(total),
    }


def per_case_accuracy(predictions: pd.Series, ground_truth: pd.Series) -> np.ndarray:
    """Compute per-case binary accuracy (1=correct, 0=incorrect) for valid predictions."""
    valid_mask = predictions.isin(["YES", "NO"])
    pred = predictions[valid_mask]
    truth = ground_truth[valid_mask].astype(bool)
    correct = ((pred == "YES") & truth) | ((pred == "NO") & ~truth)
    return correct.astype(int).values


def load_debate_results(csv_path: str, dataset: str) -> pd.DataFrame:
    """Load step2 aggregate debate CSV and normalize columns."""
    df = pd.read_csv(csv_path)

    # Normalize prediction column to uppercase YES/NO
    if "prediction" in df.columns:
        df["prediction"] = df["prediction"].astype(str).str.strip().str.upper()

    # Get target column
    target_col = DATASET_TARGET.get(dataset)
    if target_col and target_col in df.columns:
        df["ground_truth"] = df[target_col].astype(bool)
    elif "ground_truth" in df.columns:
        df["ground_truth"] = df["ground_truth"].astype(bool)

    return df


def load_sc_per_case(csv_path: str) -> pd.DataFrame:
    """Load SC per-case majority vote results."""
    return pd.read_csv(csv_path)


def load_sc_raw(csv_path: str) -> pd.DataFrame:
    """Load raw SC results for extracting zero-shot (K=1) baseline."""
    return pd.read_csv(csv_path)


def get_zero_shot(sc_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract first repetition (rep=0) as zero-shot baseline."""
    return sc_raw[sc_raw["repeat_id"] == 0].copy()


def main():
    parser = argparse.ArgumentParser(description="Compare Debate vs SC Results")
    parser.add_argument("--debate-csv", type=str, required=True,
                        help="Path to step2 aggregate debate CSV")
    parser.add_argument("--sc-per-case", type=str, required=True,
                        help="Path to SC majority-vote per-case CSV")
    parser.add_argument("--sc-raw", type=str, required=True,
                        help="Path to raw SC results CSV (for zero-shot extraction)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_TARGET.keys()),
                        help="Dataset name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DEBATE vs SELF-CONSISTENCY COMPARISON — {args.dataset.upper()}")
    print("=" * 80)

    # Load data
    debate_df = load_debate_results(args.debate_csv, args.dataset)
    sc_per_case = load_sc_per_case(args.sc_per_case)
    sc_raw = load_sc_raw(args.sc_raw)
    zs_df = get_zero_shot(sc_raw)

    print(f"  Debate cases: {len(debate_df)}")
    print(f"  SC per-case: {len(sc_per_case)}")
    print(f"  SC raw: {len(sc_raw)}")
    print(f"  Zero-shot (K=1): {len(zs_df)}")

    # Get models present in both debate and SC
    debate_models = set(debate_df["model_name"].unique())
    sc_models = set(sc_per_case["model_name"].unique())
    common_models = sorted(debate_models & sc_models)

    if not common_models:
        # Try fuzzy matching by cleaning model names
        print("\nWARN: No exact model name match. Listing available models:")
        print(f"  Debate models: {sorted(debate_models)}")
        print(f"  SC models: {sorted(sc_models)}")
        common_models = sorted(sc_models)  # SC models are authoritative

    results = []

    for model in common_models:
        print(f"\n--- {model} ---")

        # Debate metrics (aggregate across all repeats, then per-case majority)
        d = debate_df[debate_df["model_name"] == model]
        if len(d) > 0 and "prediction" in d.columns and "ground_truth" in d.columns:
            d_metrics = metrics_from_predictions(d["prediction"], d["ground_truth"])
            d_pca = per_case_accuracy(d["prediction"], d["ground_truth"])
            print(f"  Debate:    Acc={d_metrics['accuracy']:.3f} F1={d_metrics['f1']:.3f} "
                  f"YES%={d_metrics['yes_pct']:.3f} BRD={d_metrics['brd']:.3f} N={d_metrics['n']}")
        else:
            d_metrics = {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}
            d_pca = np.array([])

        # SC majority vote metrics
        sc = sc_per_case[sc_per_case["model_name"] == model]
        if len(sc) > 0:
            sc_pred = sc["mv_prediction"]
            sc_truth = sc["ground_truth"].astype(bool)
            sc_metrics = metrics_from_predictions(sc_pred, sc_truth)
            sc_pca = per_case_accuracy(sc_pred, sc_truth)
            k = int(sc["n_reps"].median())
            print(f"  SC-K={k}:   Acc={sc_metrics['accuracy']:.3f} F1={sc_metrics['f1']:.3f} "
                  f"YES%={sc_metrics['yes_pct']:.3f} BRD={sc_metrics['brd']:.3f} N={sc_metrics['n']}")
        else:
            sc_metrics = {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}
            sc_pca = np.array([])
            k = 0

        # Zero-shot (K=1) metrics
        zs = zs_df[zs_df["model_name"] == model]
        if len(zs) > 0:
            zs_pred = zs["prediction"].astype(str).str.strip().str.upper()
            zs_truth = zs["ground_truth"].astype(bool)
            zs_metrics = metrics_from_predictions(zs_pred, zs_truth)
            print(f"  Zero-shot: Acc={zs_metrics['accuracy']:.3f} F1={zs_metrics['f1']:.3f} "
                  f"YES%={zs_metrics['yes_pct']:.3f} BRD={zs_metrics['brd']:.3f} N={zs_metrics['n']}")
        else:
            zs_metrics = {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}

        # Mann-Whitney U test: debate vs SC per-case accuracy
        mw_stat, mw_p = np.nan, np.nan
        if len(d_pca) > 0 and len(sc_pca) > 0:
            try:
                mw_result = stats.mannwhitneyu(d_pca, sc_pca, alternative="two-sided")
                mw_stat, mw_p = mw_result.statistic, mw_result.pvalue
                print(f"  Mann-Whitney U (debate vs SC): U={mw_stat:.1f}, p={mw_p:.4f}")
            except ValueError:
                print("  Mann-Whitney U: could not compute (identical arrays?)")

        results.append({
            "model": model,
            "dataset": args.dataset,
            # Debate
            "debate_accuracy": d_metrics["accuracy"],
            "debate_f1": d_metrics["f1"],
            "debate_yes_pct": d_metrics["yes_pct"],
            "debate_brd": d_metrics["brd"],
            "debate_n": d_metrics["n"],
            # SC
            f"sc_k": k,
            "sc_accuracy": sc_metrics["accuracy"],
            "sc_f1": sc_metrics["f1"],
            "sc_yes_pct": sc_metrics["yes_pct"],
            "sc_brd": sc_metrics["brd"],
            "sc_n": sc_metrics["n"],
            # Zero-shot
            "zs_accuracy": zs_metrics["accuracy"],
            "zs_f1": zs_metrics["f1"],
            "zs_yes_pct": zs_metrics["yes_pct"],
            "zs_brd": zs_metrics["brd"],
            "zs_n": zs_metrics["n"],
            # Stat test
            "mw_u_stat": mw_stat,
            "mw_p_value": mw_p,
            # Deltas
            "debate_minus_sc_accuracy": d_metrics["accuracy"] - sc_metrics["accuracy"]
            if not (np.isnan(d_metrics["accuracy"]) or np.isnan(sc_metrics["accuracy"])) else np.nan,
            "debate_minus_sc_brd": d_metrics["brd"] - sc_metrics["brd"]
            if not (np.isnan(d_metrics["brd"]) or np.isnan(sc_metrics["brd"])) else np.nan,
            "base_rate": d_metrics.get("base_rate", sc_metrics.get("base_rate", np.nan)),
        })

    results_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    if len(results_df) > 0:
        summary_cols = ["model", "debate_accuracy", "sc_accuracy", "zs_accuracy",
                        "debate_brd", "sc_brd", "mw_p_value"]
        existing_cols = [c for c in summary_cols if c in results_df.columns]
        print(results_df[existing_cols].to_string(index=False, float_format="%.3f"))

        # Aggregate across models
        print("\n--- Averages across models ---")
        for method, prefix in [("Debate", "debate"), ("SC", "sc"), ("Zero-shot", "zs")]:
            acc_col = f"{prefix}_accuracy"
            brd_col = f"{prefix}_brd"
            if acc_col in results_df.columns:
                mean_acc = results_df[acc_col].mean()
                mean_brd = results_df[brd_col].mean() if brd_col in results_df.columns else np.nan
                print(f"  {method:12s}: Acc={mean_acc:.3f}, BRD={mean_brd:.3f}")

    # Save
    output_path = Path(args.output) if args.output else Path(
        f"results/comparison_{args.dataset}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
