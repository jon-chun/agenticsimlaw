#!/usr/bin/env python3
"""
Self-Consistency (SC) majority-vote aggregation.

Reads a StandardLLM results CSV (from standardllm_evaluation.py with K>1 repeats),
groups predictions by (model, prompt_type, case_id), applies majority vote,
and computes accuracy, F1, YES%, and Base Rate Deviation (BRD).

Usage:
    python src/sc_majority_vote.py results/standardllm/sc_nlsy97_results.csv
    python src/sc_majority_vote.py results/standardllm/sc_compas_results.csv --output results/standardllm/sc_nlsy97_majority.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def majority_vote(predictions: pd.Series) -> str:
    """Return the majority prediction (YES/NO) from a series of predictions."""
    valid = predictions.dropna()
    if len(valid) == 0:
        return "NO_DECISION"
    counts = valid.value_counts()
    return counts.index[0]  # most frequent


def compute_brd(yes_pct: float, base_rate: float) -> float:
    """Base Rate Deviation: |YES% - base_rate|."""
    return abs(yes_pct - base_rate)


def aggregate_sc_results(df: pd.DataFrame) -> pd.DataFrame:
    """Apply majority vote per (model, prompt_type, case_id) and compute per-case accuracy."""
    # Group and apply majority vote
    grouped = df.groupby(["model_name", "prompt_type", "case_id"]).agg(
        mv_prediction=("prediction", majority_vote),
        ground_truth=("ground_truth", "first"),
        n_reps=("prediction", "count"),
        n_valid=("prediction", lambda x: x.dropna().shape[0]),
        yes_votes=("prediction", lambda x: (x == "YES").sum()),
        no_votes=("prediction", lambda x: (x == "NO").sum()),
    ).reset_index()

    # Compute per-case correctness
    grouped["mv_correct"] = None
    mask_yes = grouped["mv_prediction"] == "YES"
    mask_no = grouped["mv_prediction"] == "NO"
    grouped.loc[mask_yes, "mv_correct"] = grouped.loc[mask_yes, "ground_truth"].astype(bool)
    grouped.loc[mask_no, "mv_correct"] = ~grouped.loc[mask_no, "ground_truth"].astype(bool)

    return grouped


def compute_metrics(per_case: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per (model, prompt_type)."""
    metrics = []

    for (model, prompt), group in per_case.groupby(["model_name", "prompt_type"]):
        valid = group[group["mv_prediction"].isin(["YES", "NO"])]
        if len(valid) == 0:
            continue

        tp = ((valid["mv_prediction"] == "YES") & (valid["ground_truth"].astype(bool))).sum()
        tn = ((valid["mv_prediction"] == "NO") & (~valid["ground_truth"].astype(bool))).sum()
        fp = ((valid["mv_prediction"] == "YES") & (~valid["ground_truth"].astype(bool))).sum()
        fn = ((valid["mv_prediction"] == "NO") & (valid["ground_truth"].astype(bool))).sum()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        yes_pct = (valid["mv_prediction"] == "YES").mean()
        base_rate = valid["ground_truth"].astype(bool).mean()
        brd = compute_brd(yes_pct, base_rate)

        avg_reps = group["n_reps"].mean()

        metrics.append({
            "model_name": model,
            "prompt_type": prompt,
            "method": f"SC-K={int(avg_reps)}",
            "n_cases": len(valid),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "yes_pct": yes_pct,
            "base_rate": base_rate,
            "brd": brd,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="SC Majority Vote Aggregation")
    parser.add_argument("input_csv", type=str, help="Path to StandardLLM results CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <input>_majority_vote.csv)")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Loading SC results from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")
    print(f"  Models: {df['model_name'].nunique()}")
    print(f"  Repeats per case: {df.groupby(['model_name', 'prompt_type', 'case_id']).size().median():.0f}")

    # Aggregate via majority vote
    per_case = aggregate_sc_results(df)
    print(f"\nMajority-vote aggregated: {len(per_case)} case-level results")

    # Compute metrics
    metrics_df = compute_metrics(per_case)

    # Print summary table
    print("\n" + "=" * 80)
    print("SC MAJORITY VOTE RESULTS")
    print("=" * 80)
    for _, row in metrics_df.iterrows():
        print(f"\n  {row['model_name']} | {row['prompt_type']} | {row['method']}")
        print(f"    Accuracy: {row['accuracy']:.3f} | F1: {row['f1_score']:.3f} | "
              f"YES%: {row['yes_pct']:.3f} | BRD: {row['brd']:.3f}")
        print(f"    TP={row['tp']} TN={row['tn']} FP={row['fp']} FN={row['fn']} | N={row['n_cases']}")

    # Save outputs
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_majority_vote.csv"
    )
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to: {output_path}")

    # Also save per-case results for downstream comparison
    per_case_path = output_path.with_name(output_path.stem + "_per_case.csv")
    per_case.to_csv(per_case_path, index=False)
    print(f"Saved per-case results to: {per_case_path}")


if __name__ == "__main__":
    main()
