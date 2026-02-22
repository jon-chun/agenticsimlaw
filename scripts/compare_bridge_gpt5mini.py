#!/usr/bin/env python3
"""
Bridge Comparison: GPT-5-mini vs GPT-4o-mini debate results.

Compares the replacement model (gpt-5-mini) against the model being sunset
(gpt-4o-mini, Feb 27 2026) on key metrics: accuracy, F1, YES%, BRD, and
per-case agreement rate.

Usage:
    python scripts/compare_bridge_gpt5mini.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths — bridge data (gpt-5-mini only) and expansion data (gpt-4o-mini + others)
BRIDGE_PATHS = {
    "nlsy97": os.path.join(REPO_ROOT, "results", "bridge_gpt5mini_nlsy97.csv"),
    "compas": os.path.join(REPO_ROOT, "results", "bridge_gpt5mini_compas.csv"),
}
DEBATE_PATHS = {
    "nlsy97": os.path.join(REPO_ROOT, "results", "debate_aggregate_nlsy97_200.csv"),
    "compas": os.path.join(REPO_ROOT, "results", "debate_aggregate_compas_200.csv"),
}

BASE_RATES = {"nlsy97": 0.36, "compas": 0.45}


def compute_metrics(df: pd.DataFrame, base_rate: float) -> dict:
    """Compute accuracy, F1, YES%, BRD from a debate dataframe."""
    valid = df[df["prediction"].isin(["YES", "NO"])].copy()
    if len(valid) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}

    gt = valid["ground_truth"].astype(bool)
    pred = valid["prediction"]

    tp = ((pred == "YES") & gt).sum()
    tn = ((pred == "NO") & ~gt).sum()
    fp = ((pred == "YES") & ~gt).sum()
    fn = ((pred == "NO") & gt).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_pct = (pred == "YES").mean()
    brd = abs(yes_pct - base_rate)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "yes_pct": yes_pct,
        "brd": brd,
        "n": int(total),
    }


def majority_vote_metrics(df: pd.DataFrame, base_rate: float) -> dict:
    """Compute metrics using majority vote across repeats per case."""
    valid = df[df["prediction"].isin(["YES", "NO"])].copy()
    if len(valid) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "yes_pct": np.nan, "brd": np.nan, "n": 0}

    # Majority vote per case
    votes = valid.groupby("case_id").apply(
        lambda g: pd.Series({
            "mv_pred": "YES" if (g["prediction"] == "YES").sum() > len(g) / 2 else "NO",
            "ground_truth": g["ground_truth"].iloc[0],
        })
    )
    gt = votes["ground_truth"].astype(bool)
    pred = votes["mv_pred"]

    tp = ((pred == "YES") & gt).sum()
    tn = ((pred == "NO") & ~gt).sum()
    fp = ((pred == "YES") & ~gt).sum()
    fn = ((pred == "NO") & gt).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_pct = (pred == "YES").mean()
    brd = abs(yes_pct - base_rate)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "yes_pct": yes_pct,
        "brd": brd,
        "n": int(total),
    }


def per_case_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Compute case-level agreement between two models using majority vote.

    Returns fraction of cases where both models' majority vote agrees.
    """
    def get_mv(df):
        valid = df[df["prediction"].isin(["YES", "NO"])]
        return valid.groupby("case_id")["prediction"].apply(
            lambda s: "YES" if (s == "YES").sum() > len(s) / 2 else "NO"
        )

    mv_a = get_mv(df_a)
    mv_b = get_mv(df_b)

    common_cases = sorted(set(mv_a.index) & set(mv_b.index))
    if not common_cases:
        return np.nan

    agree = sum(mv_a[c] == mv_b[c] for c in common_cases)
    return agree / len(common_cases)


def main():
    print("=" * 70)
    print("BRIDGE COMPARISON: GPT-5-mini vs GPT-4o-mini")
    print("=" * 70)

    for dataset in ["nlsy97", "compas"]:
        bridge_path = BRIDGE_PATHS[dataset]
        debate_path = DEBATE_PATHS[dataset]
        base_rate = BASE_RATES[dataset]

        if not os.path.exists(bridge_path):
            print(f"\n  {dataset.upper()}: Bridge CSV not found ({bridge_path}), skipping.")
            continue
        if not os.path.exists(debate_path):
            print(f"\n  {dataset.upper()}: Debate CSV not found ({debate_path}), skipping.")
            continue

        bridge_df = pd.read_csv(bridge_path)
        debate_df = pd.read_csv(debate_path)

        # Filter to gpt-5-mini (bridge) and gpt-4o-mini (debate)
        gpt5 = bridge_df[bridge_df["model_name"] == "gpt-5-mini"]
        gpt4o = debate_df[debate_df["model_name"] == "gpt-4o-mini"]

        # Also get first 100 cases of gpt-4o-mini for apples-to-apples
        gpt4o_100 = gpt4o[gpt4o["case_id"] < 100]

        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset.upper()} (base rate = {base_rate:.0%})")
        print(f"{'='*70}")
        print(f"  GPT-5-mini: {len(gpt5)} predictions, {gpt5['case_id'].nunique()} cases")
        print(f"  GPT-4o-mini: {len(gpt4o)} predictions, {gpt4o['case_id'].nunique()} cases")
        print(f"  GPT-4o-mini (first 100): {len(gpt4o_100)} predictions, {gpt4o_100['case_id'].nunique()} cases")

        # Per-prediction metrics
        print(f"\n  --- Per-prediction metrics ---")
        for label, df in [("GPT-5-mini", gpt5), ("GPT-4o-mini (all)", gpt4o), ("GPT-4o-mini (100)", gpt4o_100)]:
            m = compute_metrics(df, base_rate)
            print(f"    {label:25s}: Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
                  f"YES%={m['yes_pct']:.3f}  BRD={m['brd']:.3f}  N={m['n']}")

        # Majority-vote metrics
        print(f"\n  --- Majority-vote metrics ---")
        for label, df in [("GPT-5-mini", gpt5), ("GPT-4o-mini (all)", gpt4o), ("GPT-4o-mini (100)", gpt4o_100)]:
            m = majority_vote_metrics(df, base_rate)
            print(f"    {label:25s}: Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
                  f"YES%={m['yes_pct']:.3f}  BRD={m['brd']:.3f}  N={m['n']}")

        # Case-level agreement (same 100 cases)
        common_cases = sorted(set(gpt5["case_id"].unique()) & set(gpt4o["case_id"].unique()))
        if common_cases:
            gpt5_common = gpt5[gpt5["case_id"].isin(common_cases)]
            gpt4o_common = gpt4o[gpt4o["case_id"].isin(common_cases)]
            agreement = per_case_agreement(gpt5_common, gpt4o_common)
            print(f"\n  --- Case-level agreement (majority vote, {len(common_cases)} shared cases) ---")
            print(f"    Agreement rate: {agreement:.1%}")

        # BRD comparison summary
        gpt5_brd = compute_metrics(gpt5, base_rate)["brd"]
        gpt4o_brd = compute_metrics(gpt4o, base_rate)["brd"]
        print(f"\n  --- BRD Summary ---")
        print(f"    GPT-5-mini BRD:  {gpt5_brd:.3f}")
        print(f"    GPT-4o-mini BRD: {gpt4o_brd:.3f}")
        print(f"    Delta:           {gpt5_brd - gpt4o_brd:+.3f}")
        if gpt5_brd < 0.10:
            print(f"    GPT-5-mini shows strong calibration (BRD < 10%)")
        elif gpt5_brd < gpt4o_brd * 1.5:
            print(f"    GPT-5-mini comparable to GPT-4o-mini")
        else:
            print(f"    GPT-5-mini shows higher BRD than GPT-4o-mini")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
