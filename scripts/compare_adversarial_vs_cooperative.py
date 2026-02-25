#!/usr/bin/env python3
"""Compare adversarial vs cooperative debate calibration (BRD).

Usage:
  python scripts/compare_adversarial_vs_cooperative.py \
    --adv-dir results/ \
    --coop-dir results/ \
    --datasets nlsy97 compas credit_default
"""
import argparse
from pathlib import Path

import pandas as pd

BASE_RATES = {
    "nlsy97": 0.36,
    "compas": 0.45,
    "credit_default": 0.22,
}

TARGET_COLS = {
    "nlsy97": "y_arrestedafter2002",
    "compas": "two_year_recid",
    "credit_default": "default_payment_next_month",
}


def compute_brd(df: pd.DataFrame, dataset: str) -> float:
    """Compute BRD = |YES% - base_rate| from aggregate CSV."""
    # Aggregate CSV has prediction columns — compute YES%
    target = TARGET_COLS[dataset]
    base_rate = BASE_RATES[dataset]

    if "judge_prediction" in df.columns:
        yes_pct = (df["judge_prediction"] == "YES").mean()
    elif "prediction" in df.columns:
        yes_pct = (df["prediction"] == "YES").mean()
    else:
        raise ValueError(f"No prediction column found in {df.columns.tolist()}")

    return abs(yes_pct - base_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-dir", type=str, default="results",
        help="Directory containing adversarial aggregate CSVs")
    parser.add_argument("--coop-dir", type=str, default="results",
        help="Directory containing cooperative aggregate CSVs")
    parser.add_argument("--datasets", nargs="+",
        default=["nlsy97", "compas", "credit_default"])
    args = parser.parse_args()

    print(f"{'Model':<25} {'Dataset':<18} {'Adv BRD':>8} {'Coop BRD':>9} {'ZS BRD':>7}  {'Winner':<12}")
    print("-" * 85)

    for ds in args.datasets:
        # Find adversarial aggregate
        adv_path = Path(args.adv_dir) / f"debate_aggregate_{ds}_100.csv"
        coop_path = Path(args.coop_dir) / f"cooperative_aggregate_{ds}_100.csv"

        if not adv_path.exists():
            print(f"  [SKIP] {adv_path} not found")
            continue
        if not coop_path.exists():
            print(f"  [SKIP] {coop_path} not found")
            continue

        adv_df = pd.read_csv(adv_path)
        coop_df = pd.read_csv(coop_path)

        # Group by model
        for model in adv_df["model_name"].unique():
            adv_sub = adv_df[adv_df["model_name"] == model]
            coop_sub = coop_df[coop_df["model_name"] == model]

            if coop_sub.empty:
                continue

            adv_brd = compute_brd(adv_sub, ds)
            coop_brd = compute_brd(coop_sub, ds)
            winner = "adversarial" if adv_brd < coop_brd else "cooperative"

            print(f"{model:<25} {ds:<18} {adv_brd:>8.3f} {coop_brd:>9.3f} {'':>7}  {winner}")

    print()
    print("If adversarial consistently has lower BRD, this confirms")
    print("that adversarial role structure (not just multi-agent")
    print("interaction) is the operative calibration variable.")


if __name__ == "__main__":
    main()
