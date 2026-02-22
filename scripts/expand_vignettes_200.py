#!/usr/bin/env python3
"""
Generate 200-case vignette CSVs as supersets of existing 100-case files.

The 200-case sample includes all 100 cases from the existing 100-case file
plus 100 additional cases sampled from the full dataset (seed=200).
This ensures existing 100-case debate results can be reused.

Usage:
    python scripts/expand_vignettes_200.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NLSY97_FULL = DATA_DIR / "nlsy97_full.csv"
NLSY97_100 = DATA_DIR / "nlsy97_vignettes_100.csv"
NLSY97_200 = DATA_DIR / "nlsy97_vignettes_200.csv"

COMPAS_FULL = DATA_DIR / "compas_full_filtered.csv"
COMPAS_100 = DATA_DIR / "compas_vignettes_100.csv"
COMPAS_200 = DATA_DIR / "compas_vignettes_200.csv"
COMPAS_200_FEATURE = DATA_DIR / "compas_200_topn_feature_by_algo.json"

EXTRA_N = 100
SEED = 200

COMPAS_FEATURE_COLS = [
    "age", "sex", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree", "decile_score",
]
COMPAS_TARGET = "two_year_recid"


def expand_nlsy97():
    print("=" * 60)
    print("NLSY97: Expanding to 200 cases (superset of 100)")
    print("=" * 60)

    df_full = pd.read_csv(NLSY97_FULL)
    df_100 = pd.read_csv(NLSY97_100)
    print(f"Full dataset: {len(df_full)} rows")
    print(f"Existing 100-case: {len(df_100)} rows")

    # Drop 'id' column from 100-case for matching
    match_cols = [c for c in df_100.columns if c != "id"]

    # Find rows in full that are already in the 100-case sample
    # Use a merge to identify matching rows
    df_100_no_id = df_100[match_cols].copy()
    df_full_indexed = df_full.reset_index()
    merged = df_full_indexed.merge(df_100_no_id, on=match_cols, how="inner")
    used_indices = set(merged["index"].values)
    print(f"Matched {len(used_indices)} existing cases in full dataset")

    # Sample additional cases from remaining pool
    remaining = df_full.loc[~df_full.index.isin(used_indices)]
    print(f"Remaining pool: {len(remaining)} cases")

    np.random.seed(SEED)
    extra = remaining.sample(n=EXTRA_N, random_state=SEED).copy()
    print(f"Sampled {len(extra)} additional cases")

    # Combine: existing 100 + new 100
    df_100_clean = df_100.drop(columns=["id"], errors="ignore")
    extra_clean = extra.drop(columns=["id"], errors="ignore")
    df_200 = pd.concat([df_100_clean, extra_clean], ignore_index=True)
    df_200.insert(0, "id", range(1, len(df_200) + 1))

    df_200.to_csv(NLSY97_200, index=False)
    target = "y_arrestedafter2002"
    print(f"Saved 200-case sample to {NLSY97_200}")
    print(f"  Target balance: {df_200[target].value_counts().to_dict()}")
    print(f"  Base rate (YES=True): {df_200[target].mean():.3f}")


def expand_compas():
    print("\n" + "=" * 60)
    print("COMPAS: Expanding to 200 cases (superset of 100)")
    print("=" * 60)

    df_full = pd.read_csv(COMPAS_FULL)
    df_100 = pd.read_csv(COMPAS_100)
    print(f"Full dataset: {len(df_full)} rows")
    print(f"Existing 100-case: {len(df_100)} rows")

    # Find used indices
    match_cols = [c for c in df_100.columns if c != "id"]
    df_100_no_id = df_100[match_cols].copy()
    df_full_indexed = df_full.reset_index()
    merged = df_full_indexed.merge(df_100_no_id, on=match_cols, how="inner")
    used_indices = set(merged["index"].values)
    print(f"Matched {len(used_indices)} existing cases in full dataset")

    remaining = df_full.loc[~df_full.index.isin(used_indices)].copy()
    print(f"Remaining pool: {len(remaining)} cases")

    # Stratified sampling for additional 100
    def race_group(race):
        if race == "African-American":
            return "Black"
        if race == "Caucasian":
            return "White"
        return "Other"

    remaining["_strat"] = remaining[COMPAS_TARGET].astype(str) + "_" + remaining["race"].apply(race_group)
    strat_counts = remaining["_strat"].value_counts()
    valid = strat_counts[strat_counts >= 2].index
    remaining_valid = remaining[remaining["_strat"].isin(valid)]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=EXTRA_N, random_state=SEED)
    idx, _ = next(sss.split(remaining_valid, remaining_valid["_strat"]))
    extra = remaining_valid.iloc[idx].drop(columns=["_strat"]).copy()
    print(f"Sampled {len(extra)} additional cases (stratified)")

    # Combine
    df_100_clean = df_100.drop(columns=["id"], errors="ignore")
    extra_clean = extra.drop(columns=["id"], errors="ignore")
    df_200 = pd.concat([df_100_clean, extra_clean], ignore_index=True)
    df_200.insert(0, "id", range(1, len(df_200) + 1))

    df_200.to_csv(COMPAS_200, index=False)
    print(f"Saved 200-case sample to {COMPAS_200}")
    print(f"  Target balance: {df_200[COMPAS_TARGET].value_counts().to_dict()}")
    print(f"  Base rate: {df_200[COMPAS_TARGET].mean():.3f}")
    print(f"  Race distribution: {df_200['race'].value_counts().to_dict()}")

    # Recompute LOFO
    print("\nComputing LOFO feature importance on 200-case subset...")
    features = [c for c in COMPAS_FEATURE_COLS if c in df_200.columns]
    X = df_200.drop(columns=["id"])[features].copy()
    y = df_200[COMPAS_TARGET].values

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    baseline = cross_val_score(clf, X_enc, y, cv=4, scoring="neg_log_loss").mean()
    print(f"  LOFO baseline: {baseline:.4f}")

    importance = {}
    for feat in features:
        if feat in cat_cols:
            drop = [c for c in X_enc.columns if c.startswith(feat + "_")]
        else:
            drop = [feat]
        reduced = X_enc.drop(columns=drop)
        score = cross_val_score(clf, reduced, y, cv=4, scoring="neg_log_loss").mean()
        importance[feat] = baseline - score

    ranked = sorted(importance, key=lambda f: importance[f], reverse=True)
    for i, f in enumerate(ranked, 1):
        print(f"    {i}. {f}: {importance[f]:.6f}")

    with open(COMPAS_200_FEATURE, "w") as fp:
        json.dump({"lofo": ranked}, fp, indent=4)
    print(f"Saved feature importance to {COMPAS_200_FEATURE}")


def main():
    expand_nlsy97()
    expand_compas()
    print("\n" + "=" * 60)
    print("Done! Generated:")
    print(f"  {NLSY97_200}")
    print(f"  {COMPAS_200}")
    print(f"  {COMPAS_200_FEATURE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
