#!/usr/bin/env python3
"""
Generate 100-case vignette CSVs for W1 expanded-sample experiments.

1. NLSY97: Sample 100 cases from full dataset (seed=42).
2. COMPAS: Sample 100 from compas_full_filtered.csv with stratified sampling
   (target × race group), recompute LOFO feature importance.

Usage:
    python src/expand_vignettes.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NLSY97_FULL_DST = DATA_DIR / "nlsy97_full.csv"
NLSY97_100_DST = DATA_DIR / "nlsy97_vignettes_100.csv"

COMPAS_FULL_PATH = DATA_DIR / "compas_full_filtered.csv"
COMPAS_100_DST = DATA_DIR / "compas_vignettes_100.csv"
COMPAS_100_FEATURE_PATH = DATA_DIR / "compas_100_topn_feature_by_algo.json"

SAMPLE_N = 100
SEED = 42

# COMPAS feature columns (same as prepare_compas_dataset.py)
COMPAS_FEATURE_COLS = [
    "age", "sex", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree", "decile_score",
]
COMPAS_TARGET = "two_year_recid"


# ---------------------------------------------------------------------------
# NLSY97
# ---------------------------------------------------------------------------

def expand_nlsy97() -> None:
    """Copy full NLSY97 dataset and sample 100 cases."""
    print("=" * 60)
    print("NLSY97: Expanding to 100 cases")
    print("=" * 60)

    if not NLSY97_FULL_DST.exists():
        raise FileNotFoundError(
            f"NLSY97 full dataset not found at {NLSY97_FULL_DST}"
        )

    df = pd.read_csv(NLSY97_FULL_DST)
    print(f"Full dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Target balance: {df['y_arrestedafter2002'].value_counts().to_dict()}")

    # Sample 100 cases with seed for reproducibility
    np.random.seed(SEED)
    df_sample = df.sample(n=SAMPLE_N, random_state=SEED).copy()
    df_sample = df_sample.reset_index(drop=True)
    df_sample["id"] = range(1, len(df_sample) + 1)

    # Reorder to put id first (matching existing sample_vignettes.csv schema)
    cols = ["id"] + [c for c in df_sample.columns if c != "id"]
    df_sample = df_sample[cols]

    df_sample.to_csv(NLSY97_100_DST, index=False)
    print(f"Saved 100-case sample to {NLSY97_100_DST}")
    print(f"  Target balance: {df_sample['y_arrestedafter2002'].value_counts().to_dict()}")
    print(f"  Columns: {list(df_sample.columns)}")


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

def _race_group(race: str) -> str:
    """Collapse COMPAS race categories into 3 groups for stratification."""
    if race == "African-American":
        return "Black"
    if race == "Caucasian":
        return "White"
    return "Other"


def compute_compas_lofo(df: pd.DataFrame) -> list[str]:
    """Compute LOFO feature importance on a COMPAS subset."""
    features = [c for c in COMPAS_FEATURE_COLS if c in df.columns]
    X = df[features].copy()
    y = df[COMPAS_TARGET].values

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    baseline_scores = cross_val_score(clf, X_encoded, y, cv=4, scoring="neg_log_loss")
    baseline_mean = baseline_scores.mean()
    print(f"  LOFO baseline neg_log_loss: {baseline_mean:.4f}")

    importance = {}
    for feat in features:
        if feat in cat_cols:
            drop_cols = [c for c in X_encoded.columns if c.startswith(feat + "_")]
        else:
            drop_cols = [feat]
        X_reduced = X_encoded.drop(columns=drop_cols)
        scores = cross_val_score(clf, X_reduced, y, cv=4, scoring="neg_log_loss")
        importance[feat] = baseline_mean - scores.mean()

    ranked = sorted(importance.keys(), key=lambda f: importance[f], reverse=True)
    print("  LOFO ranking:")
    for i, feat in enumerate(ranked, 1):
        print(f"    {i}. {feat}: {importance[feat]:.6f}")
    return ranked


def expand_compas() -> None:
    """Stratified-sample 100 COMPAS cases and recompute LOFO."""
    print("\n" + "=" * 60)
    print("COMPAS: Expanding to 100 cases")
    print("=" * 60)

    if not COMPAS_FULL_PATH.exists():
        raise FileNotFoundError(f"COMPAS full filtered dataset not found at {COMPAS_FULL_PATH}")

    df = pd.read_csv(COMPAS_FULL_PATH)
    print(f"Full dataset: {len(df)} rows")
    print(f"  Target balance: {df[COMPAS_TARGET].value_counts().to_dict()}")

    # Stratified sampling by target × race group
    df_work = df.copy()
    df_work["_race_group"] = df_work["race"].apply(_race_group)
    df_work["_strat_key"] = df_work[COMPAS_TARGET].astype(str) + "_" + df_work["_race_group"]

    # Filter out strata too small for stratification
    strat_counts = df_work["_strat_key"].value_counts()
    valid_strata = strat_counts[strat_counts >= 2].index
    df_valid = df_work[df_work["_strat_key"].isin(valid_strata)]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_N, random_state=SEED)
    sample_idx, _ = next(sss.split(df_valid, df_valid["_strat_key"]))
    df_sample = df_valid.iloc[sample_idx].copy()

    # Clean up helper columns and add id
    df_sample = df_sample.drop(columns=["_race_group", "_strat_key"])
    df_sample = df_sample.reset_index(drop=True)
    df_sample.insert(0, "id", range(1, len(df_sample) + 1))

    df_sample.to_csv(COMPAS_100_DST, index=False)
    print(f"Saved 100-case sample to {COMPAS_100_DST}")
    print(f"  Target balance: {df_sample[COMPAS_TARGET].value_counts().to_dict()}")
    print(f"  Race distribution: {df_sample['race'].value_counts().to_dict()}")

    # Recompute LOFO on the 100-case subset
    print("\nComputing LOFO feature importance on 100-case subset...")
    lofo_ranking = compute_compas_lofo(df_sample.drop(columns=["id"]))

    with open(COMPAS_100_FEATURE_PATH, "w") as f:
        json.dump({"lofo": lofo_ranking}, f, indent=4)
    print(f"Saved feature importance to {COMPAS_100_FEATURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expand_nlsy97()
    expand_compas()

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print(f"  {NLSY97_FULL_DST}")
    print(f"  {NLSY97_100_DST}")
    print(f"  {COMPAS_100_DST}")
    print(f"  {COMPAS_100_FEATURE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
