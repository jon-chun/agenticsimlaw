#!/usr/bin/env python3
"""Prepare UCI Credit Default dataset for MADCal experiments.

Downloads the UCI Credit Default dataset (30,000 records, 24 features),
applies feature decoding for text serialization, computes LOFO feature
importance, and saves a stratified 30-case vignette sample.

Output files (in data/):
  - credit_default_full_filtered.csv    (full decoded dataset)
  - credit_default_vignettes.csv        (30-case stratified sample)
  - credit_default_topn_feature_by_algo.json  (LOFO feature rankings)

Usage:
  python src/prepare_credit_dataset.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# ── paths ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "credit_default_raw.xls"
FULL_FILTERED_PATH = DATA_DIR / "credit_default_full_filtered.csv"
VIGNETTES_PATH = DATA_DIR / "credit_default_vignettes.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "credit_default_topn_feature_by_algo.json"

# ── dataset constants ──────────────────────────────────
CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
TARGET_COL = "default_payment_next_month"
SAMPLE_N = 30
SEED = 42

# Features to keep (10 most interpretable of 23 original features).
# Full dataset has PAY_0..PAY_6, BILL_AMT1..6, PAY_AMT1..6 —
# we keep a representative subset to match NLSY97/COMPAS feature count.
FEATURE_COLS = [
    "LIMIT_BAL",   # credit limit
    "SEX",         # 1=male, 2=female
    "EDUCATION",   # 1=grad school, 2=university, 3=high school, 4=other
    "MARRIAGE",    # 1=married, 2=single, 3=other
    "AGE",
    "PAY_0",       # repayment status last month (-1=on time, 1-9=months late)
    "PAY_2",       # repayment status 2 months ago
    "BILL_AMT1",   # most recent bill (TWD)
    "PAY_AMT1",    # most recent payment (TWD)
    "PAY_AMT2",    # payment 2 months ago (TWD)
]


# ── step 1: download ──────────────────────────────────
def download_credit():
    """Download raw XLS from UCI repository."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print(f"  Raw file already exists: {RAW_PATH}")
        return pd.read_excel(RAW_PATH, header=1)
    print(f"  Downloading from {CREDIT_URL} ...")
    urllib.request.urlretrieve(CREDIT_URL, RAW_PATH)
    return pd.read_excel(RAW_PATH, header=1)


# ── step 2: decode categoricals ──────────────────────
def decode_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Decode categorical codes to human-readable strings for text serialization."""
    # Rename target column (varies across XLS versions)
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": TARGET_COL})
    elif "Y" in df.columns:
        df = df.rename(columns={"Y": TARGET_COL})

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Decode SEX
    df["SEX"] = df["SEX"].map({1: "male", 2: "female"})

    # Decode EDUCATION
    edu_map = {
        1: "graduate school", 2: "university",
        3: "high school", 4: "other",
        5: "other", 6: "other", 0: "other",
    }
    df["EDUCATION"] = df["EDUCATION"].map(edu_map).fillna("other")

    # Decode MARRIAGE
    mar_map = {1: "married", 2: "single", 3: "other", 0: "other"}
    df["MARRIAGE"] = df["MARRIAGE"].map(mar_map).fillna("other")

    # Decode PAY_0, PAY_2 (repayment status)
    for col in ["PAY_0", "PAY_2"]:
        df[col] = df[col].apply(
            lambda x: "on time" if x <= 0 else f"{x} month(s) late"
        )

    # Select features + target, drop rows with any NaN
    keep = FEATURE_COLS + [TARGET_COL]
    df = df[keep].dropna().reset_index(drop=True)
    print(f"  After decode+filter: {len(df)} records, "
          f"base rate = {df[TARGET_COL].mean():.1%}")
    return df


# ── step 3: stratified sample ─────────────────────────
def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Draw a stratified 30-case sample preserving target distribution."""
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=SAMPLE_N, random_state=SEED
    )
    idx, _ = next(sss.split(df, df[TARGET_COL]))
    sample = df.iloc[idx].reset_index(drop=True)
    sample.insert(0, "id", range(1, len(sample) + 1))
    print(f"  Sampled {len(sample)} vignettes, "
          f"base rate = {sample[TARGET_COL].mean():.1%}")
    return sample


# ── step 4: LOFO feature importance ───────────────────
def compute_lofo_importance(df: pd.DataFrame) -> dict:
    """Compute Leave-One-Feature-Out importance using RandomForest + 4-fold CV."""
    # Encode categoricals for RF
    X = df[FEATURE_COLS].copy()
    for col in X.select_dtypes(include="object"):
        X[col] = X[col].astype("category").cat.codes
    y = df[TARGET_COL]

    # Baseline score (all features)
    base_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        X, y, cv=4, scoring="neg_log_loss",
    )
    base_mean = base_scores.mean()
    print(f"  LOFO baseline neg_log_loss: {base_mean:.4f}")

    # Drop each feature and measure degradation
    importance = {}
    for feat in FEATURE_COLS:
        X_drop = X.drop(columns=[feat])
        drop_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=SEED),
            X_drop, y, cv=4, scoring="neg_log_loss",
        )
        # Higher degradation = more important feature
        importance[feat] = base_mean - drop_scores.mean()

    # Rank by importance (descending)
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    lofo_ranking = {}
    for i, (feat, score) in enumerate(ranked):
        lofo_ranking[str(i + 1)] = [feat, round(score, 6)]

    print(f"  Top-3 features: "
          f"{ranked[0][0]}, {ranked[1][0]}, {ranked[2][0]}")
    return lofo_ranking


# ── step 5: save outputs ──────────────────────────────
def save_outputs(
    full_df: pd.DataFrame,
    vignettes: pd.DataFrame,
    lofo_ranking: dict,
):
    """Save full dataset, vignettes CSV, and feature importance JSON."""
    full_df.to_csv(FULL_FILTERED_PATH, index=False)
    print(f"  Saved full dataset: {FULL_FILTERED_PATH} ({len(full_df)} rows)")

    vignettes.to_csv(VIGNETTES_PATH, index=False)
    print(f"  Saved vignettes: {VIGNETTES_PATH} ({len(vignettes)} rows)")

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump({"lofo": lofo_ranking}, f, indent=2)
    print(f"  Saved feature importance: {FEATURE_IMPORTANCE_PATH}")


# ── main ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Preparing UCI Credit Default dataset")
    print("=" * 60)

    print("\n1. Downloading raw data...")
    raw_df = download_credit()
    print(f"   Raw: {raw_df.shape[0]} records × {raw_df.shape[1]} columns")

    print("\n2. Decoding categoricals and filtering...")
    full_df = decode_and_filter(raw_df)

    print("\n3. Drawing stratified sample...")
    vignettes = stratified_sample(full_df)

    print("\n4. Computing LOFO feature importance...")
    lofo = compute_lofo_importance(full_df)

    print("\n5. Saving outputs...")
    save_outputs(full_df, vignettes, lofo)

    print("\n" + "=" * 60)
    print(f"Done. {len(vignettes)} vignettes, "
          f"base rate = {vignettes[TARGET_COL].mean():.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
