"""
Prepare COMPAS (ProPublica) dataset for AgenticSimLaw experiments.

Downloads the COMPAS recidivism dataset, applies ProPublica's standard filtering,
selects features, stratified-samples 30 cases, computes LOFO feature importance,
and saves output files mirroring the NLSY97 data structure.

Usage:
    python src/prepare_compas_dataset.py
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "compas_raw.csv"
VIGNETTES_PATH = DATA_DIR / "compas_vignettes.csv"
FULL_FILTERED_PATH = DATA_DIR / "compas_full_filtered.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "compas_topn_feature_by_algo.json"

TARGET_COL = "two_year_recid"
SAMPLE_N = 30
SEED = 42

FEATURE_COLS = [
    "age",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "decile_score",
]

COMPAS_FEATURE_DESCRIPTIONS = {
    "age": "age at COMPAS screening",
    "sex": "sex",
    "race": "race",
    "juv_fel_count": "juvenile felony count",
    "juv_misd_count": "juvenile misdemeanor count",
    "juv_other_count": "juvenile other offense count",
    "priors_count": "number of prior adult offenses",
    "c_charge_degree": "current charge degree",
    "decile_score": "COMPAS recidivism risk score (1-10)",
}


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

def download_compas() -> pd.DataFrame:
    """Download COMPAS CSV from ProPublica GitHub to data/compas_raw.csv."""
    print(f"Downloading COMPAS data from {COMPAS_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(COMPAS_URL, RAW_PATH)
    print(f"Saved raw data to {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


# ---------------------------------------------------------------------------
# Step 2: ProPublica standard filters
# ---------------------------------------------------------------------------

def apply_propublica_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ProPublica's standard COMPAS filtering criteria."""
    n_before = len(df)

    df = df[
        (df["days_b_screening_arrest"].abs() <= 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    print(f"ProPublica filter: {n_before} → {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Step 3: Select features
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 9 predictor features + target column."""
    cols = FEATURE_COLS + [TARGET_COL]
    df = df[cols].copy()
    df = df.dropna()
    print(f"Feature selection: {len(df)} rows, {len(cols)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 4: Stratified sample
# ---------------------------------------------------------------------------

def _race_group(race: str) -> str:
    """Collapse COMPAS race categories into 3 groups for stratification."""
    if race == "African-American":
        return "Black"
    if race == "Caucasian":
        return "White"
    return "Other"


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_N) -> pd.DataFrame:
    """Stratified-sample n cases balanced by target × race group."""
    df = df.copy()
    df["_race_group"] = df["race"].apply(_race_group)
    df["_strat_key"] = df[TARGET_COL].astype(str) + "_" + df["_race_group"]

    # Filter out strata that are too small (need at least 2 members per stratum)
    strat_counts = df["_strat_key"].value_counts()
    valid_strata = strat_counts[strat_counts >= 2].index
    df_valid = df[df["_strat_key"].isin(valid_strata)]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=SEED)
    sample_idx, _ = next(sss.split(df_valid, df_valid["_strat_key"]))
    sampled = df_valid.iloc[sample_idx].copy()

    # Add sequential id and drop helper columns
    sampled = sampled.drop(columns=["_race_group", "_strat_key"])
    sampled = sampled.reset_index(drop=True)
    sampled.insert(0, "id", range(1, len(sampled) + 1))

    print(f"Stratified sample: {len(sampled)} cases")
    print(f"  Target balance: {sampled[TARGET_COL].value_counts().to_dict()}")
    print(f"  Race distribution: {sampled['race'].value_counts().to_dict()}")
    return sampled


# ---------------------------------------------------------------------------
# Step 5: LOFO feature importance
# ---------------------------------------------------------------------------

def compute_lofo_importance(df: pd.DataFrame) -> list[str]:
    """
    Compute Leave-One-Feature-Out importance on the full filtered dataset.

    Uses RandomForestClassifier with 4-fold CV and neg_log_loss scoring,
    consistent with the NLSY97 LOFO methodology.

    Returns feature names ordered by importance (most important first).
    """
    from sklearn.model_selection import cross_val_score

    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].copy()
    y = df[TARGET_COL].values

    # Encode categorical features for the classifier
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Baseline score (all features)
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    baseline_scores = cross_val_score(
        clf, X_encoded, y, cv=4, scoring="neg_log_loss"
    )
    baseline_mean = baseline_scores.mean()
    print(f"LOFO baseline neg_log_loss: {baseline_mean:.4f}")

    importance = {}
    for feat in features:
        # Determine which encoded columns belong to this feature
        if feat in cat_cols:
            drop_cols = [c for c in X_encoded.columns if c.startswith(feat + "_")]
        else:
            drop_cols = [feat]

        X_reduced = X_encoded.drop(columns=drop_cols)
        scores = cross_val_score(
            clf, X_reduced, y, cv=4, scoring="neg_log_loss"
        )
        # Importance = how much worse the score gets when feature is removed
        # (more negative = worse, so baseline - reduced = positive if feature helps)
        importance[feat] = baseline_mean - scores.mean()

    # Sort by importance descending
    ranked = sorted(importance.keys(), key=lambda f: importance[f], reverse=True)
    print("LOFO feature ranking:")
    for i, feat in enumerate(ranked, 1):
        print(f"  {i}. {feat}: {importance[feat]:.6f}")

    return ranked


# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    vignettes: pd.DataFrame,
    full_filtered: pd.DataFrame,
    lofo_ranking: list[str],
) -> None:
    """Save all output files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Vignettes CSV (30 rows)
    vignettes.to_csv(VIGNETTES_PATH, index=False)
    print(f"Saved vignettes: {VIGNETTES_PATH} ({len(vignettes)} rows)")

    # Full filtered dataset
    full_filtered.to_csv(FULL_FILTERED_PATH, index=False)
    print(f"Saved full filtered: {FULL_FILTERED_PATH} ({len(full_filtered)} rows)")

    # Feature importance JSON (same format as topn_feature_by_algo.json)
    feature_json = {"lofo": lofo_ranking}
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feature_json, f, indent=4)
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("COMPAS Dataset Preparation for AgenticSimLaw")
    print("=" * 60)

    # 1. Download
    raw_df = download_compas()
    print(f"Raw dataset: {raw_df.shape}")

    # 2. Filter
    filtered_df = apply_propublica_filters(raw_df)

    # 3. Select features
    selected_df = select_features(filtered_df)

    # 4. Stratified sample
    vignettes_df = stratified_sample(selected_df)

    # 5. LOFO importance (on full filtered data)
    lofo_ranking = compute_lofo_importance(selected_df)

    # 6. Save
    save_outputs(vignettes_df, selected_df, lofo_ranking)

    # Summary
    print("\n" + "=" * 60)
    print("Done! Output files:")
    print(f"  {VIGNETTES_PATH}")
    print(f"  {FULL_FILTERED_PATH}")
    print(f"  {FEATURE_IMPORTANCE_PATH}")
    print("=" * 60)

    # Quick spot-check
    print("\nSample vignettes (first 3):")
    for _, row in vignettes_df.head(3).iterrows():
        parts = [f"{COMPAS_FEATURE_DESCRIPTIONS[c]} is {row[c]}" for c in FEATURE_COLS]
        print(f"  Case {row['id']}: {', '.join(parts)}")
        print(f"    Recidivated: {'Yes' if row[TARGET_COL] == 1 else 'No'}")


if __name__ == "__main__":
    main()
