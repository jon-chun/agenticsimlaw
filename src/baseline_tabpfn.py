#!/usr/bin/env python3
"""
TabPFN Baseline Evaluation Script

Evaluates TabPFN (Prior-Data Fitted Networks for Tabular Data) on the
recidivism prediction task.

Paper Reference: Section 3.6 "Traditional Baseline Models" and Table 3
- TabPFN: Specialized tabular LLM for small data regimes
- Metrics: AUC, Accuracy, Recall, F1 Score, Confusion Matrix

Paper Table 3 Expected Results:
- AUC: 0.6410
- Accuracy: 0.7011
- Recall: 0.0000 (predicted no positives)
- F1 Score: 0.0000
- Confusion Matrix: [[319, 0]; [136, 0]]

Note: TabPFN is designed for datasets with <10,000 samples and <100 features.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VIGNETTES_PATH = "../data/sample_vignettes.csv"
DEFAULT_OUTPUT_DIR = "../results/baselines"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# TabPFN hyperparameters (from paper Table 3 caption)
TABPFN_PARAMS = {
    'N_ENSEMBLE_CONFIGURATIONS': 32,  # Default ensemble size
    'DEVICE': 'cpu',  # Use 'cuda' if GPU available
}


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging."""
    log_file = output_dir / f"tabpfn_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# Data Preprocessing
# ============================================================================

def load_and_preprocess(path: str, logger: logging.Logger) -> Tuple[pd.DataFrame, str]:
    """
    Load and preprocess the vignettes dataset for TabPFN.

    TabPFN requirements:
    - Numeric features only
    - No missing values
    - Binary target (0/1)

    Returns:
        Tuple of (preprocessed DataFrame, target column name)
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    target_col = 'y_arrestedafter2002'
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Convert target to int (0/1)
    df[target_col] = df[target_col].astype(int)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")

    # Handle categorical columns - TabPFN requires numeric input
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    logger.info(f"Encoding {len(categorical_cols)} categorical columns")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Features: {df.shape[1] - 1}")

    return df, target_col


# ============================================================================
# TabPFN Evaluation
# ============================================================================

def evaluate_with_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluate using TabPFN.

    TabPFN is a transformer-based model pre-trained on synthetic data
    that can make predictions without training on the specific dataset.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.error("TabPFN not installed. Install with: pip install tabpfn")
        logger.info("Attempting alternative installation...")

        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn"])
            from tabpfn import TabPFNClassifier
        except Exception as e:
            logger.error(f"Failed to install TabPFN: {e}")
            return {'error': 'TabPFN not available'}

    logger.info("Initializing TabPFN classifier...")

    # Initialize TabPFN
    classifier = TabPFNClassifier(
        device=TABPFN_PARAMS['DEVICE'],
        N_ensemble_configurations=TABPFN_PARAMS['N_ENSEMBLE_CONFIGURATIONS'],
    )

    logger.info("Fitting TabPFN (this uses prior-fitted weights, minimal training)...")

    start_time = datetime.now()

    # TabPFN "fits" by storing the training data (it doesn't actually train)
    classifier.fit(X_train, y_train)

    fit_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Fit completed in {fit_time:.2f} seconds")

    # Predictions
    logger.info("Making predictions...")
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results = calculate_metrics(y_test, y_pred, y_prob)
    results['fit_time_sec'] = fit_time
    results['model'] = 'TabPFN'

    return results


def evaluate_with_sklearn_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Fallback evaluation using XGBoost with TabPFN-like hyperparameters
    when TabPFN is not available.
    """
    logger.warning("Using XGBoost fallback (TabPFN not available)")

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED
        )
        model_name = 'XGBoost (TabPFN fallback)'
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            random_state=RANDOM_SEED
        )
        model_name = 'GradientBoosting (TabPFN fallback)'

    logger.info(f"Training {model_name}...")

    start_time = datetime.now()
    model.fit(X_train, y_train)
    fit_time = (datetime.now() - start_time).total_seconds()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = calculate_metrics(y_test, y_pred, y_prob)
    results['fit_time_sec'] = fit_time
    results['model'] = model_name

    return results


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Calculate all evaluation metrics."""

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC (using probabilities)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'confusion_matrix': cm.tolist(),
        'predictions_positive': int(np.sum(y_pred == 1)),
        'predictions_negative': int(np.sum(y_pred == 0)),
        'actual_positive': int(np.sum(y_true == 1)),
        'actual_negative': int(np.sum(y_true == 0)),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TabPFN Baseline Evaluation for AgenticSimLaw"
    )
    parser.add_argument('--vignettes', type=str, default=DEFAULT_VIGNETTES_PATH,
                       help='Path to vignettes CSV file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for TabPFN')
    parser.add_argument('--fallback', action='store_true',
                       help='Force use of sklearn fallback instead of TabPFN')

    args = parser.parse_args()

    # Update device setting
    TABPFN_PARAMS['DEVICE'] = args.device

    # Setup paths
    script_dir = Path(__file__).parent
    vignettes_path = Path(args.vignettes)
    if not vignettes_path.is_absolute():
        vignettes_path = script_dir / vignettes_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("TabPFN Baseline Evaluation")
    logger.info("=" * 60)

    # Load and preprocess data
    df, target_col = load_and_preprocess(str(vignettes_path), logger)

    # Split data
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train class distribution: {np.bincount(y_train)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")

    # Run evaluation
    if args.fallback:
        results = evaluate_with_sklearn_fallback(
            X_train, y_train, X_test, y_test, logger
        )
    else:
        try:
            results = evaluate_with_tabpfn(
                X_train, y_train, X_test, y_test, logger
            )
        except Exception as e:
            logger.error(f"TabPFN evaluation failed: {e}")
            logger.info("Falling back to sklearn...")
            results = evaluate_with_sklearn_fallback(
                X_train, y_train, X_test, y_test, logger
            )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as JSON for detailed results
    import json
    results_json_path = output_dir / f"tabpfn_baseline_{timestamp}.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed results to: {results_json_path}")

    # Save as CSV for consistency with other scripts
    results_df = pd.DataFrame([{
        'model': results.get('model', 'TabPFN'),
        'accuracy': results['accuracy'],
        'auc': results['auc'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'true_positive': results['true_positive'],
        'true_negative': results['true_negative'],
        'false_positive': results['false_positive'],
        'false_negative': results['false_negative'],
        'fit_time_sec': results.get('fit_time_sec', np.nan),
    }])

    results_csv_path = output_dir / f"tabpfn_baseline_{timestamp}.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved CSV results to: {results_csv_path}")

    # Print summary (matching paper Table 3 format)
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY (Table 3 Format)")
    logger.info("=" * 60)

    logger.info(f"\nModel: {results.get('model', 'TabPFN')}")
    logger.info(f"AUC: {results['auc']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1 Score: {results['f1_score']:.4f}")

    cm = results['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  Predicted:    NO    YES")
    logger.info(f"  Actual NO:  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    logger.info(f"  Actual YES: [{cm[1][0]:4d}  {cm[1][1]:4d}]")

    logger.info(f"\nPrediction Distribution:")
    logger.info(f"  Predicted NO:  {results['predictions_negative']}")
    logger.info(f"  Predicted YES: {results['predictions_positive']}")

    # Compare to paper's expected results
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TO PAPER (Table 3)")
    logger.info("=" * 60)

    paper_results = {
        'auc': 0.6410,
        'accuracy': 0.7011,
        'recall': 0.0000,
        'f1_score': 0.0000,
    }

    for metric, paper_value in paper_results.items():
        actual_value = results[metric]
        diff = actual_value - paper_value
        logger.info(f"{metric:12s}: Paper={paper_value:.4f}, Actual={actual_value:.4f}, Diff={diff:+.4f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
