#!/usr/bin/env python3
"""
PyCaret ML Baseline Evaluation Script

Evaluates traditional machine learning models on the recidivism prediction task
using PyCaret's automated ML pipeline.

Paper Reference: Section 3.6 "Traditional Baseline Models" and Table 2
- 14 traditional ML models evaluated
- Cross-validation approach
- Metrics: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC, Training Time

Models evaluated (from Table 2):
- Extra Trees Classifier (et)
- CatBoost Classifier (catboost)
- Random Forest Classifier (rf)
- Gradient Boosting Classifier (gbc)
- Ridge Classifier (ridge)
- Linear Discriminant Analysis (lda)
- Logistic Regression (lr)
- Extreme Gradient Boosting (xgboost)
- AdaBoost Classifier (ada)
- Naive Bayes (nb)
- Decision Tree Classifier (dt)
- SVM - Linear Kernel (svm)
- Quadratic Discriminant Analysis (qda)
- K Neighbors Classifier (knn)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VIGNETTES_PATH = "../data/sample_vignettes.csv"
DEFAULT_OUTPUT_DIR = "../results/baselines"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # From remaining after test split

# Models to evaluate (paper Table 2)
MODELS_TO_EVALUATE = [
    'et',       # Extra Trees Classifier
    'rf',       # Random Forest Classifier
    'gbc',      # Gradient Boosting Classifier
    'ada',      # AdaBoost Classifier
    'dt',       # Decision Tree Classifier
    'knn',      # K Neighbors Classifier
    'lr',       # Logistic Regression
    'ridge',    # Ridge Classifier
    'lda',      # Linear Discriminant Analysis
    'qda',      # Quadratic Discriminant Analysis
    'nb',       # Naive Bayes
    'svm',      # SVM - Linear Kernel
    # Note: CatBoost and XGBoost require separate installation
]

# Full model names for reporting
MODEL_NAMES = {
    'et': 'Extra Trees Classifier',
    'catboost': 'CatBoost Classifier',
    'rf': 'Random Forest Classifier',
    'gbc': 'Gradient Boosting Classifier',
    'ridge': 'Ridge Classifier',
    'lda': 'Linear Discriminant Analysis',
    'lr': 'Logistic Regression',
    'xgboost': 'Extreme Gradient Boosting',
    'ada': 'AdaBoost Classifier',
    'nb': 'Naive Bayes',
    'dt': 'Decision Tree Classifier',
    'svm': 'SVM - Linear Kernel',
    'qda': 'Quadratic Discriminant Analysis',
    'knn': 'K Neighbors Classifier',
}


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging."""
    log_file = output_dir / f"pycaret_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

def load_and_preprocess(path: str, logger: logging.Logger,
                        target_col: str = 'y_arrestedafter2002') -> Tuple[pd.DataFrame, str]:
    """
    Load and preprocess the vignettes dataset.

    Returns:
        Tuple of (preprocessed DataFrame, target column name)
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Convert target to int (0/1)
    df[target_col] = df[target_col].astype(int)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    logger.info(f"Categorical columns to encode: {categorical_cols}")

    # Label encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    return df, target_col


# ============================================================================
# Scikit-learn Based Evaluation (Fallback if PyCaret not available)
# ============================================================================

def sklearn_evaluation(
    df: pd.DataFrame,
    target_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Evaluate ML models using scikit-learn directly.
    This is a fallback when PyCaret is not available.
    """
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, AdaBoostClassifier
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    # Split data
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define models
    models = {
        'et': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'rf': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'gbc': GradientBoostingClassifier(random_state=RANDOM_SEED),
        'ada': AdaBoostClassifier(random_state=RANDOM_SEED),
        'dt': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'ridge': RidgeClassifier(random_state=RANDOM_SEED),
        'lda': LinearDiscriminantAnalysis(),
        'qda': QuadraticDiscriminantAnalysis(),
        'nb': GaussianNB(),
        'svm': SVC(kernel='linear', probability=True, random_state=RANDOM_SEED),
    }

    # Try to add XGBoost and CatBoost if available
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    try:
        from catboost import CatBoostClassifier
        models['catboost'] = CatBoostClassifier(
            verbose=False,
            random_state=RANDOM_SEED
        )
    except ImportError:
        logger.warning("CatBoost not installed, skipping")

    results = []

    for model_id, model in models.items():
        logger.info(f"Training {MODEL_NAMES.get(model_id, model_id)}...")

        try:
            start_time = datetime.now()
            model.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()

            # Predictions
            y_pred = model.predict(X_test)

            # Probabilities for AUC (if available)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except (AttributeError, IndexError):
                auc = np.nan

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            results.append({
                'model_id': model_id,
                'model_name': MODEL_NAMES.get(model_id, model_id),
                'accuracy': acc,
                'auc': auc,
                'recall': rec,
                'precision': prec,
                'f1_score': f1,
                'kappa': kappa,
                'mcc': mcc,
                'train_time_sec': train_time,
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
            })

            logger.info(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        except Exception as e:
            logger.error(f"Error training {model_id}: {e}")
            results.append({
                'model_id': model_id,
                'model_name': MODEL_NAMES.get(model_id, model_id),
                'accuracy': np.nan,
                'auc': np.nan,
                'recall': np.nan,
                'precision': np.nan,
                'f1_score': np.nan,
                'kappa': np.nan,
                'mcc': np.nan,
                'train_time_sec': np.nan,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive': 0,
                'false_negative': 0,
                'error': str(e),
            })

    return pd.DataFrame(results)


def pycaret_evaluation(
    df: pd.DataFrame,
    target_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Evaluate ML models using PyCaret's automated pipeline.
    """
    try:
        from pycaret.classification import setup, compare_models, pull
    except ImportError:
        logger.warning("PyCaret not installed. Falling back to scikit-learn evaluation.")
        return sklearn_evaluation(df, target_col, logger)

    logger.info("Setting up PyCaret classification experiment...")

    # Setup PyCaret
    setup(
        data=df,
        target=target_col,
        session_id=RANDOM_SEED,
        train_size=1 - TEST_SIZE,
        fold=5,
        verbose=False,
        html=False,
    )

    logger.info("Comparing models...")

    # Compare all models
    best_models = compare_models(
        include=MODELS_TO_EVALUATE,
        n_select=len(MODELS_TO_EVALUATE),
        sort='F1',
    )

    # Pull results
    results_df = pull()

    # Rename columns to match our format
    results_df = results_df.rename(columns={
        'Model': 'model_name',
        'Accuracy': 'accuracy',
        'AUC': 'auc',
        'Recall': 'recall',
        'Prec.': 'precision',
        'F1': 'f1_score',
        'Kappa': 'kappa',
        'MCC': 'mcc',
        'TT (Sec)': 'train_time_sec',
    })

    return results_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyCaret ML Baseline Evaluation for AgenticSimLaw"
    )
    parser.add_argument('--vignettes', type=str, default=DEFAULT_VIGNETTES_PATH,
                       help='Path to vignettes CSV file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--use-sklearn', action='store_true',
                       help='Force use of scikit-learn instead of PyCaret')
    parser.add_argument('--target', type=str, default='y_arrestedafter2002',
                       help='Target column name (default: y_arrestedafter2002)')

    args = parser.parse_args()

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
    logger.info("PyCaret ML Baseline Evaluation")
    logger.info("=" * 60)

    # Load and preprocess data
    df, target_col = load_and_preprocess(str(vignettes_path), logger,
                                          target_col=args.target)

    # Run evaluation
    if args.use_sklearn:
        logger.info("Using scikit-learn evaluation (--use-sklearn flag)")
        results_df = sklearn_evaluation(df, target_col, logger)
    else:
        results_df = pycaret_evaluation(df, target_col, logger)

    # Sort by F1 score (descending)
    results_df = results_df.sort_values('f1_score', ascending=False)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"pycaret_baseline_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to: {results_path}")

    # Print summary table (matching paper Table 2 format)
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY (Table 2 Format)")
    logger.info("=" * 60)

    summary_cols = ['model_id', 'model_name', 'accuracy', 'auc', 'recall',
                    'precision', 'f1_score', 'kappa', 'mcc', 'train_time_sec']

    # Filter to available columns
    available_cols = [c for c in summary_cols if c in results_df.columns]

    print("\n" + results_df[available_cols].to_string(index=False))

    # Print best performers
    logger.info("\nTop 3 by Accuracy:")
    top_acc = results_df.nlargest(3, 'accuracy')[['model_name', 'accuracy']]
    for _, row in top_acc.iterrows():
        logger.info(f"  {row['model_name']}: {row['accuracy']:.4f}")

    logger.info("\nTop 3 by F1 Score:")
    top_f1 = results_df.nlargest(3, 'f1_score')[['model_name', 'f1_score']]
    for _, row in top_f1.iterrows():
        logger.info(f"  {row['model_name']}: {row['f1_score']:.4f}")

    if 'auc' in results_df.columns:
        logger.info("\nTop 3 by AUC:")
        top_auc = results_df.nlargest(3, 'auc')[['model_name', 'auc']]
        for _, row in top_auc.iterrows():
            logger.info(f"  {row['model_name']}: {row['auc']:.4f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
