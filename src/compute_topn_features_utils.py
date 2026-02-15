# generate_topn_datasets_utils.py

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# XGBoost
from xgboost import XGBClassifier

def create_feature_mapping(
    original_df: pd.DataFrame,
    expanded_df: pd.DataFrame,
    encoder_mapping: Optional[Dict] = None
) -> Dict[str, List[str]]:
    """
    Creates a mapping between original features and their expanded versions.
    
    Args:
        original_df: DataFrame with original features
        expanded_df: DataFrame with expanded features (after encoding)
        encoder_mapping: Optional mapping from categorical encoder
        
    Returns:
        Dictionary mapping original features to list of expanded features
    """
    feature_mapping = defaultdict(list)
    
    # Handle numeric columns (direct mapping)
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in expanded_df.columns:
            feature_mapping[col].append(col)
            
    # Handle categorical columns
    categorical_cols = original_df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    for col in categorical_cols:
        # Case 1: Direct mapping if column exists
        if col in expanded_df.columns:
            feature_mapping[col].append(col)
            continue
            
        # Case 2: One-hot encoded columns
        encoded_prefixes = [f"{col}_", f"{col}=", f"{col}."]
        for expanded_col in expanded_df.columns:
            if any(expanded_col.startswith(prefix) for prefix in encoded_prefixes):
                feature_mapping[col].append(expanded_col)
                
        # Case 3: Use encoder mapping if provided
        if encoder_mapping and col in encoder_mapping:
            mapped_features = [
                feat for feat in expanded_df.columns 
                if feat in encoder_mapping[col]
            ]
            feature_mapping[col].extend(mapped_features)
    
    return dict(feature_mapping)

def normalize_importance_scores(importance_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes importance scores to sum to 1.0
    """
    total = sum(importance_scores.values())
    if total == 0:
        return importance_scores
    return {k: v/total for k, v in importance_scores.items()}

def aggregate_feature_importance(
    original_df: pd.DataFrame,
    expanded_importance: Dict[str, float],
    feature_tracking: Dict,
    aggregation_method: str = 'sum'
) -> Dict[str, float]:
    """
    Enhanced aggregation using feature tracking information
    """
    aggregated_scores = defaultdict(list)
    
    # Use feature tracking to map expanded features back to original
    for orig_feature, expanded_features in feature_tracking['feature_mappings'].items():
        for exp_feature in expanded_features:
            if exp_feature in expanded_importance:
                aggregated_scores[orig_feature].append(expanded_importance[exp_feature])
    
    # Apply aggregation
    final_scores = {}
    for feature, scores in aggregated_scores.items():
        if not scores:
            final_scores[feature] = 0.0
            continue
            
        if aggregation_method == 'sum':
            final_scores[feature] = sum(scores)
        elif aggregation_method == 'mean':
            final_scores[feature] = np.mean(scores)
        elif aggregation_method == 'max':
            final_scores[feature] = max(scores)
    
    # Ensure all original features are included
    for feature in feature_tracking['original_columns']:
        if feature not in final_scores:
            final_scores[feature] = 0.0
    
    # Normalize
    total = sum(final_scores.values())
    if total > 0:
        final_scores = {k: v/total for k, v in final_scores.items()}
    
    return final_scores

def validate_importance_mapping(
    original_df: pd.DataFrame,
    expanded_importance: Dict[str, float],
    aggregated_importance: Dict[str, float]
) -> List[str]:
    """
    Validates the feature importance mapping and returns warnings.
    """
    warnings = []
    
    # Check for unmapped original features
    unmapped = [col for col in original_df.columns if aggregated_importance.get(col, 0) == 0]
    if unmapped:
        warnings.append(f"Features with zero importance: {unmapped}")
    
    # Check for significant importance changes
    total_expanded = sum(expanded_importance.values())
    total_aggregated = sum(aggregated_importance.values())
    if not np.isclose(total_expanded, total_aggregated, rtol=1e-5):
        warnings.append(
            f"Total importance mismatch: expanded={total_expanded:.4f}, "
            f"aggregated={total_aggregated:.4f}"
        )
    
    return warnings

def get_compatible_encoder(**kwargs):
    """
    Creates a OneHotEncoder with version-compatible parameters.
    """
    from sklearn import __version__ as sklearn_version
    from packaging import version
    from sklearn.preprocessing import OneHotEncoder
    
    if version.parse(sklearn_version) >= version.parse('1.2.0'):
        # For scikit-learn >= 1.2.0
        return OneHotEncoder(
            drop='first',
            sparse_output=False,  # new parameter name
            handle_unknown='ignore'
        )
    else:
        # For older scikit-learn versions
        return OneHotEncoder(
            drop='first',
            sparse=False,  # old parameter name
            handle_unknown='ignore'
        )


def preprocess_dataset_for_classification(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Union[pd.DataFrame, pd.Series, Dict]]:
    """
    Enhanced preprocessing with robust feature tracking
    """
    try:
        # Store original column information
        feature_tracking = {
            'original_columns': df.drop(columns=[target_column]).columns.tolist(),
            'column_types': {},
            'feature_mappings': defaultdict(list)
        }
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Track column types
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_tracking['column_types'][col] = 'numeric'
            else:
                feature_tracking['column_types'][col] = 'categorical'
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=random_state
        )
        
        # Process features with tracking
        X_train_processed = pd.DataFrame(index=X_train.index)
        X_test_processed = pd.DataFrame(index=X_test.index)
        
        # Handle numeric features
        numeric_cols = [col for col, type_ in feature_tracking['column_types'].items() 
                       if type_ == 'numeric']
        if numeric_cols:
            scaler = StandardScaler()
            X_train_num = pd.DataFrame(
                scaler.fit_transform(X_train[numeric_cols]),
                columns=numeric_cols,
                index=X_train.index
            )
            X_test_num = pd.DataFrame(
                scaler.transform(X_test[numeric_cols]),
                columns=numeric_cols,
                index=X_test.index
            )
            
            # Track numeric mappings (1:1)
            for col in numeric_cols:
                feature_tracking['feature_mappings'][col].append(col)
            
            X_train_processed = pd.concat([X_train_processed, X_train_num], axis=1)
            X_test_processed = pd.concat([X_test_processed, X_test_num], axis=1)
        
        # Handle categorical features
        categorical_cols = [col for col, type_ in feature_tracking['column_types'].items() 
                          if type_ == 'categorical']
        if categorical_cols:
            # Process each categorical column separately
            for col in categorical_cols:
                try:
                    # Create encoder for this column
                    encoder = get_compatible_encoder()
                    col_data = X_train[[col]]
                    
                    # Fit and transform
                    encoded_train = encoder.fit_transform(col_data)
                    encoded_test = encoder.transform(X_test[[col]])
                    
                    # Get feature names
                    feature_names = encoder.get_feature_names_out([col])
                    
                    # Create DataFrames
                    encoded_train_df = pd.DataFrame(
                        encoded_train,
                        columns=feature_names,
                        index=X_train.index
                    )
                    encoded_test_df = pd.DataFrame(
                        encoded_test,
                        columns=feature_names,
                        index=X_test.index
                    )
                    
                    # Track mappings
                    feature_tracking['feature_mappings'][col].extend(feature_names)
                    
                    # Add to processed data
                    X_train_processed = pd.concat([X_train_processed, encoded_train_df], axis=1)
                    X_test_processed = pd.concat([X_test_processed, encoded_test_df], axis=1)
                    
                except Exception as e:
                    print(f"Warning: Error encoding {col}: {str(e)}")
                    # Fall back to label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_train_processed[col] = le.fit_transform(X_train[col])
                    X_test_processed[col] = le.transform(X_test[col])
                    feature_tracking['feature_mappings'][col].append(col)
        
        return {
            "X_train": X_train_processed,
            "X_test": X_test_processed,
            "y_train": y_train,
            "y_test": y_test,
            "feature_tracking": feature_tracking
        }
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise



def compute_xgb_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> pd.Series:
    """
    Computes feature importance using XGBoost.
    
    Returns:
        Series of feature importances sorted in descending order
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    importance_series = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    return importance_series

def compute_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> pd.Series:
    """
    Computes mutual information between features and target.
    
    Returns:
        Series of MI scores sorted in descending order
    """
    mi_scores = mutual_info_classif(
        X_train, y_train,
        random_state=random_state
    )
    importance_series = pd.Series(
        mi_scores,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    return importance_series

def compute_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.Series:
    """
    Computes permutation importance using LogisticRegression.
    
    Returns:
        Series of permutation importance scores sorted in descending order
    """
    model = LogisticRegression(
        max_iter=500,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    result = permutation_importance(
        model, X_train, y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    importance_series = pd.Series(
        result.importances_mean,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    return importance_series

def compute_lofo_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "neg_log_loss",
    cv: int = 4,
    random_state: int = 42
) -> pd.Series:
    """
    Custom implementation of Leave-One-Feature-Out importance calculation.
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        print("\nDEBUG: Using custom LOFO implementation")
        print("-" * 50)
        
        # Initialize base model
        base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
        
        # Get baseline score with all features
        print("Computing baseline score...")
        baseline_score = np.mean(cross_val_score(
            base_model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        ))
        print(f"Baseline score: {baseline_score:.4f}")
        
        # Calculate importance for each feature
        importance_scores = {}
        
        for feature in X_train.columns:
            print(f"Processing feature: {feature}")
            
            # Create dataset without this feature
            X_without_feature = X_train.drop(columns=[feature])
            
            # Calculate score without this feature
            score_without_feature = np.mean(cross_val_score(
                base_model,
                X_without_feature,
                y_train,
                cv=cv,
                scoring=scoring
            ))
            
            # Importance is decrease in score when feature is removed
            importance = baseline_score - score_without_feature
            importance_scores[feature] = importance
            print(f"Importance for {feature}: {importance:.4f}")
        
        # Convert to series and sort
        importance_series = pd.Series(importance_scores).sort_values(ascending=False)
        
        return importance_series
            
    except Exception as e:
        print(f"\nError in custom LOFO calculation:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print("\nFull stack trace:")
        import traceback
        traceback.print_exc()
        return pd.Series()
    
def compute_shap_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> pd.Series:
    """
    Computes SHAP feature importance scores.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Series of SHAP importance scores
    """
    import shap
    
    # Train a model (using XGBoost as base model)
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # For binary classification, shap_values is a list with one element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_train.columns
    )
    
    # Sort by absolute importance
    return feature_importance.sort_values(ascending=False)


def extract_top_n_features(
    df: pd.DataFrame,
    importance_series: pd.Series,
    n_top_features: int,
    target_col: Optional[str] = None,
    output_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Extracts top N features based on importance scores.
    
    Args:
        df: Original DataFrame
        importance_series: Series of feature importance scores
        n_top_features: Number of top features to extract
        target_col: Target column to include in output (optional)
        output_filename: Path to save output CSV (optional)
        
    Returns:
        DataFrame containing only top N features (and target if specified)
    """
    top_features = importance_series.index[:n_top_features]
    
    # Filter features that exist in original df
    top_features_in_df = [col for col in top_features if col in df.columns]
    
    # Include target column if specified
    if target_col and target_col in df.columns:
        columns_to_keep = list(top_features_in_df) + [target_col]
    else:
        columns_to_keep = list(top_features_in_df)
    
    # Create subset DataFrame
    subset_df = df[columns_to_keep].copy()
    
    # Save to CSV if filename provided
    if output_filename:
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        subset_df.to_csv(output_filename, index=False)
    
    return subset_df