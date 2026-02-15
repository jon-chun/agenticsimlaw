import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from collections import defaultdict
import json
import traceback
import logging
from datetime import datetime


from compute_topn_features_utils import (
    preprocess_dataset_for_classification,
    compute_xgb_importance,
    compute_mutual_info,
    compute_permutation_importance,
    compute_lofo_importance,
    compute_shap_importance,
    extract_top_n_features,
    aggregate_feature_importance,
    validate_importance_mapping
)

INPUT_VIGNETTES_PATH = os.path.join('..', 'data', 'vignettes_final_clean.csv')
TARGET_COL = 'y_arrestedafter2002'
OUTPUT_VIGNETTES_PATH = os.path.join('..', 'data')

TOPN_COUNT_LS = [5, 10]

def validate_feature_tracking(
    feature_tracking: Dict,
    expanded_importance: Dict[str, float],
    aggregated_importance: Dict[str, float]
) -> List[str]:
    """
    Enhanced validation using feature tracking
    """
    warnings = []
    
    # Check all original features are mapped
    unmapped = set(feature_tracking['original_columns']) - set(feature_tracking['feature_mappings'].keys())
    if unmapped:
        warnings.append(f"Unmapped original features: {unmapped}")
    
    # Check all expanded features are used
    unused_expanded = set(expanded_importance.keys()) - \
                     set(feat for feats in feature_tracking['feature_mappings'].values() for feat in feats)
    if unused_expanded:
        warnings.append(f"Unused expanded features: {unused_expanded}")
    
    # Check importance conservation
    total_expanded = sum(expanded_importance.values())
    total_aggregated = sum(aggregated_importance.values())
    if not np.isclose(total_expanded, total_aggregated, rtol=1e-5):
        warnings.append(
            f"Total importance mismatch: expanded={total_expanded:.4f}, "
            f"aggregated={total_aggregated:.4f}"
        )
    
    return warnings



def create_and_print_summaries(
    expanded_results: Dict[str, pd.Series],
    aggregated_results: Dict[str, Dict[str, float]],
    output_dir: str,
    n_top_features: Optional[int] = None  # Made optional
) -> Dict:
    """
    Creates comprehensive summaries of feature importance results.
    Handles negative values and optionally limits output length.
    """
    from collections import defaultdict

    summary = {
        'expanded_features': {},
        'aggregated_features': {},
        'method_stats': {}
    }

    # Process expanded results
    print("\n=== Expanded Feature Importance Results ===")
    for method, importance in expanded_results.items():
        print(f"\nFeature importance ({method}):")
        # Don't limit to top N unless specifically requested
        features_to_show = importance if n_top_features is None else importance.head(n_top_features)
        print(features_to_show)

        # Reverse key-value pairs and sort by importance (descending)
        reversed_sorted_importance = defaultdict(list)
        for feat, score in sorted(importance.items(), key=lambda item: -item[1]):
            reversed_sorted_importance[float(score)].append(feat)

        # Convert defaultdict to a regular dict and sort keys (scores) in descending order
        summary['expanded_features'][method] = {
            score: reversed_sorted_importance[score]
            for score in sorted(reversed_sorted_importance.keys(), reverse=True)
        }

        # Calculate detailed stats including negative values
        summary['method_stats'][method] = {
            'total_absolute_importance': float(abs(importance).sum()),
            'total_positive_importance': float(importance[importance > 0].sum()),
            'total_negative_importance': float(importance[importance < 0].sum()),
            'mean_importance': float(importance.mean()),
            'std_importance': float(importance.std()),
            'num_features': len(importance),
            'num_positive_features': len(importance[importance > 0]),
            'num_negative_features': len(importance[importance < 0])
        }

    # Process aggregated results
    print(f"\n=== Aggregated Feature Importance Results ===")
    for method, importance_dict in aggregated_results.items():
        # Convert to series and sort by absolute value to show most influential features first
        importance = pd.Series(importance_dict)
        importance = importance.reindex(
            importance.abs().sort_values(ascending=False).index
        )

        print(f"\nFeature importance ({method}):")
        features_to_show = importance if n_top_features is None else importance.head(n_top_features)
        print(features_to_show)

        # Reverse key-value pairs and sort by importance (descending)
        reversed_sorted_importance = defaultdict(list)
        for feat, score in sorted(importance.items(), key=lambda item: -item[1]):
            reversed_sorted_importance[float(score)].append(feat)

        # Convert defaultdict to a regular dict and sort keys (scores) in descending order
        summary['aggregated_features'][method] = {
            score: reversed_sorted_importance[score]
            for score in sorted(reversed_sorted_importance.keys(), reverse=True)
        }

        # Add detailed stats
        base_method = method.replace('_aggregated', '')
        if base_method in summary['method_stats']:
            summary['method_stats'][base_method].update({
                'total_absolute_aggregated': float(abs(importance).sum()),
                'total_positive_aggregated': float(importance[importance > 0].sum()),
                'total_negative_aggregated': float(importance[importance < 0].sum()),
                'aggregation_ratio': float(abs(importance).sum()) /
                                   summary['method_stats'][base_method]['total_absolute_importance']
            })

    # Save complete summary to JSON
    summary_path = os.path.join(output_dir, 'feature_importance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


# Add at the top after imports
TOPN_COUNT_LS = [5, 10]

TOPN_COUNT_LS = [5, 10]

def generate_topn_dataset(
    df: pd.DataFrame,
    n: int,
    importance_series: pd.Series,
    target_column: str,
    feature_tracking: Dict = None
) -> pd.DataFrame:
    """
    Generate dataset with top N most important features plus target column,
    maintaining granularity of preprocessed features while using original data.
    
    Args:
        df: Original DataFrame with all features
        n: Number of top features to select
        importance_series: Series with feature names as index and importance scores as values
        target_column: Name of target variable column
        feature_tracking: Dictionary containing feature mapping information
        
    Returns:
        DataFrame with top N features plus target column
    """
    logger = logging.getLogger('feature_importance')
    
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be DataFrame, got {type(df)}")
        if target_column not in df.columns:
            raise ValueError(f"target_column {target_column} not in DataFrame")
            
        # Convert to pandas Series if not already
        if not isinstance(importance_series, pd.Series):
            importance_series = pd.Series(importance_series)
        
        if feature_tracking is None:
            logger.warning("No feature tracking provided, attempting direct column selection")
            # Get top N features in order of importance
            top_features = importance_series.sort_values(ascending=False).head(n).index.tolist()
            selected_cols = top_features + [target_column]
            result_df = df[selected_cols].copy()
            
        else:
            logger.info("Using feature tracking to map preprocessed features to original columns")
            
            # Get top N preprocessed features while preserving order
            top_preprocessed_features = importance_series.sort_values(ascending=False).head(n)
            
            # Create mapping from preprocessed features to original columns
            feature_to_original = {}
            for orig_col, preprocessed_features in feature_tracking['feature_mappings'].items():
                for prep_feat in preprocessed_features:
                    feature_to_original[prep_feat] = orig_col
            
            # Track which original columns we need and their preprocessed features
            needed_columns = {}  # original_col -> set of conditions
            for prep_feat in top_preprocessed_features.index:
                orig_col = feature_to_original.get(prep_feat)
                if orig_col:
                    if orig_col not in needed_columns:
                        needed_columns[orig_col] = set()
                    
                    # Extract the condition from the preprocessed feature name
                    # e.g., "college02_enrolled" -> "enrolled"
                    if '_' in prep_feat:
                        condition = prep_feat.split('_', 1)[1]
                        needed_columns[orig_col].add(condition)
            
            logger.debug(f"Needed columns and conditions: {needed_columns}")
            
            # Create result DataFrame starting with just the target column
            result_df = pd.DataFrame(index=df.index)
            result_df[target_column] = df[target_column].copy()
            
            # Add each needed column
            for orig_col, conditions in needed_columns.items():
                if len(conditions) <= 1 or orig_col in df.columns:
                    # If it's a numeric column or we only need one category,
                    # just copy the original column
                    result_df[orig_col] = df[orig_col].copy()
                else:
                    # For categorical columns where we need specific categories,
                    # create binary columns for each needed category
                    column_data = df[orig_col]
                    for condition in conditions:
                        col_name = f"{orig_col}_{condition}"
                        # Create binary column based on the condition
                        result_df[col_name] = (column_data == condition).astype(int)
            
            logger.info(f"Final columns: {result_df.columns.tolist()}")
            
        logger.info(f"Generated dataset shape: {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in generate_topn_dataset: {str(e)}")
        raise

def save_topn_features(
    df: pd.DataFrame,
    method_name: str,
    importance_series: pd.Series,
    output_dir: str,
    target_column: str,
    feature_tracking: Dict = None
) -> None:
    """
    Save datasets containing top N features for a given method.
    
    Args:
        df: Input DataFrame
        method_name: Name of feature importance method
        importance_series: Series with feature importance scores
        output_dir: Directory to save output files
        target_column: Name of target variable column
        feature_tracking: Dictionary containing feature mapping information
    """
    logger = logging.getLogger('feature_importance')
    logger.info(f"\nSaving top-N datasets for {method_name}...")
    
    # Ensure importance_series has features as index
    if importance_series.index.empty:
        logger.warning(f"Empty importance series for {method_name}")
        return
        
    logger.debug(f"Feature importance scores for {method_name}:")
    logger.debug(importance_series.sort_values(ascending=False).head())
    
    for n in TOPN_COUNT_LS:
        try:
            # Validate n is not larger than available features
            if n > len(importance_series):
                logger.warning(f"Requested top {n} but only {len(importance_series)} features available")
                continue
                
            # Generate dataset with top N features
            logger.info(f"Generating top {n} dataset...")
            topn_df = generate_topn_dataset(
                df=df,
                n=n,
                importance_series=importance_series,
                target_column=target_column,
                feature_tracking=feature_tracking  # Pass feature tracking information
            )
            
            # Validate output DataFrame
            if len(topn_df.columns) != n + 1:  # +1 for target column
                logger.warning(
                    f"Warning: Expected {n+1} columns but got {len(topn_df.columns)}. "
                    "This might be due to feature aggregation."
                )
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"vignettes_{method_name}_top{n}.csv")
            topn_df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving top {n} features for {method_name}: {str(e)}")
            logger.debug(traceback.format_exc())

# Set up logging
def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging to both file and console
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('feature_importance')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'feature_importance_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def analyze_feature_importance(
    df: pd.DataFrame,
    target_column: str = TARGET_COL,
    n_top_features: int = None,
    output_dir: str = OUTPUT_VIGNETTES_PATH,
    random_state: int = 42,
    aggregation_method: str = 'sum'
) -> Dict[str, Union[pd.Series, Dict[str, float]]]:
    """
    Enhanced framework to analyze feature importance using multiple methods.
    """
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info("Starting feature importance analysis...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionaries
    expanded_results = {}
    aggregated_results = {}
    method_warnings = defaultdict(list)
    
    try:
        # 1. Preprocess dataset with feature tracking
        logger.info("Preprocessing dataset...")
        preproc_results = preprocess_dataset_for_classification(
            df=df,
            target_column=target_column,
            test_size=0.2,
            random_state=random_state
        )
        
        X_train = preproc_results["X_train"]
        y_train = preproc_results["y_train"]
        feature_tracking = preproc_results["feature_tracking"]
        
        logger.info(f"Preprocessing complete. Shape: {X_train.shape}")
        
        # 2. Compute importances for standard methods
        standard_methods = {
            'xgboost': (compute_xgb_importance, {}),
            'mi': (compute_mutual_info, {}),
            'permutation': (compute_permutation_importance, 
                        {'scoring': 'accuracy', 'n_repeats': 10}),
            'shap': (compute_shap_importance, {})
        }
                
        # Process standard methods
        for method_name, (method_func, method_params) in standard_methods.items():
            try:
                logger.info(f"\nComputing {method_name.upper()} importance...")
                
                importance = method_func(X_train, y_train, 
                                      random_state=random_state, 
                                      **method_params)
                
                if not importance.empty:
                    logger.info(f"{method_name} importance calculation successful")
                    
                    # Store expanded results
                    expanded_results[method_name] = importance
                    expanded_importance_dict = importance.to_dict()
                    
                    # Compute aggregated importance
                    aggregated_importance = aggregate_feature_importance(
                        original_df=df.drop(columns=[target_column]),
                        expanded_importance=expanded_importance_dict,
                        feature_tracking=feature_tracking,
                        aggregation_method=aggregation_method
                    )
                    aggregated_results[f"{method_name}_aggregated"] = aggregated_importance
                    
                    # Save top-N datasets for this method
                    try:
                        logger.info(f"Saving top-N datasets for {method_name}...")

                        save_topn_features(
                            df=df,
                            method_name=method_name,
                            importance_series=importance,
                            output_dir=output_dir,
                            target_column=target_column,
                            feature_tracking=feature_tracking  # Add this parameter
                        )

                        logger.info(f"Successfully saved top-N datasets for {method_name}")
                    except Exception as e:
                        logger.error(f"Error saving top-N datasets for {method_name}: {str(e)}")
                        logger.debug(traceback.format_exc())
                    
                    # Save aggregated results
                    try:
                        aggregated_df = pd.DataFrame.from_dict(
                            aggregated_importance,
                            orient='index',
                            columns=['importance']
                        ).sort_values('importance', ascending=False)
                        
                        output_path = os.path.join(output_dir, f"vignettes_{method_name}_aggregated.csv")
                        aggregated_df.to_csv(output_path)
                        logger.info(f"Saved aggregated results to {output_path}")
                    except Exception as e:
                        logger.error(f"Error saving aggregated results for {method_name}: {str(e)}")
                        
                else:
                    logger.warning(f"Empty results for {method_name}, skipping")
                    
            except Exception as e:
                logger.error(f"Error computing {method_name} importance: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # 3. Compute LOFO importance
        logger.info("\nComputing LOFO importance...")
        try:
            lofo_importance = compute_lofo_importance(
                X_train=X_train,
                y_train=y_train,
                scoring="neg_log_loss",
                cv=5
            )
            
            if not lofo_importance.empty:
                logger.info("LOFO importance calculation successful")
                
                # Store expanded results
                expanded_results['lofo'] = lofo_importance
                expanded_importance_dict = lofo_importance.to_dict()
                
                # Save top-N datasets for LOFO
                try:
                    logger.info("Saving top-N datasets for LOFO...")
                    save_topn_features(
                        df=df,
                        method_name='lofo',
                        importance_series=lofo_importance,
                        output_dir=output_dir,
                        target_column=target_column,
                        feature_tracking=feature_tracking  # Add this parameter
                    )
                    logger.info("Successfully saved top-N datasets for LOFO")
                except Exception as e:
                    logger.error(f"Error saving top-N datasets for LOFO: {str(e)}")
                    logger.debug(traceback.format_exc())
                    
            else:
                logger.warning("Empty LOFO results, skipping")
                
        except Exception as e:
            logger.error(f"Error computing LOFO importance: {str(e)}")
            logger.debug(traceback.format_exc())
        
        logger.info("Feature importance analysis complete")
        return {
            'expanded_results': expanded_results,
            'aggregated_results': aggregated_results,
            'warnings': dict(method_warnings),
            'feature_tracking': feature_tracking
        }
        
    except Exception as e:
        logger.error(f"Critical error in feature importance analysis: {str(e)}")
        logger.debug(traceback.format_exc())
        raise



# Example usage
if __name__ == "__main__":
    # Read input data
    with open(INPUT_VIGNETTES_PATH, 'r') as f:
        df = pd.read_csv(f)

    # Run analysis with both expanded and aggregated results
    results = analyze_feature_importance(
        df=df,
        target_column=TARGET_COL,
        n_top_features=None,
        output_dir=OUTPUT_VIGNETTES_PATH,
        random_state=42,
        aggregation_method='sum'
    )

    # Print results
    print("\nExpanded Feature Importance Results:")
    for method in ['xgboost', 'mi', 'permutation', 'lofo']:
        if method in results:
            print(f"\nTop 10 features ({method}):")
            print(results[method].head(10))
            
    print("\nAggregated Feature Importance Results:")
    for method in ['xgboost_aggregated', 'mi_aggregated', 'permutation_aggregated', 'lofo_aggregated']:
        if method in results:
            print(f"\nTop 10 features ({method}):")
            importance_series = pd.Series(results[method]).sort_values(ascending=False)
            print(importance_series.head(10))