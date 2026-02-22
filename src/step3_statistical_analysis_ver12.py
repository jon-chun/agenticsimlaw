#!/usr/bin/env python3
"""
step3_statistical_analysis_ver12.py

Dataset-aware version of step3_statistical_analysis_ver11_FREEZE.py.
Accepts input CSV path via argparse instead of hardcoded constants.
Uses generic 'actual_recidivism' column with backward compatibility
fallback to 'y_arrestedafter2002'.

Key features:
- Processes prediction correctness against actual outcomes
- Calculates per-model statistics including API success rates
- Analyzes speaker metrics across multiple dialogue turns
- Handles missing or malformed data gracefully
- Generates detailed CSV, TXT, and JSON outputs
- Accepts --input-csv argument for dataset-agnostic operation
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Generic column name for the actual outcome
ACTUAL_COL = 'actual_recidivism'
# Legacy column name for backward compatibility
LEGACY_ACTUAL_COL = 'y_arrestedafter2002'


def process_prediction(row):
    """
    Process prediction correctness based on actual_recidivism and prediction values.
    Returns tuple of (is_correct, is_valid_prediction).

    Only YES/NO predictions are valid. "No decision", "NO DECISION", "UNKNOWN",
    null, and any other non-YES/NO value are treated as invalid (excluded from
    accuracy and confusion matrix calculations).
    """
    print(f"\n[DEBUG] Processing prediction for row {row.get('row_no', 'unknown')}")
    print(f"[DEBUG] Raw prediction: {row['prediction']}")
    print(f"[DEBUG] Raw actual_recidivism: {row[ACTUAL_COL]}")

    # First check if prediction is valid
    if pd.isna(row['prediction']):
        print("[DEBUG] Invalid prediction - null")
        return False, False

    pred_upper = row['prediction'].strip().upper()

    # Only YES and NO are valid predictions; everything else is indeterminate
    if pred_upper == 'YES':
        pred_bool = True
    elif pred_upper == 'NO':
        pred_bool = False
    else:
        print(f"[DEBUG] Invalid prediction - not YES/NO: '{row['prediction']}'")
        return False, False

    actual_bool = row[ACTUAL_COL]

    print(f"[DEBUG] Processed prediction_bool: {pred_bool}")
    print(f"[DEBUG] Processed actual_bool: {actual_bool}")
    print(f"[DEBUG] Prediction correct: {pred_bool == actual_bool}")

    return pred_bool == actual_bool, True


def get_available_speaker_columns(df):
    """
    Analyze DataFrame to determine which speaker-related columns are present.
    """
    print("\n[INFO] Analyzing available speaker columns...")

    speaker_columns = {
        'prompt_eval_ct': [],
        'eval_ct': [],
        'total_duration_sec': []
    }

    for spk_num in range(6):
        for metric in speaker_columns.keys():
            possible_names = [
                f'speaker_{spk_num}_{metric}',
                f'speaker{spk_num}_{metric}',
                f'spk_{spk_num}_{metric}',
                f'{metric}_{spk_num}'
            ]

            found_col = next((col for col in possible_names if col in df.columns), None)

            if found_col:
                speaker_columns[metric].append((spk_num, found_col))
                print(f"[DEBUG] Found {metric} for speaker {spk_num}: {found_col}")

    return speaker_columns


def safe_calculate_median(series):
    """
    Safely calculate median of non-zero values, handling empty or invalid series.
    """
    if series is None or len(series) == 0:
        return np.nan

    numeric_series = pd.to_numeric(series, errors='coerce')
    non_zero = numeric_series.replace(0, np.nan)

    if non_zero.isna().all():
        return np.nan

    return non_zero.median()


def calculate_speaker_metrics(df, model_name):
    """
    Calculate available speaker metrics, gracefully handling missing columns.
    """
    print(f"\n[INFO] Calculating speaker metrics for model: {model_name}")

    available_columns = get_available_speaker_columns(df)
    speaker_metrics = {}
    token_medians = []
    duration_medians = []

    # Get all unique speaker numbers that have any metrics
    all_speakers = set()
    for metric, columns in available_columns.items():
        all_speakers.update(spk_num for spk_num, _ in columns)

    for spk_num in sorted(all_speakers):
        print(f"\n[DEBUG] Processing speaker {spk_num}")
        metrics = {}

        # Process each available metric for this speaker
        for metric_name, col_list in available_columns.items():
            col = next((col for num, col in col_list if num == spk_num), None)
            if col:
                val = safe_calculate_median(df[col])
                metrics[metric_name] = val
                print(f"[DEBUG] {metric_name} median: {val}")
            else:
                metrics[metric_name] = np.nan
                print(f"[DEBUG] No {metric_name} data available")

        # Store available metrics for this speaker
        if not np.isnan(metrics.get('prompt_eval_ct', np.nan)):
            speaker_metrics[f'speaker{spk_num}_prompt_eval_ct_median'] = metrics['prompt_eval_ct']

        if not np.isnan(metrics.get('eval_ct', np.nan)):
            speaker_metrics[f'speaker{spk_num}_eval_ct_median'] = metrics['eval_ct']

        # Calculate token total if both components are available
        prompt_eval = metrics.get('prompt_eval_ct', np.nan)
        eval_ct = metrics.get('eval_ct', np.nan)
        if not np.isnan(prompt_eval) and not np.isnan(eval_ct):
            token_total = prompt_eval + eval_ct
            speaker_metrics[f'speaker{spk_num}_token_total_median'] = token_total
            token_medians.append(token_total)

        duration = metrics.get('total_duration_sec', np.nan)
        if not np.isnan(duration):
            speaker_metrics[f'speaker{spk_num}_total_duration_sec_median'] = duration
            duration_medians.append(duration)

    # Calculate overall speaker metrics if we have data
    if token_medians:
        speaker_metrics['speaker_all_token_median'] = np.median(token_medians)
    else:
        speaker_metrics['speaker_all_token_median'] = np.nan

    if duration_medians:
        speaker_metrics['speaker_all_total_duration_sec_median'] = np.median(duration_medians)
    else:
        speaker_metrics['speaker_all_total_duration_sec_median'] = np.nan

    return speaker_metrics


def get_output_columns(summary_data):
    """
    Dynamically generate output column list based on available data.
    """
    base_columns = [
        'model_name', 'prompt_type', 'total_attempts', 'api_call_ct',
        'api_success_ct', 'api_success_percent', 'prediction_yes_percent',
        'prediction_correct_percent', 'confidence_median',
        # New metrics
        'f1_score', 'accuracy', 'precision', 'recall',
        'true_positive', 'true_negative', 'false_positive', 'false_negative'
    ]

    all_columns = set()
    for row in summary_data:
        all_columns.update(row.keys())

    output_columns = base_columns + [
        col for col in [
            'accuracy_median', 'f1_score_median', 'confusion_matrix',
            'confidence_median'
        ] if col in all_columns
    ]

    # Add available speaker metrics in order
    for i in range(6):
        for metric in [
            f'speaker{i}_prompt_eval_ct_median',
            f'speaker{i}_eval_ct_median',
            f'speaker{i}_token_total_median',
            f'speaker{i}_total_duration_sec_median'
        ]:
            if any(metric in row for row in summary_data):
                output_columns.append(metric)

    # Add overall speaker metrics if available
    for metric in ['speaker_all_token_median', 'speaker_all_total_duration_sec_median']:
        if any(metric in row for row in summary_data):
            output_columns.append(metric)

    return output_columns


def calculate_prediction_metrics(valid_predictions):
    """
    Calculate detailed prediction metrics for a model based on valid predictions.

    This function computes standard machine learning evaluation metrics:
    - True/False Positives/Negatives
    - Precision, Recall, F1-Score, Accuracy

    Args:
        valid_predictions: DataFrame containing valid predictions with columns:
                         'prediction' and 'actual_recidivism'

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    print("[DEBUG] Calculating detailed prediction metrics...")

    # Initialize metrics dictionary
    metrics = {
        'true_positive': 0,
        'true_negative': 0,
        'false_positive': 0,
        'false_negative': 0,
        'precision': np.nan,
        'recall': np.nan,
        'f1_score': np.nan,
        'accuracy': np.nan
    }

    # Return default metrics if no valid predictions
    if len(valid_predictions) == 0:
        print("[DEBUG] No valid predictions found")
        return metrics

    try:
        # Convert predictions to boolean
        predictions = valid_predictions['prediction'].str.upper().eq('YES')
        actuals = valid_predictions[ACTUAL_COL]

        # Calculate confusion matrix components
        metrics['true_positive'] = sum((predictions == True) & (actuals == True))
        metrics['true_negative'] = sum((predictions == False) & (actuals == False))
        metrics['false_positive'] = sum((predictions == True) & (actuals == False))
        metrics['false_negative'] = sum((predictions == False) & (actuals == True))

        print(f"[DEBUG] Confusion matrix components:")
        print(f"  True Positives: {metrics['true_positive']}")
        print(f"  True Negatives: {metrics['true_negative']}")
        print(f"  False Positives: {metrics['false_positive']}")
        print(f"  False Negatives: {metrics['false_negative']}")

        # Calculate derived metrics
        total = len(valid_predictions)

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        metrics['accuracy'] = (
            (metrics['true_positive'] + metrics['true_negative']) / total
            if total > 0 else np.nan
        )

        # Precision: TP / (TP + FP)
        precision_denominator = metrics['true_positive'] + metrics['false_positive']
        metrics['precision'] = (
            metrics['true_positive'] / precision_denominator
            if precision_denominator > 0 else np.nan
        )

        # Recall: TP / (TP + FN)
        recall_denominator = metrics['true_positive'] + metrics['false_negative']
        metrics['recall'] = (
            metrics['true_positive'] / recall_denominator
            if recall_denominator > 0 else np.nan
        )

        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if not np.isnan(metrics['precision']) and not np.isnan(metrics['recall']):
            if (metrics['precision'] + metrics['recall']) > 0:
                metrics['f1_score'] = (
                    2 * metrics['precision'] * metrics['recall'] /
                    (metrics['precision'] + metrics['recall'])
                )

        print(f"[DEBUG] Calculated metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")

    except Exception as e:
        print(f"[WARNING] Error calculating metrics: {str(e)}")

    return metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze court transcript predictions and generate summary statistics."
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        required=True,
        help='Path to step2 aggregate CSV'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    print("\n=== Starting Summary Statistics Analysis ===")
    start_time = datetime.now()

    args = parse_args()

    # Define input/output paths
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    INPUT_CSV_FULLPATH = args.input_csv

    # Derive output filenames from input CSV basename
    input_basename = os.path.splitext(os.path.basename(INPUT_CSV_FULLPATH))[0]
    ROOT_FILENAME = f"{input_basename}_stat_summary_{datetime_str}"

    OUTPUT_DIR = os.path.join("..", "transcripts_summary_reports")
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"{ROOT_FILENAME}.csv")
    OUTPUT_TXT_PATH = os.path.join(OUTPUT_DIR, f"{ROOT_FILENAME}.txt")
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{ROOT_FILENAME}.json")

    # Load and validate input data
    try:
        print(f"\n[INFO] Reading input CSV file: {INPUT_CSV_FULLPATH}")
        df = pd.read_csv(INPUT_CSV_FULLPATH)
        print(f"[INFO] Successfully loaded {len(df)} rows")

        # Backward compatibility: rename legacy column if needed
        if ACTUAL_COL not in df.columns and LEGACY_ACTUAL_COL in df.columns:
            print(f"[INFO] Column '{ACTUAL_COL}' not found. "
                  f"Falling back to legacy column '{LEGACY_ACTUAL_COL}'.")
            df.rename(columns={LEGACY_ACTUAL_COL: ACTUAL_COL}, inplace=True)

        # Verify required columns exist
        required_columns = ['model_name', 'prediction', ACTUAL_COL]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[ERROR] Missing required columns: {missing_columns}")
            return

    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {str(e)}")
        return

    # Process predictions
    print("\n[INFO] Processing predictions...")
    df['prediction_valid'] = False
    df['prediction_correct'] = False

    prediction_counts = {'valid': 0, 'invalid': 0, 'correct': 0, 'incorrect': 0}

    for idx, row in df.iterrows():
        try:
            is_correct, is_valid = process_prediction(row)
            df.at[idx, 'prediction_correct'] = is_correct
            df.at[idx, 'prediction_valid'] = is_valid

            prediction_counts['valid' if is_valid else 'invalid'] += 1
            if is_valid:
                prediction_counts['correct' if is_correct else 'incorrect'] += 1

        except Exception as e:
            print(f"[WARNING] Error processing row {idx}: {str(e)}")
            continue

    # Calculate per-model statistics
    print("\n[INFO] Calculating per-model statistics...")
    grouped = df.groupby('model_name')
    output_rows = []

    for model_name, model_df in grouped:
        try:
            print(f"\n[INFO] Processing model: {model_name}")
            row_dict = {
                'model_name': model_name,
                'prompt_type': 'agenticsim',
                'total_attempts': len(model_df),
                'api_call_ct': len(model_df),
                'api_success_ct': model_df['prediction_valid'].sum()
            }

            # Calculate percentages
            if row_dict['api_call_ct'] > 0:
                row_dict['api_success_percent'] = (
                    row_dict['api_success_ct'] / row_dict['api_call_ct'] * 100
                )
            else:
                row_dict['api_success_percent'] = 0

            valid_predictions = model_df[model_df['prediction_valid']]
            if len(valid_predictions) > 0:
                row_dict['prediction_yes_percent'] = (
                    valid_predictions['prediction'].str.upper().eq('YES').sum() /
                    len(valid_predictions) * 100
                )
                row_dict['prediction_correct_percent'] = (
                    valid_predictions['prediction_correct'].sum() /
                    len(valid_predictions) * 100
                )
            else:
                row_dict['prediction_yes_percent'] = 0
                row_dict['prediction_correct_percent'] = 0

            # Add confidence median if available
            if 'confidence' in model_df.columns:
                row_dict['confidence_median'] = safe_calculate_median(model_df['confidence'])

            # Calculate detailed prediction metrics
            prediction_metrics = calculate_prediction_metrics(valid_predictions)
            row_dict.update(prediction_metrics)

            # Calculate speaker metrics
            speaker_metrics = calculate_speaker_metrics(model_df, model_name)
            row_dict.update(speaker_metrics)

            output_rows.append(row_dict)

        except Exception as e:
            print(f"[WARNING] Error processing model {model_name}: {str(e)}")
            continue

    # Generate output files
    try:
        print("\n[INFO] Preparing output files...")

        # Get dynamic column list based on available data
        output_columns = get_output_columns(output_rows)

        # Create output DataFrame
        summary_df = pd.DataFrame(output_rows)
        summary_df = summary_df[output_columns]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

        # Save CSV
        summary_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"[INFO] CSV output written to: {OUTPUT_CSV_PATH}")

        # Save detailed text report
        with open(OUTPUT_TXT_PATH, 'w') as f:
            f.write("=== TRANSCRIPT ANALYSIS SUMMARY ===\n")
            f.write(f"Generated: {datetime.now()}\n\n")

            for row in output_rows:
                f.write(f"Model: {row['model_name']}\n")
                f.write(f"{'='*50}\n")
                for key, value in row.items():
                    if key != 'model_name':
                        f.write(f"{key}: {value}\n")
                f.write("\n")

        print(f"[INFO] Text report written to: {OUTPUT_TXT_PATH}")

    except Exception as e:
        print(f"[ERROR] Failed to generate output files: {str(e)}")

    # Calculate execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n[INFO] Analysis completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
