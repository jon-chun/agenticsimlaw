"""
step5_merge_standard-agenticsim_ver4.py

Dataset-aware version of step5_merge_standard-agenticsim_ver3_o1.py.
Adds argparse CLI, a 'dataset' column, and graceful handling of missing columns.
"""

import argparse
import os

import pandas as pd


def standardize_prompt_type(prompt_type):
    """
    Maps known prompt types to simplified names,
    e.g. 'PromptType.SYSTEM1' -> 'system1',
         'PromptType.COT_NSHOT' -> 'cot-nshot'.
    Unmapped types return as-is.
    """
    mapping = {
        'PromptType.SYSTEM1': 'system1',
        'PromptType.COT': 'cot',
        'PromptType.COT_NSHOT': 'cot-nshot'
    }
    return mapping.get(prompt_type, prompt_type)


def select_available_columns(df, desired_columns, label):
    """Return a copy of df with only the columns that actually exist.

    Logs a warning for each missing column so the user knows what was skipped.
    """
    present = [col for col in desired_columns if col in df.columns]
    missing = [col for col in desired_columns if col not in df.columns]
    if missing:
        print(f"[WARN] {label}: skipping missing columns: {missing}")
    return df[present].copy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge StandardLLM and AgenticSimLaw summary CSVs into a single report."
    )
    parser.add_argument(
        '--standardllm',
        required=True,
        help='Path to StandardLLM summary CSV'
    )
    parser.add_argument(
        '--agenticsim',
        required=True,
        help='Path to AgenticSimLaw summary CSV'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output merged CSV path'
    )
    parser.add_argument(
        '--dataset',
        default='nlsy97',
        help='Dataset label added as a column (default: nlsy97)'
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Read input files
    # -----------------------------------------------------------------
    print(f"[INFO] Reading Standard LLM data from: {args.standardllm}")
    standardllm_df = pd.read_csv(args.standardllm)
    print(f"[INFO] Standard LLM DataFrame shape: {standardllm_df.shape}")

    print(f"[INFO] Reading Agenticsim LLM data from: {args.agenticsim}")
    agenticsimllm_df = pd.read_csv(args.agenticsim)
    print(f"[INFO] Agenticsim LLM DataFrame shape: {agenticsimllm_df.shape}")

    # -----------------------------------------------------------------
    # 2. Create 'model_name_prompt' for each dataset
    # -----------------------------------------------------------------
    print("[INFO] Generating 'model_name_prompt' for standard LLM rows...")
    standardllm_df['model_name_prompt'] = standardllm_df.apply(
        lambda row: f"{row['model_name']}_{standardize_prompt_type(row['prompt_type'])}",
        axis=1
    )
    print("[INFO] Sample 'model_name_prompt' from standard LLM:\n",
          standardllm_df['model_name_prompt'].head(3))

    print("[INFO] Generating 'model_name_prompt' for agenticsim LLM rows...")
    agenticsimllm_df['model_name_prompt'] = agenticsimllm_df['model_name'].apply(
        lambda x: f"{x}_agenticsim"
    )
    print("[INFO] Sample 'model_name_prompt' from agenticsim LLM:\n",
          agenticsimllm_df['model_name_prompt'].head(3))

    # -----------------------------------------------------------------
    # 3. Select columns from each DataFrame (skip missing gracefully)
    # -----------------------------------------------------------------
    standard_columns = [
        'model_name_prompt', 'model_params', 'model_quantization',
        'successful_attempts', 'failed_attempts', 'execution_time_median',
        'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
        'eval_duration_median', 'true_positives', 'true_negatives',
        'false_positives', 'false_negatives', 'f1_score'
    ]

    agenticsim_columns = [
        'model_name_prompt', 'model_params', 'model_quantization',
        'successful_attempts', 'failed_attempts', 'execution_time_median',
        'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
        'eval_duration_median', 'true_positives', 'true_negatives',
        'false_positives', 'false_negatives', 'f1_score',
        'prompt_eval_count_median', 'eval_count_median', 'confidence_txt_missing_count'
    ]

    print("[INFO] Selecting relevant columns for Standard LLM...")
    standard_selected = select_available_columns(standardllm_df, standard_columns, "StandardLLM")
    print(f"[INFO] Standard LLM selected columns shape: {standard_selected.shape}")

    print("[INFO] Selecting relevant columns for Agenticsim LLM...")
    agenticsim_selected = select_available_columns(agenticsimllm_df, agenticsim_columns, "AgenticSimLLM")
    print(f"[INFO] Agenticsim LLM selected columns shape: {agenticsim_selected.shape}")

    # -----------------------------------------------------------------
    # 4. Rename columns with 'standard_' or 'agenticsim_' prefixes,
    #    except for 'model_name_prompt'
    # -----------------------------------------------------------------
    standard_selected.columns = [
        'model_name_prompt' if col == 'model_name_prompt'
        else 'standard_' + col
        for col in standard_selected.columns
    ]

    agenticsim_selected.columns = [
        'model_name_prompt' if col == 'model_name_prompt'
        else 'agenticsim_' + col
        for col in agenticsim_selected.columns
    ]

    # -----------------------------------------------------------------
    # 5. Merge dataframes using outer join on 'model_name_prompt'
    # -----------------------------------------------------------------
    print("[INFO] Merging DataFrames with outer join on 'model_name_prompt'...")
    merged_df = pd.merge(
        standard_selected,
        agenticsim_selected,
        on='model_name_prompt',
        how='outer'
    )
    print(f"[INFO] Merged DataFrame shape: {merged_df.shape}")

    # -----------------------------------------------------------------
    # 6. Add dataset column
    # -----------------------------------------------------------------
    merged_df.insert(0, 'dataset', args.dataset)
    print(f"[INFO] Added 'dataset' column with value: {args.dataset}")

    # -----------------------------------------------------------------
    # 7. Sort by 'model_name_prompt'
    # -----------------------------------------------------------------
    merged_df = merged_df.sort_values('model_name_prompt')
    print("[INFO] DataFrame sorted by 'model_name_prompt'.")

    # -----------------------------------------------------------------
    # 8. Save merged DataFrame to CSV
    # -----------------------------------------------------------------
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(args.output, index=False)
    print(f"[INFO] Merged CSV written to: {args.output}")


if __name__ == '__main__':
    main()
