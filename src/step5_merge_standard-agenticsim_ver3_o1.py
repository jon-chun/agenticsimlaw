import os
import pandas as pd

# Global variables
INPUT_ROOT_DIR = os.path.join('..', 'summary_reports')
INPUT_STANDARDLLM_FILENAME = 'transcripts_standardllm.csv'
INPUT_AGENTICSIMLLM_FILENAME = 'transcripts_agenticsimllm.csv'
OUTPUT_MERGED_FILENAME = 'transcripts_merged_ver3.csv'

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

# -------------------------------------------------------------------
# 1. Read input files
# -------------------------------------------------------------------
print(f"[INFO] Reading Standard LLM data from: {os.path.join(INPUT_ROOT_DIR, INPUT_STANDARDLLM_FILENAME)}")
standardllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_STANDARDLLM_FILENAME))
print(f"[INFO] Standard LLM DataFrame shape: {standardllm_df.shape}")

print(f"[INFO] Reading Agenticsim LLM data from: {os.path.join(INPUT_ROOT_DIR, INPUT_AGENTICSIMLLM_FILENAME)}")
agenticsimllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_AGENTICSIMLLM_FILENAME))
print(f"[INFO] Agenticsim LLM DataFrame shape: {agenticsimllm_df.shape}")

# -------------------------------------------------------------------
# 2. Create 'model_name_prompt' for each dataset
# -------------------------------------------------------------------
print("[INFO] Generating 'model_name_prompt' for standard LLM rows...")
standardllm_df['model_name_prompt'] = standardllm_df.apply(
    lambda row: f"{row['model_name']}_{standardize_prompt_type(row['prompt_type'])}", 
    axis=1
)
print("[INFO] Sample 'model_name_prompt' from standard LLM:\n", standardllm_df['model_name_prompt'].head(3))

print("[INFO] Generating 'model_name_prompt' for agenticsim LLM rows...")
agenticsimllm_df['model_name_prompt'] = agenticsimllm_df['model_name'].apply(
    lambda x: f"{x}_agenticsim"
)
print("[INFO] Sample 'model_name_prompt' from agenticsim LLM:\n", agenticsimllm_df['model_name_prompt'].head(3))

# -------------------------------------------------------------------
# 3. Select columns from each DataFrame
# -------------------------------------------------------------------
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
standard_selected = standardllm_df[standard_columns].copy()
print(f"[INFO] Standard LLM selected columns shape: {standard_selected.shape}")

print("[INFO] Selecting relevant columns for Agenticsim LLM...")
agenticsim_selected = agenticsimllm_df[agenticsim_columns].copy()
print(f"[INFO] Agenticsim LLM selected columns shape: {agenticsim_selected.shape}")

# -------------------------------------------------------------------
# 4. Rename columns with 'standard_' or 'agenticsim_' prefixes, 
#    except for 'model_name_prompt'
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 5. Merge dataframes using outer join on 'model_name_prompt'
# -------------------------------------------------------------------
print("[INFO] Merging DataFrames with outer join on 'model_name_prompt'...")
merged_df = pd.merge(
    standard_selected,
    agenticsim_selected, 
    on='model_name_prompt', 
    how='outer'
)
print(f"[INFO] Merged DataFrame shape: {merged_df.shape}")

# -------------------------------------------------------------------
# 6. Sort by 'model_name_prompt'
# -------------------------------------------------------------------
merged_df = merged_df.sort_values('model_name_prompt')
print("[INFO] DataFrame sorted by 'model_name_prompt'.")

# -------------------------------------------------------------------
# 7. Save merged DataFrame to CSV
# -------------------------------------------------------------------
output_path = os.path.join(INPUT_ROOT_DIR, OUTPUT_MERGED_FILENAME)
merged_df.to_csv(output_path, index=False)
print(f"[INFO] Merged CSV written to: {output_path}")
