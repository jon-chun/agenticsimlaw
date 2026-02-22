import os
import pandas as pd

# Global variables
INPUT_ROOT_DIR = os.path.join('..', 'summary_reports')
INPUT_STANDARDLLM_FILENAME = 'transcripts_standardllm.csv'
INPUT_AGENTICSIMLLM_FILENAME = 'transcripts_agenticsimllm.csv'
OUTPUT_MERGED_FILENAME = 'transcripts_merged_ver2.csv'

# Read input files
standardllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_STANDARDLLM_FILENAME))
agenticsimllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_AGENTICSIMLLM_FILENAME))

# Function to standardize prompt type names
def standardize_prompt_type(prompt_type):
    mapping = {
        'PromptType.SYSTEM1': 'system1',
        'PromptType.COT': 'cot',
        'PromptType.COT_NSHOT': 'cot-nshot'
    }
    return mapping.get(prompt_type, prompt_type)

# Create model_name_prompt for standardllm_df
standardllm_df['model_name_prompt'] = standardllm_df.apply(
    lambda row: f"{row['model_name']}_{standardize_prompt_type(row['prompt_type'])}", 
    axis=1
)

# Create model_name_prompt for agenticsimllm_df
agenticsimllm_df['model_name_prompt'] = agenticsimllm_df['model_name'].apply(
    lambda x: f"{x}_agenticsim"
)

# Select columns from standardllm_df
standard_columns = [
    'model_name_prompt', 'model_params', 'model_quantization',
    'successful_attempts', 'failed_attempts', 'execution_time_median',
    'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
    'eval_duration_median', 'true_positives', 'true_negatives',
    'false_positives', 'false_negatives', 'f1_score'
]

# Select columns from agenticsimllm_df
agenticsim_columns = [
    'model_name_prompt', 'model_params', 'model_quantization',
    'successful_attempts', 'failed_attempts', 'execution_time_median',
    'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
    'eval_duration_median', 'true_positives', 'true_negatives',
    'false_positives', 'false_negatives', 'f1_score',
    'prompt_eval_count_median', 'eval_count_median', 'confidence_txt_missing_count'
]

# Create selected dataframes
standard_selected = standardllm_df[standard_columns].copy()
agenticsim_selected = agenticsimllm_df[agenticsim_columns].copy()

# Rename columns with prefixes (except model_name_prompt)
standard_selected.columns = ['model_name_prompt' if col == 'model_name_prompt' 
                           else 'standard_' + col for col in standard_selected.columns]
agenticsim_selected.columns = ['model_name_prompt' if col == 'model_name_prompt' 
                             else 'agenticsim_' + col for col in agenticsim_selected.columns]

# Merge dataframes using outer join to keep all rows from both dataframes
merged_df = pd.merge(standard_selected, agenticsim_selected, 
                    on='model_name_prompt', 
                    how='outer')

# Sort by model_name_prompt
merged_df = merged_df.sort_values('model_name_prompt')

# Save merged dataframe
merged_df.to_csv(os.path.join(INPUT_ROOT_DIR, OUTPUT_MERGED_FILENAME), index=False)