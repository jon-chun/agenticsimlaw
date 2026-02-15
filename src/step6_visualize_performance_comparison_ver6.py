#!/usr/bin/env python3
"""
step6_visualize_performance_comparison_ver4.py

Revised code with the following changes:
  1) PLOT_MODEL_CT = 16 (a global variable), which determines the maximum number of models
     to show in the bar plots for each prompt_type. If a prompt_type has more than 16 models,
     we only plot the top 16 sorted by F1-score. Otherwise, we plot them all.
  2) A new function create_performance_plot_report() that summarizes F1/accuracy metrics
     for each model/prompt_type as text. We also include a success_ratio for standard
     vs. agenticsim runs (if data is present).
  3) We keep 'standard_f1_score' and 'agenticsim_f1_score' for the different prompt_types,
     unify them into a single 'f1_score' integer [0..100]. Similarly for accuracy.
  4) The entire new code base is printed below.

Date:   2025-01-26
"""

import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

###############################################################################
# Global Variables for Input/Output
###############################################################################
INPUT_ROOT_DIR = os.path.join("..", "summary_reports_working")
INPUT_FILENAME = "transcripts_merged_manual_ver2.csv"
INPUT_FILENAME_FULLPATH = os.path.join(INPUT_ROOT_DIR, INPUT_FILENAME)

OUTPUT_ROOT_DIR = os.path.join("..", "summary_reports", "plots")

###############################################################################
# Other Global Configuration
###############################################################################
# We unify system1 -> standard1
EXPECTED_PROMPT_TYPES = ["standard1", "cot", "cot-nshot", "agenticsim"]

# The set of model names we want to analyze.
ENSEMBLE_MODEL_LS = [
    "aya-expanse:8b-q4_K_M",
    "deepseek-r1:7b",
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",
    "llama3_1_8b_instruct_q4_k_m",
    "marco-o1:7b-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "olmo2:7b-1124-instruct_q4_K_M",
    "phi4:14b-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "tulu3:8b-q4_K_M",
]

# If you want to exclude certain models:
EXCLUDE_MODEL_LS = []

# Logging level
LOG_LEVEL = logging.DEBUG

# Maximum number of models to plot per prompt_type
PLOT_MODEL_CT = 16

###############################################################################
# Logging Setup
###############################################################################
def setup_logging():
    """Creates a logger for the script, setting up a console handler."""
    logger = logging.getLogger("prompt_performance_logger")
    logger.setLevel(LOG_LEVEL)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logging()

###############################################################################
# CSV Parsing and Data Preparation
###############################################################################
def parse_data_new_file(input_csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV using header=0 so that the first row is used as column headers.
    Expects columns like:
        model_name, prompt_type, standard_f1_score, standard_prediction_accuracy, ...
        agenticsim_f1_score, agenticsim_prediction_accuracy, ...
        standard_successful_attempts, standard_failed_attempts, ...
        agenticsim_successful_attempts, agenticsim_failed_attempts, etc.
    Unifies 'system1' -> 'standard1'.

    Then we do the following:
      - Keep both standard_f1_score & standard_prediction_accuracy,
        plus agenticsim_f1_score & agenticsim_prediction_accuracy.
      - For each row, we create 'f1_score' and 'accuracy' columns:
          If prompt_type in ['system1','standard1','cot','cot-nshot']:
            f1_score    = standard_f1_score
            accuracy    = standard_prediction_accuracy
          If prompt_type == 'agenticsim':
            f1_score    = agenticsim_f1_score
            accuracy    = agenticsim_prediction_accuracy
      - We convert these to integer values in [0..100].
        If the raw value is between 0..1.5, multiply by 100; else assume 0..100 range.

      - We also keep track of successful_attempts and failed_attempts in unified columns:
          success_attempts = standard_successful_attempts if standard/cot/cot-nshot
                            = agenticsim_successful_attempts if agenticsim
          failed_attempts  = standard_failed_attempts or agenticsim_failed_attempts
        so we can compute a success ratio later.

    Returns a DataFrame with at least columns:
       [model_name, prompt_type, f1_score, accuracy, success_attempts, failed_attempts, ...]
    """
    logger.info(f"Reading CSV from: {input_csv_path}")
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Could not find input CSV file: {input_csv_path}")

    df_raw = pd.read_csv(
        input_csv_path,
        header=0,             # read real column headers from first row
        comment="#",
        skipinitialspace=True
    )

    # Unify 'system1' -> 'standard1'
    if "prompt_type" in df_raw.columns:
        df_raw["prompt_type"] = df_raw["prompt_type"].replace("system1", "standard1")
    else:
        logger.error("No 'prompt_type' column found in CSV. Check your input file.")
        return pd.DataFrame()

    # Keep only known prompt types
    before_count = len(df_raw)
    df_raw = df_raw[df_raw["prompt_type"].isin(EXPECTED_PROMPT_TYPES)]
    after_count = len(df_raw)
    logger.info(f"Dropped {before_count - after_count} rows with unknown prompt_type")

    # Safe float conversion
    def safe_float(val):
        """Convert to float, or return None if not numeric."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # Some known columns we want to parse if present:
    wanted_numeric_cols = [
        "standard_f1_score",
        "standard_prediction_accuracy",
        "agenticsim_f1_score",
        "agenticsim_prediction_accuracy",
        "standard_successful_attempts",
        "standard_failed_attempts",
        "agenticsim_successful_attempts",
        "agenticsim_failed_attempts",
    ]
    for col in wanted_numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(safe_float)
        else:
            logger.warning(f"Column '{col}' not found in CSV. Some data might be missing.")

    # Now create the unified columns
    f1_list = []
    acc_list = []
    success_list = []
    fail_list = []

    for idx, row in df_raw.iterrows():
        ptype = row["prompt_type"]
        if ptype in ["standard1", "cot", "cot-nshot"]:
            raw_f1 = row.get("standard_f1_score", None)
            raw_acc = row.get("standard_prediction_accuracy", None)
            raw_succ = row.get("standard_successful_attempts", None)
            raw_fail = row.get("standard_failed_attempts", None)
        elif ptype == "agenticsim":
            raw_f1 = row.get("agenticsim_f1_score", None)
            raw_acc = row.get("agenticsim_prediction_accuracy", None)
            raw_succ = row.get("agenticsim_successful_attempts", None)
            raw_fail = row.get("agenticsim_failed_attempts", None)
        else:
            raw_f1 = None
            raw_acc = None
            raw_succ = None
            raw_fail = None

        # Convert F1 to integer [0..100]
        if raw_f1 is not None:
            if 0 <= raw_f1 <= 1.5:
                raw_f1 *= 100
            raw_f1 = int(round(raw_f1)) if raw_f1 >= 0 else 0

        # Convert Accuracy to integer [0..100]
        if raw_acc is not None:
            if 0 <= raw_acc <= 1.5:
                raw_acc *= 100
            raw_acc = int(round(raw_acc)) if raw_acc >= 0 else 0

        # success/fail as floats => store them as int for clarity
        if raw_succ is not None:
            raw_succ = int(round(raw_succ)) if raw_succ >= 0 else 0
        if raw_fail is not None:
            raw_fail = int(round(raw_fail)) if raw_fail >= 0 else 0

        f1_list.append(raw_f1)
        acc_list.append(raw_acc)
        success_list.append(raw_succ)
        fail_list.append(raw_fail)

    df_raw["f1_score"] = f1_list
    df_raw["accuracy"] = acc_list
    df_raw["success_attempts"] = success_list
    df_raw["failed_attempts"] = fail_list

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"DataFrame columns: {list(df_raw.columns)}")
        logger.debug(f"Sample data:\n{df_raw.head(5)}")

    return df_raw

###############################################################################
# Filtering & Validation
###############################################################################
def filter_models_and_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Filters out any rows whose 'model_name' is not in ENSEMBLE_MODEL_LS,
       or is in EXCLUDE_MODEL_LS.
    2) Ensures that only those models which have all required prompt_types
       (standard1, cot, cot-nshot, agenticsim) are kept.
    3) Logs debug info about partial coverage if present.

    Returns:
        A filtered DataFrame containing only the fully covered models.
    """
    logger.info("Filtering DataFrame by model list and prompt type coverage...")

    df_filtered = df[df["model_name"].isin(ENSEMBLE_MODEL_LS)]
    df_filtered = df_filtered[~df_filtered["model_name"].isin(EXCLUDE_MODEL_LS)]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"After model_name filtering, we have {len(df_filtered)} rows.")

    # Only keep models that have all 4 prompt_types
    valid_model_names = []
    for model_name, subdf in df_filtered.groupby("model_name"):
        prompt_types_found = set(subdf["prompt_type"].unique())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Model {model_name} has prompt types: {prompt_types_found}")
        # Check if all expected prompt_types exist
        if all(pt in prompt_types_found for pt in EXPECTED_PROMPT_TYPES):
            valid_model_names.append(model_name)

    df_final = df_filtered[df_filtered["model_name"].isin(valid_model_names)].copy()

    logger.info(f"  - Original rows: {len(df)}")
    logger.info(f"  - After filtering & coverage check: {len(df_final)}")
    logger.info("Models retained with full prompt coverage:")
    for m in sorted(df_final["model_name"].unique()):
        logger.info(f"    * {m}")

    if logger.isEnabledFor(logging.DEBUG) and len(df_final) == 0:
        missing_coverage = sorted(df_filtered["model_name"].unique())
        logger.debug(f"No models found with full coverage. Missing at least one prompt type: {missing_coverage}")

    return df_final

###############################################################################
# Plotting Functions
###############################################################################
def create_plot_prompt_performance_plot(df: pd.DataFrame,
                                        output_dir: str,
                                        root_filename: str) -> None:
    """
    Creates FOUR separate bar plots (one for each prompt_type in EXPECTED_PROMPT_TYPES).
    Each bar plot does the following:
      - Sort models in descending order by 'f1_score' for that prompt_type
      - If more than PLOT_MODEL_CT models, only plot the top PLOT_MODEL_CT
      - For each model_name on the x-axis, create two side-by-side bars:
          1) f1_score (integer 0..100)
          2) accuracy (integer 0..100)
      - Annotate numeric values on top of each bar, rotated 90° CCW
      - Save as a PNG with a name like:
            {root_filename}_{prompt_type}_prompt_perf.png

    We assume 'f1_score' and 'accuracy' are integer columns [0..100].
    """
    sns.set_theme(style="whitegrid")

    for prompt_type in EXPECTED_PROMPT_TYPES:
        # Subset for that prompt_type
        df_sub = df[df["prompt_type"] == prompt_type].copy()
        if df_sub.empty:
            logger.warning(f"No data for prompt_type={prompt_type}. Skipping plot.")
            continue

        # Sort descending by f1_score
        df_sub = df_sub.sort_values("f1_score", ascending=False)

        # Only plot rows with non-null f1_score and accuracy
        df_sub = df_sub.dropna(subset=["f1_score", "accuracy"])
        if df_sub.empty:
            logger.warning(f"After dropna, no valid numeric data for {prompt_type}. Skipping plot.")
            continue

        # If more models than PLOT_MODEL_CT, take top PLOT_MODEL_CT
        if len(df_sub) > PLOT_MODEL_CT:
            df_sub = df_sub.head(PLOT_MODEL_CT)
            logger.info(
                f"For prompt_type={prompt_type}, we have more than {PLOT_MODEL_CT} models. "
                f"Plotting only top {PLOT_MODEL_CT} by F1-score."
            )

        # Prepare for side-by-side barplot
        plot_df = pd.DataFrame({
            "model_name": df_sub["model_name"],
            "F1_score": df_sub["f1_score"],
            "Accuracy": df_sub["accuracy"]
        })

        # Melt so we get two bars per model
        melted = plot_df.melt(id_vars=["model_name"], var_name="metric", value_name="value")

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="model_name",
            y="value",
            hue="metric",
            data=melted,
            palette="muted"
        )

        plt.title(f"Prompt Performance: {prompt_type} (F1 & Accuracy)")
        # Limit y-axis to [0..100] since these are int percentages
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha="right", fontsize=8)

        # Label each bar with numeric value
        for c in ax.containers:
            ax.bar_label(
                c,
                fmt="%.0f",   # integer, no decimals
                label_type="edge",
                rotation=90,
                fontsize=8
            )

        plt.legend(title="Metric", labels=["F1 Score", "Accuracy"])
        plt.tight_layout()

        # Save figure
        plot_filename = f"{root_filename}_{prompt_type}_prompt_perf.png"
        fullpath = os.path.join(output_dir, plot_filename)
        plt.savefig(fullpath)
        plt.close()
        logger.info(f"Saved prompt performance plot for '{prompt_type}' to {fullpath}")

###############################################################################
# Reporting Function
###############################################################################
def create_performance_plot_report(df: pd.DataFrame) -> None:
    """
    Generates a text-based summary of F1/Accuracy metrics by model/prompt_type,
    plus any non-plotted summary stats like success ratio.

    We'll compute success_ratio = success_attempts / (success_attempts + failed_attempts).

    We simply LOG (INFO level) a table-like format. In real usage, you might want
    to write to a file or produce a more elegant markdown table.
    """
    logger.info("Generating text-based performance report...")

    # For consistent ordering
    df_sorted = df.sort_values(["prompt_type", "f1_score"], ascending=[True, False])

    # We'll create a short text table in logs
    # Columns: model_name, prompt_type, f1_score, accuracy, success_ratio
    def safe_ratio(succ, fail):
        if succ is None or fail is None:
            return None
        denom = succ + fail
        if denom <= 0:
            return None
        return round(100.0 * succ / denom, 2)

    logger.info("----- Performance Report (f1/acc/success%) by model & prompt -----")
    logger.info(f"{'Model':35s}  {'Prompt':10s}  {'F1':>5s}  {'Acc':>5s}  {'Success%':>9s}")
    for idx, row in df_sorted.iterrows():
        model_nm = row["model_name"][:35]
        ptype = row["prompt_type"][:10]
        f1v = row["f1_score"]
        accv = row["accuracy"]
        succ_ratio = safe_ratio(row["success_attempts"], row["failed_attempts"])
        succ_str = f"{succ_ratio:.2f}" if succ_ratio is not None else "N/A"
        logger.info(f"{model_nm:35s}  {ptype:10s}  {f1v:5.0f}  {accv:5.0f}  {succ_str:>9s}")

    logger.info("----- End of Performance Report -----")

###############################################################################
# Main Execution
###############################################################################
def main():
    """
    Main driver:
      1) Read the CSV from the globally defined INPUT_FILENAME_FULLPATH
      2) Parse and unify standard/agenticsim f1/accuracy into [0..100] integer columns
      3) Filter the data so we only keep models with full coverage of all 4 prompt_types
      4) Create bar plots for each prompt_type in descending F1-score (top PLOT_MODEL_CT)
      5) Generate a text-based performance plot report
    """
    logger.info("Starting main execution for new prompt-based performance plotting")
    logger.info(f"Input CSV: {INPUT_FILENAME_FULLPATH}")
    logger.info(f"Output Dir: {OUTPUT_ROOT_DIR}")
    logger.info(f"PLOT_MODEL_CT: {PLOT_MODEL_CT}")

    if not os.path.isdir(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
        logger.info(f"Created output directory: {OUTPUT_ROOT_DIR}")

    # 1) Parse CSV
    df_all = parse_data_new_file(INPUT_FILENAME_FULLPATH)
    if df_all.empty:
        logger.warning("Parsed DataFrame is empty. Exiting.")
        return

    # 2) Filter data by model coverage
    df_filtered = filter_models_and_prompts(df_all)
    if df_filtered.empty:
        logger.warning("No data remains after coverage filtering. Exiting without plots.")
        return

    # 3) Generate bar plots
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_filename = f"prompt_performance_{datetime_str}"
    create_plot_prompt_performance_plot(df_filtered, OUTPUT_ROOT_DIR, root_filename)

    # 4) Generate text-based performance report
    create_performance_plot_report(df_filtered)

    logger.info("All done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logger.error(f"Script execution failed: {str(ex)}")
        sys.exit(1)
