#!/usr/bin/env python3
"""
Compute per-model accuracy, F1, precision, recall, and confidence statistics
from AgenticSimLaw debate transcript JSON files.

Auto-discovers transcripts_ver26_*/ directories in the project root, reads
each transcript JSON, and computes binary classification metrics (YES = positive,
NO = negative) along with confidence statistics and "No Decision" rates.

Outputs:
    results/accuracy_f1_table.txt  -- human-readable and LaTeX-ready tables
    results/accuracy_f1_table.csv  -- machine-readable per-model rows

Usage:
    python scripts/compute_accuracy_f1_table.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_GLOB = "transcripts_ver26_*"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ground-truth label keys per dataset
GT_KEYS = {
    "nlsy97": "y_arrestedafter2002",
    "compas": "two_year_recid",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_experiment_dirs() -> list[Path]:
    """Return sorted list of transcripts_ver26_*/ directories in project root."""
    dirs = sorted(PROJECT_ROOT.glob(TRANSCRIPT_GLOB))
    dirs = [d for d in dirs if d.is_dir()]
    return dirs


def extract_dataset_name(exp_dir: Path) -> str:
    """Extract dataset name (nlsy97 or compas) from the directory name."""
    name = exp_dir.name  # e.g. transcripts_ver26_compas_20260209-124320
    parts = name.split("_")
    # Expected pattern: transcripts_ver26_{dataset}_{timestamp}
    for candidate in ("nlsy97", "compas"):
        if candidate in parts:
            return candidate
    # Fallback: check if the name contains the dataset string anywhere
    lower = name.lower()
    for candidate in ("nlsy97", "compas"):
        if candidate in lower:
            return candidate
    return "unknown"


def get_ground_truth(case_data: dict, dataset: str) -> bool | None:
    """Return the ground-truth boolean label, or None if unavailable."""
    gt_key = GT_KEYS.get(dataset)
    if gt_key and gt_key in case_data:
        val = case_data[gt_key]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
    return None


def normalize_prediction(pred: str) -> str:
    """Normalize prediction to YES / NO / NO_DECISION."""
    if not isinstance(pred, str):
        return "NO_DECISION"
    p = pred.strip().upper()
    if p == "YES":
        return "YES"
    if p == "NO":
        return "NO"
    return "NO_DECISION"


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Compute accuracy, precision, recall, F1 from confusion matrix counts."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_experiment(exp_dir: Path) -> list[dict]:
    """Process all transcript JSONs in an experiment directory.

    Returns a list of per-debate record dicts.
    """
    dataset = extract_dataset_name(exp_dir)
    records = []
    error_count = 0

    model_dirs = sorted(
        [d for d in exp_dir.iterdir()
         if d.is_dir() and d.name != "raw_api_responses"]
    )

    for model_dir in model_dirs:
        model_name = model_dir.name
        json_files = sorted(model_dir.glob("transcript_*.json"))

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                error_count += 1
                print(f"  WARNING: Could not read {jf.name}: {exc}")
                continue

            case_data = data.get("case", {})
            final_ruling = data.get("final_ruling", {})
            raw_pred = final_ruling.get("prediction", "")
            pred = normalize_prediction(raw_pred)
            confidence = final_ruling.get("confidence")
            gt = get_ground_truth(case_data, dataset)

            # Determine accuracy: prefer stored field, fall back to computation
            stored_acc = data.get("prediction_accurate")
            if stored_acc is not None:
                if isinstance(stored_acc, bool):
                    accurate = stored_acc
                elif isinstance(stored_acc, str):
                    accurate = stored_acc.strip().lower() == "true"
                else:
                    accurate = bool(stored_acc)
            elif gt is not None and pred in ("YES", "NO"):
                pred_bool = (pred == "YES")
                accurate = (pred_bool == gt)
            else:
                accurate = None

            # Parse case_id and repetition from filename
            # Format: transcript_row-{id}_ver-{rep}.json
            fname = jf.stem  # e.g. transcript_row-0_ver-1
            case_id = None
            repetition = None
            try:
                parts = fname.replace("transcript_row-", "").split("_ver-")
                case_id = int(parts[0])
                repetition = int(parts[1])
            except (ValueError, IndexError):
                pass

            records.append({
                "experiment": exp_dir.name,
                "dataset": dataset,
                "model": model_name,
                "case_id": case_id,
                "repetition": repetition,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "ground_truth": gt,
                "accurate": accurate,
                "confidence": confidence,
                "file": str(jf),
            })

    if error_count > 0:
        print(f"  Skipped {error_count} unreadable files in {exp_dir.name}")
    return records


def aggregate_model_stats(records: list[dict]) -> pd.DataFrame:
    """Aggregate per-debate records into per-model statistics."""
    rows = []

    # Group by experiment + model
    groups = defaultdict(list)
    for r in records:
        key = (r["experiment"], r["dataset"], r["model"])
        groups[key].append(r)

    for (experiment, dataset, model), debates in sorted(groups.items()):
        total_debates = len(debates)
        valid = [d for d in debates if d["prediction"] in ("YES", "NO")]
        no_decision = [d for d in debates if d["prediction"] == "NO_DECISION"]
        n_valid = len(valid)
        n_no_decision = len(no_decision)

        # Confusion matrix (YES = positive)
        tp = fp = tn = fn = 0
        n_accurate = 0
        confidences = []

        for d in valid:
            pred_yes = (d["prediction"] == "YES")
            gt = d["ground_truth"]
            acc = d["accurate"]

            if acc is True:
                n_accurate += 1

            if gt is not None:
                if pred_yes and gt:
                    tp += 1
                elif pred_yes and not gt:
                    fp += 1
                elif not pred_yes and not gt:
                    tn += 1
                elif not pred_yes and gt:
                    fn += 1

            if d["confidence"] is not None:
                try:
                    confidences.append(float(d["confidence"]))
                except (ValueError, TypeError):
                    pass

        metrics = compute_binary_metrics(tp, fp, tn, fn)

        rows.append({
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "total_debates": total_debates,
            "valid_predictions": n_valid,
            "no_decision_count": n_no_decision,
            "no_decision_rate": n_no_decision / total_debates if total_debates > 0 else 0.0,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "mean_confidence": np.mean(confidences) if confidences else np.nan,
            "std_confidence": np.std(confidences) if confidences else np.nan,
            "n_accurate": n_accurate,
        })

    return pd.DataFrame(rows)


def format_text_table(df: pd.DataFrame, title: str) -> str:
    """Format a DataFrame as an aligned text table."""
    lines = []
    lines.append("=" * 120)
    lines.append(title)
    lines.append("=" * 120)

    cols = [
        ("Model", "model", "{}", 45),
        ("N", "valid_predictions", "{:>4d}", 4),
        ("Acc", "accuracy", "{:.3f}", 6),
        ("Prec", "precision", "{:.3f}", 6),
        ("Rec", "recall", "{:.3f}", 6),
        ("F1", "f1", "{:.3f}", 6),
        ("Conf_u", "mean_confidence", "{:.1f}", 6),
        ("Conf_s", "std_confidence", "{:.1f}", 6),
        ("NoDec%", "no_decision_rate", "{:.1%}", 6),
    ]

    header = "  ".join(f"{name:<{width}}" for name, _, _, width in cols)
    lines.append(header)
    lines.append("-" * 120)

    for _, row in df.iterrows():
        parts = []
        for name, col, fmt, width in cols:
            val = row[col]
            if pd.isna(val):
                formatted = "N/A"
            elif isinstance(val, (int, np.integer)):
                formatted = fmt.format(int(val))
            else:
                formatted = fmt.format(val)
            parts.append(f"{formatted:<{width}}")
        lines.append("  ".join(parts))

    lines.append("")
    return "\n".join(lines)


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format a DataFrame as a LaTeX table."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l r r r r r r r r}")
    lines.append("\\toprule")
    lines.append("Model & N & Acc & Prec & Rec & F1 & $\\mu$(Conf) & $\\sigma$(Conf) & NoDec\\% \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        model = row["model"].replace("_", "\\_")
        n = int(row["valid_predictions"])
        acc = f"{row['accuracy']:.3f}"
        prec = f"{row['precision']:.3f}"
        rec = f"{row['recall']:.3f}"
        f1 = f"{row['f1']:.3f}"
        conf_u = f"{row['mean_confidence']:.1f}" if not pd.isna(row["mean_confidence"]) else "N/A"
        conf_s = f"{row['std_confidence']:.1f}" if not pd.isna(row["std_confidence"]) else "N/A"
        nodec = f"{row['no_decision_rate']:.1%}"

        lines.append(f"{model} & {n} & {acc} & {prec} & {rec} & {f1} & {conf_u} & {conf_s} & {nodec} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def format_cross_dataset_table(df: pd.DataFrame) -> str:
    """Format a cross-dataset comparison table: one row per model showing
    performance on each dataset side by side."""
    lines = []
    lines.append("=" * 140)
    lines.append("Cross-Dataset Comparison (model performance across datasets)")
    lines.append("=" * 140)

    # Pivot: group by model, show metrics for each dataset
    datasets = sorted(df["dataset"].unique())
    if len(datasets) < 2:
        lines.append("Only one dataset found; cross-dataset comparison not applicable.")
        lines.append("")
        return "\n".join(lines)

    # Build header
    header_parts = [f"{'Model':<45}"]
    for ds in datasets:
        header_parts.append(f"{'Acc_' + ds:<8}")
        header_parts.append(f"{'F1_' + ds:<8}")
        header_parts.append(f"{'N_' + ds:<6}")
    lines.append("  ".join(header_parts))
    lines.append("-" * 140)

    # Get unique models that appear in any dataset
    all_models = sorted(df["model"].unique())
    for model in all_models:
        parts = [f"{model:<45}"]
        for ds in datasets:
            subset = df[(df["model"] == model) & (df["dataset"] == ds)]
            if len(subset) > 0:
                # If a model appears in multiple experiments for the same dataset,
                # aggregate across them
                row = subset.iloc[0] if len(subset) == 1 else None
                if row is not None:
                    parts.append(f"{row['accuracy']:.3f}   ")
                    parts.append(f"{row['f1']:.3f}   ")
                    parts.append(f"{int(row['valid_predictions']):<6d}")
                else:
                    # Multiple experiment runs for same model+dataset: average
                    avg_acc = subset["accuracy"].mean()
                    avg_f1 = subset["f1"].mean()
                    total_n = int(subset["valid_predictions"].sum())
                    parts.append(f"{avg_acc:.3f}*  ")
                    parts.append(f"{avg_f1:.3f}*  ")
                    parts.append(f"{total_n:<6d}")
            else:
                parts.append(f"{'---':<8}")
                parts.append(f"{'---':<8}")
                parts.append(f"{'---':<6}")
        lines.append("  ".join(parts))

    lines.append("")
    lines.append("* = averaged across multiple experiment runs for the same model+dataset")
    lines.append("")
    return "\n".join(lines)


def main():
    print(f"Project root: {PROJECT_ROOT}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    exp_dirs = discover_experiment_dirs()
    if not exp_dirs:
        print("ERROR: No transcripts_ver26_*/ directories found.")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiment directories:")
    for d in exp_dirs:
        print(f"  {d.name}")
    print()

    # Process all experiments
    all_records = []
    for exp_dir in exp_dirs:
        print(f"Processing {exp_dir.name} ...")
        records = process_experiment(exp_dir)
        print(f"  {len(records)} debates loaded")
        all_records.extend(records)

    if not all_records:
        print("ERROR: No debate records found.")
        sys.exit(1)

    print(f"\nTotal records: {len(all_records)}")

    # Aggregate per-model stats
    stats_df = aggregate_model_stats(all_records)
    print(f"Computed statistics for {len(stats_df)} model-experiment combinations\n")

    # Build output text
    output_lines = []
    output_lines.append("AgenticSimLaw - Accuracy, F1, Precision, Recall, Confidence Statistics")
    output_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")

    # Per-experiment tables
    for exp_name in sorted(stats_df["experiment"].unique()):
        exp_df = stats_df[stats_df["experiment"] == exp_name].copy()
        exp_df = exp_df.sort_values("f1", ascending=False)
        dataset = exp_df["dataset"].iloc[0]
        title = f"Experiment: {exp_name}  (dataset: {dataset}, {len(exp_df)} models)"
        output_lines.append(format_text_table(exp_df, title))

    # Cross-dataset comparison
    output_lines.append(format_cross_dataset_table(stats_df))

    # LaTeX tables
    output_lines.append("=" * 120)
    output_lines.append("LaTeX Tables")
    output_lines.append("=" * 120)
    output_lines.append("")

    for i, exp_name in enumerate(sorted(stats_df["experiment"].unique())):
        exp_df = stats_df[stats_df["experiment"] == exp_name].copy()
        exp_df = exp_df.sort_values("f1", ascending=False)
        dataset = exp_df["dataset"].iloc[0]
        caption = f"AgenticSimLaw per-model metrics -- {dataset} ({exp_name})"
        label = f"tab:agenticsimlaw_{dataset}_{i}"
        output_lines.append(format_latex_table(exp_df, caption, label))

    # Write text output
    txt_path = RESULTS_DIR / "accuracy_f1_table.txt"
    full_text = "\n".join(output_lines)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(full_text)
    print(f"Wrote {txt_path}")

    # Write CSV output
    csv_path = RESULTS_DIR / "accuracy_f1_table.csv"
    stats_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")

    # Print summary to stdout
    print()
    print(full_text)


if __name__ == "__main__":
    main()
