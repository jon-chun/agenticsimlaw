#!/usr/bin/env python3
"""
Compute cross-dataset (NLSY97 vs COMPAS) performance comparison for the
AgenticSimLaw multi-agent courtroom debate framework.

Auto-discovers transcripts_ver26_nlsy97_* and transcripts_ver26_compas_*
directories in the project root, reads all transcript JSON files, computes
per-model binary classification metrics (accuracy, F1, precision, recall),
and produces a side-by-side comparison table with paired statistical tests.

Outputs:
    results/dataset_comparison_table.txt  -- human-readable comparison tables
    results/dataset_comparison_table.csv  -- machine-readable per-model rows

Usage:
    python scripts/compute_dataset_comparison.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
TRANSCRIPT_GLOB_NLSY97 = "transcripts_ver26_nlsy97_*"
TRANSCRIPT_GLOB_COMPAS = "transcripts_ver26_compas_*"

# Ground-truth label keys per dataset
GT_KEYS = {
    "nlsy97": "y_arrestedafter2002",
    "compas": "two_year_recid",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_experiment_dirs(dataset: str) -> list[Path]:
    """Return sorted list of transcript directories for the given dataset."""
    if dataset == "nlsy97":
        glob_pattern = TRANSCRIPT_GLOB_NLSY97
    elif dataset == "compas":
        glob_pattern = TRANSCRIPT_GLOB_COMPAS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dirs = sorted(PROJECT_DIR.glob(glob_pattern))
    dirs = [d for d in dirs if d.is_dir()]
    return dirs


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
# Transcript processing
# ---------------------------------------------------------------------------
def process_experiment(exp_dir: Path, dataset: str) -> list[dict]:
    """Process all transcript JSONs in an experiment directory.

    Returns a list of per-debate record dicts.
    """
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
            gt = get_ground_truth(case_data, dataset)

            # Determine accuracy
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

            records.append({
                "experiment": exp_dir.name,
                "dataset": dataset,
                "model": model_name,
                "prediction": pred,
                "ground_truth": gt,
                "accurate": accurate,
                "file": str(jf),
            })

    if error_count > 0:
        print(f"  Skipped {error_count} unreadable files in {exp_dir.name}")
    return records


def aggregate_model_stats(records: list[dict]) -> pd.DataFrame:
    """Aggregate per-debate records into per-(dataset, model) statistics.

    If a model appears in multiple experiment directories for the same dataset,
    aggregate across all of them.
    """
    rows = []

    # Group by dataset + model (merge across experiment dirs for same dataset)
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["model"])
        groups[key].append(r)

    for (dataset, model), debates in sorted(groups.items()):
        total_debates = len(debates)
        valid = [d for d in debates if d["prediction"] in ("YES", "NO")]
        no_decision = [d for d in debates if d["prediction"] == "NO_DECISION"]
        n_valid = len(valid)
        n_no_decision = len(no_decision)

        # Confusion matrix (YES = positive)
        tp = fp = tn = fn = 0
        for d in valid:
            pred_yes = (d["prediction"] == "YES")
            gt = d["ground_truth"]
            if gt is not None:
                if pred_yes and gt:
                    tp += 1
                elif pred_yes and not gt:
                    fp += 1
                elif not pred_yes and not gt:
                    tn += 1
                elif not pred_yes and gt:
                    fn += 1

        metrics = compute_binary_metrics(tp, fp, tn, fn)

        rows.append({
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
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------
def build_comparison_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison table: one row per model with metrics
    from both datasets and deltas.

    Only models appearing in BOTH datasets are included in the delta columns.
    Models in only one dataset are still listed with dashes for the other.
    """
    nlsy97 = stats_df[stats_df["dataset"] == "nlsy97"].set_index("model")
    compas = stats_df[stats_df["dataset"] == "compas"].set_index("model")

    all_models = sorted(set(nlsy97.index) | set(compas.index))

    rows = []
    for model in all_models:
        row = {"model": model}

        if model in nlsy97.index:
            row["nlsy97_acc"] = nlsy97.loc[model, "accuracy"]
            row["nlsy97_f1"] = nlsy97.loc[model, "f1"]
            row["nlsy97_prec"] = nlsy97.loc[model, "precision"]
            row["nlsy97_rec"] = nlsy97.loc[model, "recall"]
            row["nlsy97_n"] = int(nlsy97.loc[model, "valid_predictions"])
        else:
            row["nlsy97_acc"] = np.nan
            row["nlsy97_f1"] = np.nan
            row["nlsy97_prec"] = np.nan
            row["nlsy97_rec"] = np.nan
            row["nlsy97_n"] = 0

        if model in compas.index:
            row["compas_acc"] = compas.loc[model, "accuracy"]
            row["compas_f1"] = compas.loc[model, "f1"]
            row["compas_prec"] = compas.loc[model, "precision"]
            row["compas_rec"] = compas.loc[model, "recall"]
            row["compas_n"] = int(compas.loc[model, "valid_predictions"])
        else:
            row["compas_acc"] = np.nan
            row["compas_f1"] = np.nan
            row["compas_prec"] = np.nan
            row["compas_rec"] = np.nan
            row["compas_n"] = 0

        # Deltas (COMPAS - NLSY97): only for models in both
        if model in nlsy97.index and model in compas.index:
            row["delta_acc"] = row["compas_acc"] - row["nlsy97_acc"]
            row["delta_f1"] = row["compas_f1"] - row["nlsy97_f1"]
        else:
            row["delta_acc"] = np.nan
            row["delta_f1"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_aggregate_stats(stats_df: pd.DataFrame) -> dict:
    """Compute mean and std of accuracy and F1 across models for each dataset."""
    agg = {}
    for dataset in ("nlsy97", "compas"):
        subset = stats_df[stats_df["dataset"] == dataset]
        if len(subset) == 0:
            continue
        agg[dataset] = {
            "n_models": len(subset),
            "acc_mean": subset["accuracy"].mean(),
            "acc_std": subset["accuracy"].std(),
            "f1_mean": subset["f1"].mean(),
            "f1_std": subset["f1"].std(),
            "prec_mean": subset["precision"].mean(),
            "prec_std": subset["precision"].std(),
            "rec_mean": subset["recall"].mean(),
            "rec_std": subset["recall"].std(),
        }
    return agg


def run_paired_tests(stats_df: pd.DataFrame) -> dict | None:
    """Run paired t-tests on accuracy and F1 for models appearing in both datasets.

    Returns dict with test results, or None if fewer than 2 paired models.
    """
    nlsy97 = stats_df[stats_df["dataset"] == "nlsy97"].set_index("model")
    compas = stats_df[stats_df["dataset"] == "compas"].set_index("model")
    paired_models = sorted(set(nlsy97.index) & set(compas.index))

    if len(paired_models) < 2:
        return None

    nlsy97_acc = np.array([nlsy97.loc[m, "accuracy"] for m in paired_models])
    compas_acc = np.array([compas.loc[m, "accuracy"] for m in paired_models])
    nlsy97_f1 = np.array([nlsy97.loc[m, "f1"] for m in paired_models])
    compas_f1 = np.array([compas.loc[m, "f1"] for m in paired_models])

    acc_t, acc_p = scipy_stats.ttest_rel(nlsy97_acc, compas_acc)
    f1_t, f1_p = scipy_stats.ttest_rel(nlsy97_f1, compas_f1)

    return {
        "n_paired": len(paired_models),
        "paired_models": paired_models,
        "acc_t_stat": acc_t,
        "acc_p_value": acc_p,
        "acc_mean_diff": float(np.mean(nlsy97_acc - compas_acc)),
        "f1_t_stat": f1_t,
        "f1_p_value": f1_p,
        "f1_mean_diff": float(np.mean(nlsy97_f1 - compas_f1)),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_comparison_text(
    comp_df: pd.DataFrame,
    agg: dict,
    test_results: dict | None,
) -> str:
    """Format the full human-readable comparison report."""
    W = 140
    lines = []

    lines.append("=" * W)
    lines.append("AgenticSimLaw - Cross-Dataset Performance Comparison (NLSY97 vs COMPAS)")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)
    lines.append("")

    # ----- Section 1: Side-by-side table -----
    lines.append("1. SIDE-BY-SIDE MODEL COMPARISON")
    lines.append("-" * W)

    header = (
        f"{'Model':<45}  "
        f"{'NLSY97 Acc':>10}  {'NLSY97 F1':>9}  {'N':>4}  "
        f"{'COMPAS Acc':>10}  {'COMPAS F1':>9}  {'N':>4}  "
        f"{'Delta Acc':>9}  {'Delta F1':>8}"
    )
    lines.append(header)
    lines.append("-" * W)

    # Sort by NLSY97 accuracy descending (models present in both first)
    comp_sorted = comp_df.copy()
    comp_sorted["_both"] = comp_sorted["delta_acc"].notna().astype(int)
    comp_sorted = comp_sorted.sort_values(
        ["_both", "nlsy97_acc"], ascending=[False, False]
    )
    comp_sorted = comp_sorted.drop(columns=["_both"])

    for _, row in comp_sorted.iterrows():
        model = row["model"]
        if len(model) > 43:
            model = model[:40] + "..."

        def _fmt_metric(val, width=10):
            if pd.isna(val):
                return f"{'---':>{width}}"
            return f"{val:>{width}.3f}"

        def _fmt_n(val, width=4):
            if val == 0 or pd.isna(val):
                return f"{'---':>{width}}"
            return f"{int(val):>{width}d}"

        def _fmt_delta(val, width=9):
            if pd.isna(val):
                return f"{'---':>{width}}"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:>{width - 1}.3f}"

        line = (
            f"{model:<45}  "
            f"{_fmt_metric(row['nlsy97_acc'])}  "
            f"{_fmt_metric(row['nlsy97_f1'], 9)}  "
            f"{_fmt_n(row['nlsy97_n'])}  "
            f"{_fmt_metric(row['compas_acc'])}  "
            f"{_fmt_metric(row['compas_f1'], 9)}  "
            f"{_fmt_n(row['compas_n'])}  "
            f"{_fmt_delta(row['delta_acc'])}  "
            f"{_fmt_delta(row['delta_f1'], 8)}"
        )
        lines.append(line)

    lines.append("-" * W)
    lines.append("")
    lines.append("  Delta = COMPAS - NLSY97 (positive means higher on COMPAS)")
    lines.append("  '---' indicates the model was not evaluated on that dataset")
    lines.append("")

    # ----- Section 2: Aggregate statistics -----
    lines.append("")
    lines.append("2. AGGREGATE STATISTICS (mean +/- std across models)")
    lines.append("=" * W)
    lines.append("")

    agg_header = (
        f"{'Dataset':<12}  {'Models':>6}  "
        f"{'Acc Mean':>8}  {'Acc Std':>7}  "
        f"{'F1 Mean':>7}  {'F1 Std':>6}  "
        f"{'Prec Mean':>9}  {'Prec Std':>8}  "
        f"{'Rec Mean':>8}  {'Rec Std':>7}"
    )
    lines.append(agg_header)
    lines.append("-" * W)

    for dataset in ("nlsy97", "compas"):
        if dataset not in agg:
            continue
        a = agg[dataset]
        line = (
            f"{dataset.upper():<12}  {a['n_models']:>6d}  "
            f"{a['acc_mean']:>8.3f}  {a['acc_std']:>7.3f}  "
            f"{a['f1_mean']:>7.3f}  {a['f1_std']:>6.3f}  "
            f"{a['prec_mean']:>9.3f}  {a['prec_std']:>8.3f}  "
            f"{a['rec_mean']:>8.3f}  {a['rec_std']:>7.3f}"
        )
        lines.append(line)

    lines.append("")

    # ----- Section 3: Statistical tests -----
    lines.append("")
    lines.append("3. PAIRED STATISTICAL TESTS (models appearing in both datasets)")
    lines.append("=" * W)
    lines.append("")

    if test_results is None:
        lines.append("  Fewer than 2 models appear in both datasets; "
                      "paired tests cannot be computed.")
    else:
        lines.append(f"  Number of paired models: {test_results['n_paired']}")
        lines.append(f"  Paired models: {', '.join(test_results['paired_models'])}")
        lines.append("")
        lines.append("  Paired t-test (NLSY97 vs COMPAS):")
        lines.append("")
        lines.append(f"    Accuracy:")
        lines.append(f"      Mean difference (NLSY97 - COMPAS): "
                      f"{test_results['acc_mean_diff']:+.4f}")
        lines.append(f"      t-statistic: {test_results['acc_t_stat']:.4f}")
        lines.append(f"      p-value:     {test_results['acc_p_value']:.4f}"
                      f"  {'***' if test_results['acc_p_value'] < 0.001 else '**' if test_results['acc_p_value'] < 0.01 else '*' if test_results['acc_p_value'] < 0.05 else 'n.s.'}")
        lines.append("")
        lines.append(f"    F1 Score:")
        lines.append(f"      Mean difference (NLSY97 - COMPAS): "
                      f"{test_results['f1_mean_diff']:+.4f}")
        lines.append(f"      t-statistic: {test_results['f1_t_stat']:.4f}")
        lines.append(f"      p-value:     {test_results['f1_p_value']:.4f}"
                      f"  {'***' if test_results['f1_p_value'] < 0.001 else '**' if test_results['f1_p_value'] < 0.01 else '*' if test_results['f1_p_value'] < 0.05 else 'n.s.'}")
        lines.append("")
        lines.append("  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")

    lines.append("")

    # ----- Section 4: LaTeX table -----
    lines.append("")
    lines.append("4. LATEX TABLE (for paper)")
    lines.append("=" * W)
    lines.append("")
    lines.append(format_latex_table(comp_df))

    lines.append("")
    lines.append("=" * W)
    lines.append("END OF REPORT")
    lines.append("=" * W)

    return "\n".join(lines)


def format_latex_table(comp_df: pd.DataFrame) -> str:
    """Format a LaTeX table for the cross-dataset comparison."""
    # Only include models in both datasets for the main paper table
    paired = comp_df[comp_df["delta_acc"].notna()].copy()
    paired = paired.sort_values("nlsy97_acc", ascending=False)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-dataset comparison of AgenticSimLaw performance "
                  "(NLSY97 vs COMPAS). $\\Delta$ = COMPAS $-$ NLSY97.}")
    lines.append("\\label{tab:cross-dataset}")
    lines.append("\\begin{tabular}{l rr rr rr}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{2}{c}{NLSY97} & \\multicolumn{2}{c}{COMPAS} "
                  "& \\multicolumn{2}{c}{$\\Delta$} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    lines.append("Model & Acc & F1 & Acc & F1 & $\\Delta$Acc & $\\Delta$F1 \\\\")
    lines.append("\\midrule")

    for _, row in paired.iterrows():
        model = row["model"].replace("_", "\\_")
        if len(model) > 40:
            model = model[:37] + "..."

        delta_acc_str = f"{row['delta_acc']:+.3f}"
        delta_f1_str = f"{row['delta_f1']:+.3f}"

        lines.append(
            f"{model} & {row['nlsy97_acc']:.3f} & {row['nlsy97_f1']:.3f} "
            f"& {row['compas_acc']:.3f} & {row['compas_f1']:.3f} "
            f"& {delta_acc_str} & {delta_f1_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Project root: {PROJECT_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover experiment directories for each dataset
    nlsy97_dirs = discover_experiment_dirs("nlsy97")
    compas_dirs = discover_experiment_dirs("compas")

    print(f"Found {len(nlsy97_dirs)} NLSY97 experiment directories:")
    for d in nlsy97_dirs:
        print(f"  {d.name}")

    print(f"Found {len(compas_dirs)} COMPAS experiment directories:")
    for d in compas_dirs:
        print(f"  {d.name}")

    if not nlsy97_dirs and not compas_dirs:
        print("ERROR: No transcript directories found for either dataset.")
        sys.exit(1)

    if not nlsy97_dirs:
        print("WARNING: No NLSY97 directories found. Cross-dataset comparison "
              "will be limited.")
    if not compas_dirs:
        print("WARNING: No COMPAS directories found. Cross-dataset comparison "
              "will be limited.")

    print()

    # Process all experiments
    all_records = []

    for exp_dir in nlsy97_dirs:
        print(f"Processing {exp_dir.name} ...")
        records = process_experiment(exp_dir, "nlsy97")
        print(f"  {len(records)} debates loaded")
        all_records.extend(records)

    for exp_dir in compas_dirs:
        print(f"Processing {exp_dir.name} ...")
        records = process_experiment(exp_dir, "compas")
        print(f"  {len(records)} debates loaded")
        all_records.extend(records)

    if not all_records:
        print("ERROR: No debate records found.")
        sys.exit(1)

    print(f"\nTotal records: {len(all_records)}")

    # Aggregate per-(dataset, model) stats
    stats_df = aggregate_model_stats(all_records)
    print(f"Computed statistics for {len(stats_df)} (dataset, model) combinations")

    n_nlsy97 = len(stats_df[stats_df["dataset"] == "nlsy97"])
    n_compas = len(stats_df[stats_df["dataset"] == "compas"])
    print(f"  NLSY97: {n_nlsy97} models")
    print(f"  COMPAS: {n_compas} models")

    # Build comparison table
    comp_df = build_comparison_table(stats_df)
    n_paired = comp_df["delta_acc"].notna().sum()
    print(f"  Paired (both datasets): {n_paired} models")
    print()

    # Aggregate statistics
    agg = compute_aggregate_stats(stats_df)

    # Paired statistical tests
    test_results = run_paired_tests(stats_df)

    # Format text report
    report_text = format_comparison_text(comp_df, agg, test_results)

    # Write text output
    txt_path = RESULTS_DIR / "dataset_comparison_table.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"Wrote {txt_path}")

    # Write CSV output (one row per model, with both-dataset columns)
    csv_path = RESULTS_DIR / "dataset_comparison_table.csv"
    comp_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")

    # Print report to stdout
    print()
    print(report_text)


if __name__ == "__main__":
    main()
