#!/usr/bin/env python3
"""
Analyze how judge opinions evolve across the 7-turn debate protocol in
AgenticSimLaw transcript JSON files.

The 7-turn protocol has 6 silent judge opinion snapshots (after each
Prosecutor or Defense turn, before the final ruling at turn 7):
    Turn 1: After Prosecutor opening
    Turn 2: After Defense rebuttal
    Turn 3: After Prosecutor cross-exam
    Turn 4: After Defense responds
    Turn 5: After Prosecutor closing
    Turn 6: After Defense closing
    (Turn 7 = Judge's final ruling, stored in final_ruling)

For each model this script computes:
    - Mean confidence at each turn position (turns 1-6 + final ruling)
    - Prediction flip rate: how often the judge changes YES/NO between
      consecutive turns
    - Final vs initial agreement rate
    - Persuasion effect: how often the judge changes prediction between
      turn 1 and the final ruling (turn 7)

Outputs:
    results/judge_opinion_evolution.txt  -- human-readable + LaTeX tables
    results/judge_opinion_evolution.csv  -- per-model-turn rows

Usage:
    python scripts/compute_judge_opinion_evolution.py
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

# Turn labels for the 6 silent opinions + final ruling
TURN_LABELS = [
    "T1_Pros_Open",
    "T2_Def_Rebut",
    "T3_Pros_Cross",
    "T4_Def_Resp",
    "T5_Pros_Close",
    "T6_Def_Close",
    "T7_Final_Ruling",
]

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
    """Extract dataset name from the directory name."""
    name = exp_dir.name.lower()
    for candidate in ("nlsy97", "compas"):
        if candidate in name:
            return candidate
    return "unknown"


def normalize_prediction(pred) -> str:
    """Normalize prediction to YES / NO / NO_DECISION."""
    if not isinstance(pred, str):
        return "NO_DECISION"
    p = pred.strip().upper()
    if p == "YES":
        return "YES"
    if p == "NO":
        return "NO"
    return "NO_DECISION"


# ---------------------------------------------------------------------------
# Per-debate extraction
# ---------------------------------------------------------------------------
def extract_evolution(data: dict) -> dict | None:
    """Extract the judge opinion evolution from a single transcript.

    Returns a dict with:
        turns: list of dicts with keys (turn_idx, turn_label, prediction, confidence)
               covering turns 1-6 from judge_opinion_evolution plus turn 7 from
               final_ruling.
        flips: number of prediction flips between consecutive turns
        initial_pred: prediction after turn 1
        final_pred: prediction from final ruling (turn 7)
        initial_final_agree: bool, whether turn 1 == turn 7 prediction
    """
    evol = data.get("judge_opinion_evolution", [])
    final_ruling = data.get("final_ruling", {})

    if not evol:
        return None

    turns = []
    for i, entry in enumerate(evol):
        pred = normalize_prediction(entry.get("prediction"))
        conf = entry.get("confidence")
        try:
            conf = float(conf) if conf is not None else np.nan
        except (ValueError, TypeError):
            conf = np.nan
        turn_idx = i + 1  # 1-based
        label = TURN_LABELS[i] if i < len(TURN_LABELS) else f"T{turn_idx}"
        turns.append({
            "turn_idx": turn_idx,
            "turn_label": label,
            "prediction": pred,
            "confidence": conf,
        })

    # Add turn 7 = final ruling
    final_pred = normalize_prediction(final_ruling.get("prediction"))
    final_conf = final_ruling.get("confidence")
    try:
        final_conf = float(final_conf) if final_conf is not None else np.nan
    except (ValueError, TypeError):
        final_conf = np.nan
    turns.append({
        "turn_idx": 7,
        "turn_label": "T7_Final_Ruling",
        "prediction": final_pred,
        "confidence": final_conf,
    })

    # Compute flips (consecutive prediction changes among YES/NO only)
    valid_preds = [(t["turn_idx"], t["prediction"])
                   for t in turns if t["prediction"] in ("YES", "NO")]
    flips = 0
    for j in range(1, len(valid_preds)):
        if valid_preds[j][1] != valid_preds[j - 1][1]:
            flips += 1

    initial_pred = turns[0]["prediction"] if turns else "NO_DECISION"
    final_pred_val = turns[-1]["prediction"] if turns else "NO_DECISION"
    initial_final_agree = (initial_pred == final_pred_val)

    return {
        "turns": turns,
        "flips": flips,
        "total_valid_transitions": max(len(valid_preds) - 1, 0),
        "initial_pred": initial_pred,
        "final_pred": final_pred_val,
        "initial_final_agree": initial_final_agree,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_experiment(exp_dir: Path) -> list[dict]:
    """Process all transcript JSONs in an experiment directory.

    Returns a list of per-debate record dicts including turn-level info.
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

            evol = extract_evolution(data)
            if evol is None:
                error_count += 1
                continue

            # Parse case_id and repetition from filename
            fname = jf.stem
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
                "turns": evol["turns"],
                "flips": evol["flips"],
                "total_valid_transitions": evol["total_valid_transitions"],
                "initial_pred": evol["initial_pred"],
                "final_pred": evol["final_pred"],
                "initial_final_agree": evol["initial_final_agree"],
                "file": str(jf),
            })

    if error_count > 0:
        print(f"  Skipped {error_count} unreadable/empty files in {exp_dir.name}")
    return records


def build_turn_level_df(records: list[dict]) -> pd.DataFrame:
    """Expand records into a per-model per-turn DataFrame for aggregation."""
    rows = []
    for rec in records:
        for turn in rec["turns"]:
            rows.append({
                "experiment": rec["experiment"],
                "dataset": rec["dataset"],
                "model": rec["model"],
                "case_id": rec["case_id"],
                "repetition": rec["repetition"],
                "turn_idx": turn["turn_idx"],
                "turn_label": turn["turn_label"],
                "prediction": turn["prediction"],
                "confidence": turn["confidence"],
            })
    return pd.DataFrame(rows)


def aggregate_model_turn_stats(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate turn-level data into per-model per-turn statistics."""
    grouped = turn_df.groupby(
        ["experiment", "dataset", "model", "turn_idx", "turn_label"],
        sort=True,
    )

    rows = []
    for (experiment, dataset, model, turn_idx, turn_label), grp in grouped:
        confs = grp["confidence"].dropna()
        preds = grp["prediction"]
        n_yes = (preds == "YES").sum()
        n_no = (preds == "NO").sum()
        n_nodec = (preds == "NO_DECISION").sum()
        total = len(preds)

        rows.append({
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "turn_idx": turn_idx,
            "turn_label": turn_label,
            "mean_confidence": confs.mean() if len(confs) > 0 else np.nan,
            "std_confidence": confs.std() if len(confs) > 0 else np.nan,
            "median_confidence": confs.median() if len(confs) > 0 else np.nan,
            "yes_rate": n_yes / total if total > 0 else 0.0,
            "no_rate": n_no / total if total > 0 else 0.0,
            "no_decision_rate": n_nodec / total if total > 0 else 0.0,
            "n_debates": total,
        })

    return pd.DataFrame(rows)


def aggregate_flip_stats(records: list[dict]) -> pd.DataFrame:
    """Aggregate per-debate flip statistics into per-model summary."""
    groups = defaultdict(list)
    for r in records:
        key = (r["experiment"], r["dataset"], r["model"])
        groups[key].append(r)

    rows = []
    for (experiment, dataset, model), debates in sorted(groups.items()):
        n = len(debates)
        flips = [d["flips"] for d in debates]
        transitions = [d["total_valid_transitions"] for d in debates]
        agreements = [d["initial_final_agree"] for d in debates]

        # Persuasion effect: initial != final prediction
        persuasion_changes = sum(1 for d in debates if not d["initial_final_agree"])

        # Flip rate: total flips / total valid transitions
        total_flips = sum(flips)
        total_transitions = sum(transitions)
        flip_rate = total_flips / total_transitions if total_transitions > 0 else 0.0

        rows.append({
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "n_debates": n,
            "mean_flips_per_debate": np.mean(flips),
            "std_flips_per_debate": np.std(flips),
            "max_flips": max(flips) if flips else 0,
            "overall_flip_rate": flip_rate,
            "initial_final_agreement_rate": np.mean(agreements),
            "persuasion_change_count": persuasion_changes,
            "persuasion_change_rate": persuasion_changes / n if n > 0 else 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_evolution_table(turn_stats: pd.DataFrame, title: str) -> str:
    """Format mean confidence at each turn position per model."""
    lines = []
    lines.append("=" * 130)
    lines.append(title)
    lines.append("=" * 130)

    # Pivot: model vs turn_label, values = mean_confidence
    pivot = turn_stats.pivot_table(
        index="model", columns="turn_label", values="mean_confidence",
        aggfunc="mean",
    )
    # Reorder columns by turn index
    ordered_cols = [lbl for lbl in TURN_LABELS if lbl in pivot.columns]
    pivot = pivot[ordered_cols]

    # Header
    header = f"{'Model':<45}"
    for col in ordered_cols:
        header += f"  {col:<14}"
    lines.append(header)
    lines.append("-" * 130)

    for model in sorted(pivot.index):
        row_str = f"{model:<45}"
        for col in ordered_cols:
            val = pivot.loc[model, col]
            if pd.isna(val):
                row_str += f"  {'N/A':<14}"
            else:
                row_str += f"  {val:<14.1f}"
        lines.append(row_str)

    lines.append("")
    return "\n".join(lines)


def format_yes_rate_table(turn_stats: pd.DataFrame, title: str) -> str:
    """Format YES rate at each turn position per model."""
    lines = []
    lines.append("=" * 130)
    lines.append(title)
    lines.append("=" * 130)

    pivot = turn_stats.pivot_table(
        index="model", columns="turn_label", values="yes_rate",
        aggfunc="mean",
    )
    ordered_cols = [lbl for lbl in TURN_LABELS if lbl in pivot.columns]
    pivot = pivot[ordered_cols]

    header = f"{'Model':<45}"
    for col in ordered_cols:
        header += f"  {col:<14}"
    lines.append(header)
    lines.append("-" * 130)

    for model in sorted(pivot.index):
        row_str = f"{model:<45}"
        for col in ordered_cols:
            val = pivot.loc[model, col]
            if pd.isna(val):
                row_str += f"  {'N/A':<14}"
            else:
                row_str += f"  {val:<14.1%}"
        lines.append(row_str)

    lines.append("")
    return "\n".join(lines)


def format_flip_table(flip_stats: pd.DataFrame, title: str) -> str:
    """Format flip rate and persuasion effect statistics."""
    lines = []
    lines.append("=" * 130)
    lines.append(title)
    lines.append("=" * 130)

    cols = [
        ("Model", "model", "{}", 45),
        ("N", "n_debates", "{:>5d}", 5),
        ("MeanFlips", "mean_flips_per_debate", "{:.2f}", 10),
        ("StdFlips", "std_flips_per_debate", "{:.2f}", 10),
        ("MaxFlips", "max_flips", "{:>4d}", 8),
        ("FlipRate", "overall_flip_rate", "{:.3f}", 8),
        ("Init=Fin", "initial_final_agreement_rate", "{:.1%}", 8),
        ("Persuasion", "persuasion_change_rate", "{:.1%}", 10),
    ]

    header = "  ".join(f"{name:<{width}}" for name, _, _, width in cols)
    lines.append(header)
    lines.append("-" * 130)

    for _, row in flip_stats.iterrows():
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


def format_latex_evolution(turn_stats: pd.DataFrame, caption: str, label: str) -> str:
    """Format mean confidence evolution as a LaTeX table."""
    pivot = turn_stats.pivot_table(
        index="model", columns="turn_label", values="mean_confidence",
        aggfunc="mean",
    )
    ordered_cols = [lbl for lbl in TURN_LABELS if lbl in pivot.columns]
    pivot = pivot[ordered_cols]

    n_cols = len(ordered_cols)
    col_spec = "l " + " ".join(["r"] * n_cols)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Column headers
    short_labels = {
        "T1_Pros_Open": "T1",
        "T2_Def_Rebut": "T2",
        "T3_Pros_Cross": "T3",
        "T4_Def_Resp": "T4",
        "T5_Pros_Close": "T5",
        "T6_Def_Close": "T6",
        "T7_Final_Ruling": "T7",
    }
    header = "Model & " + " & ".join(short_labels.get(c, c) for c in ordered_cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for model in sorted(pivot.index):
        model_esc = model.replace("_", "\\_")
        vals = []
        for col in ordered_cols:
            v = pivot.loc[model, col]
            vals.append(f"{v:.1f}" if not pd.isna(v) else "---")
        lines.append(f"{model_esc} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def format_latex_flip(flip_stats: pd.DataFrame, caption: str, label: str) -> str:
    """Format flip statistics as a LaTeX table."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l r r r r r r}")
    lines.append("\\toprule")
    lines.append("Model & N & $\\mu$(Flips) & FlipRate & Init=Fin & Persuasion\\% \\\\")
    lines.append("\\midrule")

    for _, row in flip_stats.iterrows():
        model = row["model"].replace("_", "\\_")
        n = int(row["n_debates"])
        mean_f = f"{row['mean_flips_per_debate']:.2f}"
        frate = f"{row['overall_flip_rate']:.3f}"
        init_fin = f"{row['initial_final_agreement_rate']:.1%}"
        persuas = f"{row['persuasion_change_rate']:.1%}"
        lines.append(f"{model} & {n} & {mean_f} & {frate} & {init_fin} & {persuas} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # Build turn-level DataFrame
    turn_df = build_turn_level_df(all_records)
    print(f"Turn-level rows: {len(turn_df)}")

    # Aggregate stats
    turn_stats = aggregate_model_turn_stats(turn_df)
    flip_stats = aggregate_flip_stats(all_records)
    print(f"Turn stats rows: {len(turn_stats)}")
    print(f"Flip stats rows: {len(flip_stats)}\n")

    # Build output text
    output_lines = []
    output_lines.append("AgenticSimLaw - Judge Opinion Evolution Analysis")
    output_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    output_lines.append("Turn positions:")
    output_lines.append("  T1 = After Prosecutor opening")
    output_lines.append("  T2 = After Defense rebuttal")
    output_lines.append("  T3 = After Prosecutor cross-examination")
    output_lines.append("  T4 = After Defense response")
    output_lines.append("  T5 = After Prosecutor closing")
    output_lines.append("  T6 = After Defense closing")
    output_lines.append("  T7 = Judge's final ruling")
    output_lines.append("")

    # Per-experiment tables
    for exp_name in sorted(turn_stats["experiment"].unique()):
        exp_turn = turn_stats[turn_stats["experiment"] == exp_name]
        exp_flip = flip_stats[flip_stats["experiment"] == exp_name]
        dataset = exp_turn["dataset"].iloc[0] if len(exp_turn) > 0 else "?"

        output_lines.append(format_evolution_table(
            exp_turn,
            f"Mean Confidence per Turn -- {exp_name} (dataset: {dataset})"
        ))

        output_lines.append(format_yes_rate_table(
            exp_turn,
            f"YES Rate per Turn -- {exp_name} (dataset: {dataset})"
        ))

        output_lines.append(format_flip_table(
            exp_flip,
            f"Flip & Persuasion Statistics -- {exp_name} (dataset: {dataset})"
        ))

    # LaTeX tables
    output_lines.append("=" * 130)
    output_lines.append("LaTeX Tables")
    output_lines.append("=" * 130)
    output_lines.append("")

    for i, exp_name in enumerate(sorted(turn_stats["experiment"].unique())):
        exp_turn = turn_stats[turn_stats["experiment"] == exp_name]
        exp_flip = flip_stats[flip_stats["experiment"] == exp_name]
        dataset = exp_turn["dataset"].iloc[0] if len(exp_turn) > 0 else "?"

        output_lines.append(format_latex_evolution(
            exp_turn,
            f"Judge opinion confidence evolution -- {dataset} ({exp_name})",
            f"tab:evolution_{dataset}_{i}",
        ))

        output_lines.append(format_latex_flip(
            exp_flip,
            f"Judge flip and persuasion statistics -- {dataset} ({exp_name})",
            f"tab:flips_{dataset}_{i}",
        ))

    # Write text output
    txt_path = RESULTS_DIR / "judge_opinion_evolution.txt"
    full_text = "\n".join(output_lines)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(full_text)
    print(f"Wrote {txt_path}")

    # Write CSV output (turn-level stats)
    csv_path = RESULTS_DIR / "judge_opinion_evolution.csv"
    turn_stats.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")

    # Also write flip stats CSV
    flip_csv_path = RESULTS_DIR / "judge_flip_stats.csv"
    flip_stats.to_csv(flip_csv_path, index=False, float_format="%.4f")
    print(f"Wrote {flip_csv_path}")

    # Print summary to stdout
    print()
    print(full_text)


if __name__ == "__main__":
    main()
