#!/bin/bash
# Reproduce all derived statistics and tables for the TMLR 2026 paper.
# Run from project root: bash scripts/reproduce_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "AgenticSimLaw — Reproducibility Script"
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo "============================================================"
echo ""

mkdir -p results

echo "[1/4] Computing experiment costs, timing, and token estimates..."
python scripts/compute_experiment_costs.py
echo "  -> results/experiment_costs_report.txt"
echo ""

echo "[2/4] Computing accuracy, F1, precision, recall per model..."
python scripts/compute_accuracy_f1_table.py
echo "  -> results/accuracy_f1_table.txt"
echo "  -> results/accuracy_f1_table.csv"
echo ""

echo "[3/4] Computing judge opinion evolution and persuasion analysis..."
python scripts/compute_judge_opinion_evolution.py
echo "  -> results/judge_opinion_evolution.txt"
echo "  -> results/judge_opinion_evolution.csv"
echo "  -> results/judge_flip_stats.csv"
echo ""

echo "[4/4] Computing cross-dataset comparison (NLSY97 vs COMPAS)..."
python scripts/compute_dataset_comparison.py
echo "  -> results/dataset_comparison_table.txt"
echo "  -> results/dataset_comparison_table.csv"
echo ""

echo "============================================================"
echo "All reproducibility scripts complete."
echo "Results in: $PROJECT_DIR/results/"
echo "============================================================"
ls -la results/
