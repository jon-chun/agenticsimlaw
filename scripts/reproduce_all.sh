#!/bin/bash
# Reproduce all derived statistics and tables for the TMLR 2026 paper.
# Run from project root: bash scripts/reproduce_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MADCal — Reproducibility Script"
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo "============================================================"
echo ""

mkdir -p results
mkdir -p figures

echo "[1/7] Computing experiment costs, timing, and token estimates..."
python scripts/compute_experiment_costs.py
echo "  -> results/experiment_costs_report.txt"
echo ""

echo "[2/7] Computing accuracy, F1, precision, recall per model..."
python scripts/compute_accuracy_f1_table.py
echo "  -> results/accuracy_f1_table.txt"
echo "  -> results/accuracy_f1_table.csv"
echo ""

echo "[3/7] Computing judge opinion evolution and persuasion analysis..."
python scripts/compute_judge_opinion_evolution.py
echo "  -> results/judge_opinion_evolution.txt"
echo "  -> results/judge_opinion_evolution.csv"
echo "  -> results/judge_flip_stats.csv"
echo ""

echo "[4/7] Computing cross-dataset comparison (NLSY97 vs COMPAS)..."
python scripts/compute_dataset_comparison.py
echo "  -> results/dataset_comparison_table.txt"
echo "  -> results/dataset_comparison_table.csv"
echo ""

echo "[5/7] Computing sign test (debate vs SC calibration)..."
python scripts/compute_sign_test.py
echo "  -> Sign test results printed above"
echo ""

echo "[6/7] Comparing adversarial vs cooperative debate..."
python scripts/compare_adversarial_vs_cooperative.py
echo "  -> Cooperative comparison results printed above"
echo ""

echo "[7/7] Generating paper figures..."
python scripts/generate_all_figures.py
echo "  -> figures/*.pdf"
echo ""

echo "============================================================"
echo "All reproducibility scripts complete."
echo "Results in: $PROJECT_DIR/results/"
echo "Figures in: $PROJECT_DIR/figures/"
echo "============================================================"
echo ""
echo "Results:"
ls -la results/
echo ""
echo "Figures:"
ls -la figures/
