# Reproducibility Scripts

All scripts in this directory compute derived statistics, metrics, and analyses
reported in the TMLR 2026 paper. Each script reads from the raw experiment
outputs and can be re-run independently.

## Scripts

| Script | Paper Section | What it Computes |
|--------|---------------|------------------|
| `compute_experiment_costs.py` | Appendix (Table A1) | API costs, token counts, wall-clock timing, debates/model |
| `compute_accuracy_f1_table.py` | Tables 5-6 | Per-model accuracy, F1, precision, recall for AgenticSimLaw |
| `compute_judge_opinion_evolution.py` | Section 5.2, Fig 3 | Judge confidence trajectories across debate turns |
| `compute_dataset_comparison.py` | Section 5.3 | Cross-dataset (NLSY97 vs COMPAS) performance comparison |
| `reproduce_all.sh` | -- | Runs all scripts in order, produces all derived tables/figures |

## Usage

```bash
cd /path/to/agenticsimlaw
source .venv/bin/activate

# Run individual script
python scripts/compute_experiment_costs.py

# Run all reproducibility scripts
bash scripts/reproduce_all.sh
```

## Dependencies

All scripts use only packages from `requirements.txt`. Token estimation
uses `tiktoken` (with fallback to character-based estimation if unavailable).

## Output

Results are written to `results/` directory with timestamped filenames
where applicable.
