# AgenticSimLaw: Multi-Agent Courtroom Debate for Recidivism Prediction

Replication package for the TMLR 2026 submission. This repository contains all code, data, experiment results, and reproducibility scripts needed to replicate the findings reported in the paper.

## Environment Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys should be set via a `.env` file or environment variables:
`OPENAI_API_KEY`, `FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`.

## Reproducing Paper Results

All derived statistics, tables, and figures can be reproduced from the raw experiment outputs included in `results/`:

```bash
bash scripts/reproduce_all.sh
```

Individual reproducibility scripts can also be run separately (see `scripts/README.md`).

## Running Experiments

### Multi-Agent Courtroom Debates (Step 1)

```bash
python src/step1_ai-debators_ver26.py \
    --dataset nlsy97 --ensemble commercial \
    --cases 100 --repeats 3 --concurrency 30
```

### Standard LLM Baselines

```bash
python src/standardllm_evaluation.py \
    --dataset nlsy97 --models "gpt-4o-mini" \
    --prompts system1 cot cot-nshot --cases 150
```

### Self-Consistency Majority Voting

```bash
python src/sc_majority_vote.py
```

### Full Pipeline

```bash
# Step 2: Aggregate debate transcripts into CSV
python src/step2_aggregate_transcripts_ver5_FREEZE.py

# Step 3: Compute statistical summaries
python src/step3_statistical_analysis_ver11_FREEZE.py

# Step 5: Merge standard LLM and agentic results
python src/step5_merge_standard-agenticsim_ver3_o1.py

# Steps 4/6: Generate plots
python src/step4_visualize_model_statistics_ver5_FREEZE.py
python src/step6_visualize_performance_comparison_ver6.py
```

## Directory Structure

```
agenticsimlaw/
├── configs/              # Model ensemble YAML configurations
├── data/                 # Datasets (vignettes CSVs)
│   ├── sample_vignettes.csv          # NLSY97 base sample
│   ├── nlsy97_full.csv               # NLSY97 full dataset
│   ├── nlsy97_vignettes_100.csv      # NLSY97 100-case expanded
│   ├── compas_vignettes.csv          # COMPAS base sample
│   ├── compas_full_filtered.csv      # COMPAS full filtered dataset
│   └── compas_vignettes_100.csv      # COMPAS 100-case expanded
├── results/              # Raw experiment outputs
│   ├── *.csv / *.txt                 # Aggregated debate results, statistics
│   ├── baselines_compas/             # PyCaret ML baseline results
│   └── standardllm/                  # Standard LLM prompting results
├── scripts/              # Reproducibility scripts for paper tables/figures
│   └── reproduce_all.sh              # Run all reproducibility scripts
├── src/                  # Source code
│   ├── llm_client.py                 # Dual-path LLM client (Ollama / LiteLLM)
│   ├── model_config.py               # Model ensemble management
│   ├── dataset_config.py             # Dataset configurations
│   ├── step1_ai-debators_ver26.py    # Multi-agent courtroom debate engine
│   ├── standardllm_evaluation.py     # Standard prompting baselines
│   ├── sc_majority_vote.py           # Self-consistency majority voting
│   └── step2-6 scripts               # Aggregation, statistics, visualization
└── requirements.txt
```

## Datasets

- **NLSY97**: 3-year rearrest prediction using 22 demographic/behavioral features from the National Longitudinal Survey of Youth 1997.
- **COMPAS**: 2-year recidivism prediction using 9 features from the ProPublica COMPAS dataset. A variant without the algorithmic `decile_score` is also evaluated.

## Key Design Decisions

- **Restartability**: Experiments skip completed work when resuming into an existing output directory.
- **Model routing**: Automatic detection of Ollama vs. commercial API based on model ID format.
- **JSON error recovery**: 3-layer fallback (stdlib JSON, `json_repair`, regex) for malformed LLM responses.
- **Parallelization**: Semaphore-bounded concurrency with fair distribution across API providers.
