#!/usr/bin/env python3
"""
StandardLLM Evaluation Script

Implements three prompting strategies from the AgenticSimLaw paper (Appendix A):
1. Zero-shot (system1): Direct prediction request
2. Chain of Thought (CoT): Step-by-step reasoning before prediction
3. N-shot CoT: 30 labeled examples + reasoning + prediction

Paper Reference: Section 3.3 "Reasoning Methodologies"
- Temperature: 0.0 for deterministic output
- Cases: 150 per model/prompt combination
- Models: 16 models in small ensemble (7-14b parameters)
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import re

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import yaml

# Try to import ollama, provide helpful error if not installed
try:
    import ollama
except ImportError:
    ollama = None

try:
    import litellm
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    litellm = None

# ============================================================================
# Configuration
# ============================================================================

# Default paths
DEFAULT_VIGNETTES_PATH = "../data/sample_vignettes.csv"
DEFAULT_OUTPUT_DIR = "../results/standardllm"
DEFAULT_CONFIG_PATH = "../configs/config_ollama_models_size.yaml"

# Experiment parameters (from paper Section 3.2)
DEFAULT_TEMPERATURE = 0.0  # Deterministic for StandardLLM
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CASE_COUNT = 150   # Per model/prompt combo (paper: 150)
DEFAULT_NSHOT_COUNT = 30   # Examples for n-shot prompting
RANDOM_SEED = 42

# Model ensembles (from paper Table 1)
SMALL_ENSEMBLE_MODELS = [
    "aya-expanse:8b-q4_K_M",
    "deepseek-r1:7b-q4_K_M",
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "marco-o1:7b-q4_K_M",
    "mistral:7b-instruct-v0.2-q4_K_M",
    "olmo2:7b-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "tulu3:8b-q4_K_M",
]

# Prompt types
PROMPT_TYPES = ["system1", "cot", "cot-nshot"]

# ============================================================================
# Prompt Templates (from paper Appendix A)
# ============================================================================

PROMPT_PERSONA = """You are a highly experienced criminal justice risk assessment expert employing advanced statistical and judicial knowledge."""

PROMPT_SYSTEM1 = """You are a risk assessment expert. Analyze the following risk factors to determine if they indicate high risk for {recidivism_desc}.

You must respond ONLY with a JSON object in exactly this format, with no additional text:

{{
  "prediction": "YES" or "NO",
  "confidence": number between 0 and 100
}}

Risk Factors to analyze:
{risk_factors}"""

PROMPT_COT = """You are a criminal risk assessment expert. Step back, take a deep breath and carefully think step by step to assign a relative weight [low, medium, high] to each risk factor before using them to collectively predict {recidivism_desc}:

{risk_factors}

After your analysis, you must respond ONLY with a JSON object in exactly this format, with no additional text:

{{{{
  "risk_factors": [
    ["factor1", "low"|"medium"|"high", "textual reasoning"],
    ["factor2", "low"|"medium"|"high", "textual reasoning"]
  ],
  "prediction": "YES" or "NO",
  "confidence": number between 0 and 100
}}}}"""

PROMPT_COT_NSHOT = """You are a criminal risk assessment expert. First, analyze the following example cases of risk factors and resulting {recidivism_desc} outcome:

{nshot_examples}

Next, step back and take a deep breath and carefully think step by step to assign a relative weight [low, medium, high] to each risk factor before using them to collectively predict {recidivism_desc} for this case:

{risk_factors}

Based on the above n-shot examples and your weighted risk factor analysis for this particular case, predict the {recidivism_desc} outcome for this case.

Respond ONLY with a JSON object in exactly this format, with no additional text:

{{{{
  "risk_factors": [
    ["factor1", "low"|"medium"|"high", "textual reasoning"],
    ["factor2", "low"|"medium"|"high", "textual reasoning"]
  ],
  "prediction": "YES" or "NO",
  "confidence": number between 0 and 100
}}}}"""""


# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIGS = {
    'nlsy97': {
        'vignettes_path': '../data/sample_vignettes.csv',
        'target_col': 'y_arrestedafter2002',
        'exclude_cols': ['id', 'y_arrestedafter2002'],
        'recidivism_desc': '3-year rearrest recidivism',
        'outcome_yes': 'YES (rearrested)',
        'outcome_no': 'NO (not rearrested)',
    },
    'compas': {
        'vignettes_path': '../data/compas_vignettes.csv',
        'target_col': 'two_year_recid',
        'exclude_cols': ['id', 'two_year_recid'],
        'recidivism_desc': '2-year recidivism',
        'outcome_yes': 'YES (recidivated)',
        'outcome_no': 'NO (did not recidivate)',
    },
    'compas_nodecile': {
        'vignettes_path': '../data/compas_vignettes.csv',
        'target_col': 'two_year_recid',
        'exclude_cols': ['id', 'two_year_recid', 'decile_score'],
        'recidivism_desc': '2-year recidivism',
        'outcome_yes': 'YES (recidivated)',
        'outcome_no': 'NO (did not recidivate)',
    },
}


# ============================================================================
# Data Models
# ============================================================================

class PredictionResult(BaseModel):
    """Structured prediction result from LLM."""
    prediction: str = Field(description="YES or NO")
    confidence: int = Field(ge=0, le=100, description="Confidence 0-100")
    risk_factors: Optional[List[Any]] = Field(default=None, description="Factor analysis for CoT")


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    model_name: str
    prompt_type: str
    case_id: int
    repeat_id: int
    dataset: str
    ground_truth: bool
    prediction: Optional[str]
    confidence: Optional[int]
    is_correct: Optional[bool]
    response_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: str
    error: Optional[str]
    timestamp: str


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / f"standardllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_vignettes(path: str, target_col: str) -> pd.DataFrame:
    """Load vignettes CSV file."""
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Missing required column: {target_col}")

    return df


def row_to_risk_factors(row: pd.Series, exclude_cols: List[str] = None) -> str:
    """Convert a DataFrame row to natural language risk factors string."""
    if exclude_cols is None:
        exclude_cols = ['id', 'y_arrestedafter2002']

    factors = []
    for col in row.index:
        if col not in exclude_cols:
            value = row[col]
            readable_col = col.replace('_', ' ')
            factors.append(f"{readable_col} is {value}")

    return "; ".join(factors)


def create_nshot_examples(df: pd.DataFrame, ds_cfg: dict, n: int = 30, exclude_idx: int = -1) -> str:
    """Create n-shot examples string from dataframe."""
    available = df[df.index != exclude_idx] if exclude_idx >= 0 else df

    n_available = min(n, len(available))
    samples = available.sample(n=n_available, random_state=RANDOM_SEED)

    target_col = ds_cfg['target_col']
    exclude_cols = ds_cfg['exclude_cols']

    examples = []
    for _, row in samples.iterrows():
        factors = row_to_risk_factors(row, exclude_cols)
        outcome = ds_cfg['outcome_yes'] if row[target_col] else ds_cfg['outcome_no']
        examples.append(f"Case: {factors}\nOutcome: {outcome}")

    return "\n\n".join(examples)


def parse_prediction(response_text: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse prediction and confidence from LLM response."""
    # Try JSON parsing first
    try:
        # Find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            prediction = data.get('prediction', '').upper()
            confidence = int(data.get('confidence', 50))

            if prediction in ['YES', 'NO']:
                return prediction, min(max(confidence, 0), 100)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex parsing
    pred_match = re.search(r'"prediction"\s*:\s*"(YES|NO)"', response_text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*(\d+)', response_text)

    prediction = pred_match.group(1).upper() if pred_match else None
    confidence = int(conf_match.group(1)) if conf_match else None

    if confidence is not None:
        confidence = min(max(confidence, 0), 100)

    return prediction, confidence


def clean_model_name(model_name: str) -> str:
    """Clean model name for file/directory naming."""
    return model_name.replace(':', '_').replace('/', '_').replace('.', '_')


# ============================================================================
# LLM Evaluation
# ============================================================================

def call_ollama(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[str, float, Dict[str, int]]:
    """
    Call Ollama API with given prompt.

    Returns:
        Tuple of (response_text, response_time_sec, token_counts)
    """
    start_time = time.time()

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': temperature,
                'num_predict': max_tokens,
            }
        )

        elapsed = time.time() - start_time

        token_counts = {
            'prompt_tokens': response.get('prompt_eval_count', 0),
            'completion_tokens': response.get('eval_count', 0),
            'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
        }

        return response.get('response', ''), elapsed, token_counts

    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f"Ollama API error: {str(e)}") from e


def call_litellm(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[str, float, Dict[str, int]]:
    """Call LiteLLM API (OpenAI, Google, xAI, etc.) with given prompt."""
    if litellm is None:
        raise RuntimeError("litellm package not installed. Run: pip install litellm")

    start_time = time.time()
    try:
        if model in THINKING_MODELS:
            # Reasoning models reject temperature and max_tokens;
            # use max_completion_tokens (covers reasoning + output)
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens * THINKING_MODEL_TOKEN_MULTIPLIER,
            )
        else:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content or ""
        usage = response.usage
        token_counts = {
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        }
        return content, elapsed, token_counts

    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f"LiteLLM API error: {str(e)}") from e


COMMERCIAL_PREFIXES = ('gpt-', 'o1-', 'o3-', 'o4-', 'claude-', 'chatgpt-')

# Reasoning/thinking models that reject temperature and max_tokens params
THINKING_MODELS = frozenset({
    "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-nano",
    "gpt-5-pro", "gpt-5.1-pro", "gpt-5.2-pro",
    "o3", "o3-pro", "o4-mini", "o1", "o1-pro",
})
THINKING_MODEL_TOKEN_MULTIPLIER = 16


def call_llm(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[str, float, Dict[str, int]]:
    """Route to Ollama or LiteLLM based on model name."""
    if '/' in model or model.startswith(COMMERCIAL_PREFIXES):
        return call_litellm(model, prompt, temperature, max_tokens)
    else:
        if ollama is None:
            raise RuntimeError("ollama package not installed. Run: pip install ollama")
        return call_ollama(model, prompt, temperature, max_tokens)


def evaluate_single_case(
    model: str,
    prompt_type: str,
    row: pd.Series,
    case_id: int,
    repeat_id: int = 0,
    ds_cfg: dict = None,
    temperature: float = DEFAULT_TEMPERATURE,
    nshot_examples: str = "",
    logger: Optional[logging.Logger] = None
) -> EvaluationResult:
    """Evaluate a single case with given model and prompt type."""
    if ds_cfg is None:
        ds_cfg = DATASET_CONFIGS['nlsy97']

    timestamp = datetime.now().isoformat()
    risk_factors = row_to_risk_factors(row, ds_cfg['exclude_cols'])
    ground_truth = bool(row[ds_cfg['target_col']])
    recidivism_desc = ds_cfg['recidivism_desc']

    # Build prompt based on type
    if prompt_type == "system1":
        prompt = PROMPT_PERSONA + "\n\n" + PROMPT_SYSTEM1.format(
            risk_factors=risk_factors, recidivism_desc=recidivism_desc)
    elif prompt_type == "cot":
        prompt = PROMPT_PERSONA + "\n\n" + PROMPT_COT.format(
            risk_factors=risk_factors, recidivism_desc=recidivism_desc)
    elif prompt_type == "cot-nshot":
        prompt = PROMPT_PERSONA + "\n\n" + PROMPT_COT_NSHOT.format(
            nshot_examples=nshot_examples,
            risk_factors=risk_factors,
            recidivism_desc=recidivism_desc)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    # Call LLM
    try:
        response_text, response_time, token_counts = call_llm(model, prompt, temperature)
        prediction, confidence = parse_prediction(response_text)

        is_correct = None
        if prediction:
            predicted_positive = (prediction == "YES")
            is_correct = (predicted_positive == ground_truth)

        return EvaluationResult(
            model_name=model,
            prompt_type=prompt_type,
            case_id=case_id,
            repeat_id=repeat_id,
            dataset=ds_cfg.get('target_col', 'unknown'),
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
            is_correct=is_correct,
            response_time_sec=response_time,
            prompt_tokens=token_counts['prompt_tokens'],
            completion_tokens=token_counts['completion_tokens'],
            total_tokens=token_counts['total_tokens'],
            raw_response=response_text[:1000],
            error=None,
            timestamp=timestamp
        )

    except Exception as e:
        if logger:
            logger.error(f"Error evaluating case {case_id} rep {repeat_id} with {model}/{prompt_type}: {e}")

        return EvaluationResult(
            model_name=model,
            prompt_type=prompt_type,
            case_id=case_id,
            repeat_id=repeat_id,
            dataset=ds_cfg.get('target_col', 'unknown'),
            ground_truth=ground_truth,
            prediction=None,
            confidence=None,
            is_correct=None,
            response_time_sec=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            raw_response="",
            error=str(e),
            timestamp=timestamp
        )


def run_evaluation(
    models: List[str],
    prompt_types: List[str],
    df: pd.DataFrame,
    case_count: int,
    repeats: int,
    ds_cfg: dict,
    temperature: float,
    output_dir: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Run full evaluation across models, prompt types, and repeats.
    """
    results = []

    # Select cases (use seed for reproducibility)
    np.random.seed(RANDOM_SEED)
    available_cases = min(case_count, len(df))
    case_indices = np.random.choice(len(df), size=available_cases, replace=False)

    nshot_df = df.copy()
    nshot_examples = create_nshot_examples(nshot_df, ds_cfg, n=DEFAULT_NSHOT_COUNT)

    total_evals = len(models) * len(prompt_types) * available_cases * repeats
    current_eval = 0

    for model in models:
        logger.info(f"Starting evaluation with model: {model}")

        # Check model availability — only for Ollama models (skip for commercial APIs)
        if '/' not in model and not model.startswith(COMMERCIAL_PREFIXES):
            try:
                ollama.show(model)
            except Exception as e:
                logger.warning(f"Model {model} not available: {e}. Skipping.")
                continue

        for prompt_type in prompt_types:
            logger.info(f"  Prompt type: {prompt_type}")

            for rep in range(repeats):
                if repeats > 1:
                    logger.info(f"    Repeat {rep + 1}/{repeats}")

                for i, case_idx in enumerate(case_indices):
                    current_eval += 1
                    row = df.iloc[case_idx]

                    if current_eval % 10 == 0:
                        logger.info(f"  Progress: {current_eval}/{total_evals} ({100*current_eval/total_evals:.1f}%)")

                    result = evaluate_single_case(
                        model=model,
                        prompt_type=prompt_type,
                        row=row,
                        case_id=int(case_idx),
                        repeat_id=rep,
                        ds_cfg=ds_cfg,
                        temperature=temperature,
                        nshot_examples=nshot_examples if prompt_type == "cot-nshot" else "",
                        logger=logger
                    )

                    results.append(asdict(result))

                # Save intermediate results after each prompt/repeat combo
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_dir / "results_intermediate.csv", index=False)

    return pd.DataFrame(results)


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per model/prompt combination."""
    metrics = []

    if results_df.empty:
        return pd.DataFrame(metrics)

    for (model, prompt_type), group in results_df.groupby(['model_name', 'prompt_type']):
        valid = group[group['prediction'].notna()]

        if len(valid) == 0:
            continue

        # Confusion matrix components
        tp = ((valid['prediction'] == 'YES') & (valid['ground_truth'] == True)).sum()
        tn = ((valid['prediction'] == 'NO') & (valid['ground_truth'] == False)).sum()
        fp = ((valid['prediction'] == 'YES') & (valid['ground_truth'] == False)).sum()
        fn = ((valid['prediction'] == 'NO') & (valid['ground_truth'] == True)).sum()

        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            'model_name': model,
            'prompt_type': prompt_type,
            'total_cases': len(group),
            'valid_responses': len(valid),
            'success_rate': len(valid) / len(group),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'avg_confidence': valid['confidence'].mean(),
            'avg_response_time': valid['response_time_sec'].mean(),
            'avg_tokens': valid['total_tokens'].mean(),
        })

    return pd.DataFrame(metrics)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="StandardLLM Evaluation for AgenticSimLaw")
    parser.add_argument('--dataset', type=str, default='nlsy97',
                       choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset to evaluate (default: nlsy97)')
    parser.add_argument('--vignettes', type=str, default=None,
                       help='Override path to vignettes CSV file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to evaluate (default: small ensemble)')
    parser.add_argument('--prompts', type=str, nargs='+', default=PROMPT_TYPES,
                       choices=PROMPT_TYPES, help='Prompt types to use')
    parser.add_argument('--cases', type=int, default=DEFAULT_CASE_COUNT,
                       help='Number of cases per model/prompt combo')
    parser.add_argument('--repeats', type=int, default=1,
                       help='Number of repetitions per case (default: 1)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Override temperature (default: 0.0 for StandardLLM)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without running')

    args = parser.parse_args()

    # Dataset configuration
    ds_cfg = DATASET_CONFIGS[args.dataset]
    temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE

    # Setup paths
    script_dir = Path(__file__).parent
    if args.vignettes:
        vignettes_path = Path(args.vignettes)
    else:
        vignettes_path = Path(ds_cfg['vignettes_path'])
    if not vignettes_path.is_absolute():
        vignettes_path = script_dir / vignettes_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("StandardLLM Evaluation")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading vignettes from: {vignettes_path}")
    df = load_vignettes(str(vignettes_path), ds_cfg['target_col'])
    logger.info(f"Loaded {len(df)} cases")

    # Determine models
    models = args.models if args.models else SMALL_ENSEMBLE_MODELS

    # Print configuration
    logger.info(f"Dataset: {args.dataset} ({ds_cfg['recidivism_desc']})")
    logger.info(f"Models: {len(models)} — {models}")
    logger.info(f"Prompt types: {args.prompts}")
    logger.info(f"Cases per combo: {args.cases}")
    logger.info(f"Repeats: {args.repeats}")
    logger.info(f"Temperature: {temperature}")
    total_evals = len(models) * len(args.prompts) * min(args.cases, len(df)) * args.repeats
    logger.info(f"Total evaluations: {total_evals}")
    logger.info(f"Output directory: {output_dir}")

    if args.dry_run:
        logger.info("Dry run - exiting without evaluation")
        # Print a sample prompt
        if len(df) > 0:
            sample_row = df.iloc[0]
            factors = row_to_risk_factors(sample_row, ds_cfg['exclude_cols'])
            sample_prompt = PROMPT_PERSONA + "\n\n" + PROMPT_SYSTEM1.format(
                risk_factors=factors, recidivism_desc=ds_cfg['recidivism_desc'])
            logger.info(f"\nSample system1 prompt:\n{sample_prompt}")
        return

    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = time.time()

    results_df = run_evaluation(
        models=models,
        prompt_types=args.prompts,
        df=df,
        case_count=args.cases,
        repeats=args.repeats,
        ds_cfg=ds_cfg,
        temperature=temperature,
        output_dir=output_dir,
        logger=logger
    )

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed:.1f} seconds")

    # Save detailed results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"standardllm_results_{args.dataset}_{ts}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved detailed results to: {results_path}")

    # Compute and save metrics
    metrics_df = compute_metrics(results_df)
    metrics_path = output_dir / f"standardllm_metrics_{args.dataset}_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    for prompt_type in args.prompts:
        subset = metrics_df[metrics_df['prompt_type'] == prompt_type]
        if len(subset) > 0:
            logger.info(f"\n{prompt_type.upper()}:")
            logger.info(f"  Mean Accuracy: {subset['accuracy'].mean():.3f}")
            logger.info(f"  Mean F1 Score: {subset['f1_score'].mean():.3f}")
            logger.info(f"  Mean Response Time: {subset['avg_response_time'].mean():.2f}s")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
