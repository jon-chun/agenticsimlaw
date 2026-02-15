#!/usr/bin/env python3
"""
Compute cost, timing, and resource metrics for the AgenticSimLaw paper experiments.

Produces a comprehensive report covering commercial and OSS experiments across
NLSY97 and COMPAS datasets.

Usage:
    python src/compute_experiment_costs.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "experiment_costs_report.txt"
MONITOR_LOGS = [Path("/tmp/monitor_oss_3a.log"), Path("/tmp/monitor_oss_3b.log")]

# Pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    # Commercial models
    "gpt-4o-mini":       (0.15, 0.60),
    "gpt-4.1-mini":      (0.40, 1.60),
    "gemini/gemini-2.5-flash": (0.15, 0.60),
    "xai/grok-4-1-fast-non-reasoning-latest": (0.60, 4.00),
    # OSS via OpenRouter -- small models (7-14B)
    "openrouter/meta-llama/llama-3.1-8b-instruct": (0.065, 0.065),
    "openrouter/google/gemma-2-9b-it":             (0.065, 0.065),
    "openrouter/mistralai/mistral-7b-instruct":    (0.065, 0.065),
    "openrouter/qwen/qwen-2.5-7b-instruct":       (0.065, 0.065),
    "openrouter/cohere/aya-expanse-8b":            (0.065, 0.065),
    "openrouter/deepseek/deepseek-r1-distill-llama-8b": (0.065, 0.065),
    "openrouter/nousresearch/hermes-3-llama-3.1-8b": (0.065, 0.065),
    "openrouter/microsoft/phi-4":                  (0.14, 0.14),   # 14B
    # OSS -- medium models (24B+)
    "openrouter/cognitivecomputations/dolphin3.0-mistral-24b": (0.14, 0.14),
    # OSS -- large models (32B)
    "openrouter/allenai/olmo-2-0325-32b-instruct": (0.20, 0.20),
}

# Debate protocol: 7 speaker turns + ~7 silent judge opinions = ~13 API calls per debate
# (Prosecutor-opening, Defense-rebuttal, Prosecutor-cross, Defense-responds,
#  Prosecutor-closing, Defense-closing, Judge-verdict,
#  plus SilentJudge after each of ~6 speaker turns + initial)
EXPECTED_API_CALLS_PER_DEBATE = 13

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_tiktoken_encoding = None


def _get_encoding():
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        try:
            import tiktoken
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_encoding = "fallback"
    return _tiktoken_encoding


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc == "fallback":
        return max(1, len(text) // 4)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ModelStats:
    """Accumulates statistics for a single model within one experiment."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_api_duration_sec = 0.0
        self.debates_completed = 0  # from JSON file count
        self.parse_ok = 0
        self.parse_fail = 0

    @property
    def est_cost(self) -> float:
        price_in, price_out = self._get_pricing()
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000

    def _get_pricing(self) -> tuple:
        if self.model_name in MODEL_PRICING:
            return MODEL_PRICING[self.model_name]
        # Fuzzy match for model names that differ slightly
        for key, val in MODEL_PRICING.items():
            if key in self.model_name or self.model_name in key:
                return val
        # Default OSS pricing
        return (0.065, 0.065)

    @property
    def avg_response_sec(self) -> float:
        return self.total_api_duration_sec / self.api_calls if self.api_calls else 0.0


class ExperimentStats:
    """Accumulates statistics for one experiment directory."""

    def __init__(self, name: str, dataset: str, exp_type: str, dir_path: Path):
        self.name = name
        self.dataset = dataset
        self.exp_type = exp_type  # "commercial" or "oss"
        self.dir_path = dir_path
        self.models: dict[str, ModelStats] = {}
        self.wall_clock_sec: float | None = None
        self.avg_debates_per_min: float | None = None
        self.avg_sec_per_debate: float | None = None

    def get_or_create_model(self, model_name: str) -> ModelStats:
        if model_name not in self.models:
            self.models[model_name] = ModelStats(model_name)
        return self.models[model_name]

    @property
    def total_debates(self) -> int:
        return sum(m.debates_completed for m in self.models.values())

    @property
    def total_api_calls(self) -> int:
        return sum(m.api_calls for m in self.models.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.models.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.models.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(m.est_cost for m in self.models.values())

    @property
    def total_api_duration_sec(self) -> float:
        return sum(m.total_api_duration_sec for m in self.models.values())

    @property
    def model_count(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def classify_experiment(dirname: str) -> tuple:
    """Return (dataset, exp_type) from directory name."""
    dataset = "NLSY97" if "nlsy97" in dirname else "COMPAS" if "compas" in dirname else "unknown"
    return dataset


def count_debate_jsons(exp_dir: Path) -> dict:
    """Count JSON debate transcripts per model subdirectory."""
    counts = {}
    if not exp_dir.is_dir():
        return counts
    for subdir in sorted(exp_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "raw_api_responses":
            json_count = len(list(subdir.glob("*.json")))
            if json_count > 0:
                counts[subdir.name] = json_count
    return counts


def dir_name_to_model_id(dir_name: str) -> str:
    """Map a filesystem model directory name back to the API model ID.

    e.g. 'gpt_4o_mini' -> 'gpt-4o-mini'
         'openrouter_meta_llama_llama_3_1_8b_instruct' -> 'openrouter/meta-llama/llama-3.1-8b-instruct'
    """
    # Direct known mappings
    KNOWN = {
        "gpt_4o_mini": "gpt-4o-mini",
        "gpt_4_1_mini": "gpt-4.1-mini",
        "gemini_gemini_2_5_flash": "gemini/gemini-2.5-flash",
        "xai_grok_4_1_fast_non_reasoning_latest": "xai/grok-4-1-fast-non-reasoning-latest",
        "openrouter_meta_llama_llama_3_1_8b_instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "openrouter_google_gemma_2_9b_it": "openrouter/google/gemma-2-9b-it",
        "openrouter_mistralai_mistral_7b_instruct": "openrouter/mistralai/mistral-7b-instruct",
        "openrouter_qwen_qwen_2_5_7b_instruct": "openrouter/qwen/qwen-2.5-7b-instruct",
        "openrouter_microsoft_phi_4": "openrouter/microsoft/phi-4",
        "openrouter_allenai_olmo_2_0325_32b_instruct": "openrouter/allenai/olmo-2-0325-32b-instruct",
        "openrouter_cognitivecomputations_dolphin3_0_mistral_24b": "openrouter/cognitivecomputations/dolphin3.0-mistral-24b",
        "openrouter_cohere_aya_expanse_8b": "openrouter/cohere/aya-expanse-8b",
        "openrouter_deepseek_deepseek_r1_distill_llama_8b": "openrouter/deepseek/deepseek-r1-distill-llama-8b",
        "openrouter_nousresearch_hermes_3_llama_3_1_8b": "openrouter/nousresearch/hermes-3-llama-3.1-8b",
    }
    return KNOWN.get(dir_name, dir_name)


def parse_raw_api_responses(exp_dir: Path) -> dict:
    """Parse all JSONL raw API response files in an experiment directory.

    Returns dict mapping model_name -> ModelStats (without debate counts).
    """
    raw_dir = exp_dir / "raw_api_responses"
    stats: dict[str, ModelStats] = {}

    if not raw_dir.is_dir():
        return stats

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        line_count = 0
        error_count = 0
        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    error_count += 1
                    continue

                model = rec.get("model", "unknown")
                if model not in stats:
                    stats[model] = ModelStats(model)
                ms = stats[model]

                ms.api_calls += 1
                meta = rec.get("metadata", {})

                # Token counts -- use prompt_eval_count / eval_count from metadata
                prompt_tokens = meta.get("prompt_eval_count") or 0
                completion_tokens = meta.get("eval_count") or 0

                # If token counts are zero, estimate from raw_response text
                raw_resp = rec.get("raw_response", "")
                if completion_tokens == 0 and raw_resp:
                    completion_tokens = estimate_tokens(raw_resp)
                if prompt_tokens == 0:
                    # Rough estimate: prompt grows with each turn in a debate
                    # Average prompt ~800 tokens for first turn, growing to ~3000 for later turns
                    prompt_tokens = 800  # conservative default

                ms.input_tokens += prompt_tokens
                ms.output_tokens += completion_tokens

                # Timing
                duration = meta.get("python_api_duration_sec") or meta.get("total_duration_sec") or 0.0
                if duration:
                    ms.total_api_duration_sec += float(duration)

                # Parse status
                if rec.get("parse_status") == "ok":
                    ms.parse_ok += 1
                else:
                    ms.parse_fail += 1

                line_count += 1

        if error_count > 0:
            print(f"  Warning: {error_count} JSON decode errors in {jsonl_file.name}")

    return stats


def parse_monitor_logs() -> dict:
    """Parse monitor logs to extract timing information.

    Returns dict mapping experiment_label -> {wall_clock_sec, avg_debates_per_min, avg_sec_per_debate, final_done, final_total}
    """
    results = {}

    for log_path in MONITOR_LOGS:
        if not log_path.exists():
            print(f"  Monitor log not found: {log_path}")
            continue

        label = None
        last_time_min = 0
        last_done = 0
        last_total = 0
        last_sec_per_debate = 0
        last_debates_per_min = 0

        with open(log_path) as f:
            for line in f:
                line = line.strip()

                # Extract experiment label from header
                header_match = re.match(r"=== MONITOR:\s+(.+?)\s+===", line)
                if header_match:
                    label = header_match.group(1)
                    continue

                # Extract timing data from status lines
                # Format: [MM:SS] LABEL | Done: N/Total (X%) | Parse OK: N (Y%) | +Z/min | Ws/debate | ETA: Xm
                status_match = re.match(
                    r"\[(\d+):(\d+)\]\s+\S+\s+\|\s+Done:\s+(\d+)/(\d+)\s+\([\d.]+%\)\s+\|.*?\|\s+\+(\d+)/min\s+\|\s+([\d.]+)s/debate",
                    line
                )
                if status_match:
                    minutes = int(status_match.group(1))
                    seconds = int(status_match.group(2))
                    done = int(status_match.group(3))
                    total = int(status_match.group(4))
                    dpm = int(status_match.group(5))
                    spd = float(status_match.group(6))

                    last_time_min = minutes + seconds / 60.0
                    last_done = done
                    last_total = total
                    last_debates_per_min = dpm
                    last_sec_per_debate = spd

        if label and last_time_min > 0:
            results[label] = {
                "wall_clock_sec": last_time_min * 60,
                "avg_debates_per_min": last_debates_per_min,
                "avg_sec_per_debate": last_sec_per_debate,
                "final_done": last_done,
                "final_total": last_total,
                "log_file": str(log_path),
            }

    return results


# ---------------------------------------------------------------------------
# Experiment directory discovery and classification
# ---------------------------------------------------------------------------

def discover_experiments() -> list:
    """Find all transcripts_ver26_* directories and classify them."""
    experiments = []

    for item in sorted(PROJECT_DIR.iterdir()):
        if item.is_dir() and item.name.startswith("transcripts_ver26_"):
            dirname = item.name
            dataset = classify_experiment(dirname)

            # Determine experiment type based on model subdirectories
            subdirs = [d.name for d in item.iterdir() if d.is_dir() and d.name != "raw_api_responses"]
            has_openrouter = any("openrouter" in s for s in subdirs)
            has_commercial = any(s in ("gpt_4o_mini", "gpt_4_1_mini", "gemini_gemini_2_5_flash",
                                       "xai_grok_4_1_fast_non_reasoning_latest") for s in subdirs)

            if has_openrouter:
                exp_type = "oss"
            elif has_commercial:
                exp_type = "commercial"
            else:
                exp_type = "unknown"

            experiments.append(ExperimentStats(
                name=dirname,
                dataset=dataset,
                exp_type=exp_type,
                dir_path=item,
            ))

    return experiments


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_experiments(experiments: list) -> list:
    """Process all experiments: count debates, parse API responses, merge."""

    monitor_data = parse_monitor_logs()

    for exp in experiments:
        print(f"\nProcessing: {exp.name} ({exp.dataset}, {exp.exp_type})")

        # 1. Count debate JSON files per model
        debate_counts = count_debate_jsons(exp.dir_path)
        print(f"  Debate JSON files found: {sum(debate_counts.values())} across {len(debate_counts)} models")

        # 2. Parse raw API responses
        api_stats = parse_raw_api_responses(exp.dir_path)
        print(f"  Raw API response records: {sum(m.api_calls for m in api_stats.values())} across {len(api_stats)} models")

        # 3. Merge: create model entries for ALL models found in either source
        all_model_dirs = set(debate_counts.keys())
        all_api_models = set(api_stats.keys())

        # Map dir names to API model IDs for matching
        dir_to_api = {d: dir_name_to_model_id(d) for d in all_model_dirs}

        # Start with API stats (these have token/timing data)
        for model_id, ms in api_stats.items():
            exp.models[model_id] = ms

        # Add debate counts, matching dir names to API model IDs
        for dir_name, count in debate_counts.items():
            api_id = dir_to_api[dir_name]
            if api_id in exp.models:
                exp.models[api_id].debates_completed = count
            else:
                # Model has debates but no raw API log (e.g. gpt-4o-mini early run,
                # or OSS models run in separate processes without logging)
                ms = exp.get_or_create_model(api_id)
                ms.debates_completed = count
                # Estimate tokens from debate count: ~13 API calls/debate
                # Average ~900 input tokens, ~400 output tokens per call
                ms.api_calls = count * EXPECTED_API_CALLS_PER_DEBATE
                ms.input_tokens = ms.api_calls * 900
                ms.output_tokens = ms.api_calls * 400
                # Estimate timing: ~4.5 sec/API call average
                ms.total_api_duration_sec = ms.api_calls * 4.5
                ms.parse_ok = ms.api_calls
                print(f"  Estimated tokens for {api_id} ({count} debates, no raw API log)")

        # 4. Attach monitor log timing if available
        for label, mdata in monitor_data.items():
            # Match monitor label to experiment
            if exp.dataset.upper().replace("NLSY97", "NLSY97") in label.upper():
                if exp.exp_type == "oss" and "OSS" in label.upper():
                    exp.wall_clock_sec = mdata["wall_clock_sec"]
                    exp.avg_debates_per_min = mdata["avg_debates_per_min"]
                    exp.avg_sec_per_debate = mdata["avg_sec_per_debate"]
                    print(f"  Matched monitor log: {label} ({mdata['wall_clock_sec']/60:.0f} min logged)")

    return experiments


def format_number(n: int | float, decimal: int = 0) -> str:
    """Format a number with thousands separators."""
    if decimal > 0:
        return f"{n:,.{decimal}f}"
    return f"{int(n):,}"


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def generate_report(experiments: list) -> str:
    """Generate the full text report."""
    lines = []
    W = 120  # report width

    lines.append("=" * W)
    lines.append("AGENTICSIMLAW EXPERIMENT COST, TIMING & RESOURCE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    # -----------------------------------------------------------------------
    # Section 1: Summary Table
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("1. EXPERIMENT SUMMARY TABLE")
    lines.append("-" * W)

    header = (
        f"{'Experiment':<52} {'Dataset':<8} {'Type':<11} {'Models':>6} {'Debates':>8} "
        f"{'API Calls':>10} {'Tokens':>12} {'Est Cost':>10} {'Wall Time':>10} {'s/debate':>9}"
    )
    lines.append(header)
    lines.append("-" * W)

    # Sort: commercial first, then by dataset
    sorted_exps = sorted(experiments, key=lambda e: (0 if e.exp_type == "commercial" else 1, e.dataset, e.name))

    for exp in sorted_exps:
        wall = format_duration(exp.wall_clock_sec) if exp.wall_clock_sec else "N/A"
        spd = f"{exp.avg_sec_per_debate:.1f}" if exp.avg_sec_per_debate else "N/A"

        line = (
            f"{exp.name:<52} {exp.dataset:<8} {exp.exp_type:<11} {exp.model_count:>6} "
            f"{exp.total_debates:>8} {format_number(exp.total_api_calls):>10} "
            f"{format_number(exp.total_tokens):>12} ${exp.total_cost:>8.2f} "
            f"{wall:>10} {spd:>9}"
        )
        lines.append(line)

    lines.append("-" * W)

    # -----------------------------------------------------------------------
    # Section 2: Per-Model Breakdown (grouped by experiment)
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("2. PER-MODEL BREAKDOWN")
    lines.append("=" * W)

    for exp in sorted_exps:
        lines.append("")
        lines.append(f"  Experiment: {exp.name}")
        lines.append(f"  Dataset: {exp.dataset} | Type: {exp.exp_type}")
        lines.append(f"  {'─' * (W - 4)}")

        model_header = (
            f"  {'Model':<55} {'Debates':>8} {'API Calls':>10} "
            f"{'Input Tok':>12} {'Output Tok':>12} {'Est Cost':>10} {'Avg Resp':>9}"
        )
        lines.append(model_header)
        lines.append(f"  {'─' * (W - 4)}")

        for model_name in sorted(exp.models.keys()):
            ms = exp.models[model_name]
            display_name = model_name
            if len(display_name) > 53:
                display_name = "..." + display_name[-50:]

            line = (
                f"  {display_name:<55} {ms.debates_completed:>8} "
                f"{format_number(ms.api_calls):>10} "
                f"{format_number(ms.input_tokens):>12} "
                f"{format_number(ms.output_tokens):>12} "
                f"${ms.est_cost:>8.4f} "
                f"{ms.avg_response_sec:>8.2f}s"
            )
            lines.append(line)

        lines.append(f"  {'─' * (W - 4)}")
        lines.append(
            f"  {'SUBTOTAL':<55} {exp.total_debates:>8} "
            f"{format_number(exp.total_api_calls):>10} "
            f"{format_number(exp.total_input_tokens):>12} "
            f"{format_number(exp.total_output_tokens):>12} "
            f"${exp.total_cost:>8.4f}"
        )

    # -----------------------------------------------------------------------
    # Section 3: Aggregate Totals
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("")
    lines.append("3. AGGREGATE TOTALS")
    lines.append("=" * W)

    grand_debates = sum(e.total_debates for e in experiments)
    grand_api_calls = sum(e.total_api_calls for e in experiments)
    grand_input = sum(e.total_input_tokens for e in experiments)
    grand_output = sum(e.total_output_tokens for e in experiments)
    grand_tokens = grand_input + grand_output
    grand_cost = sum(e.total_cost for e in experiments)
    grand_api_sec = sum(e.total_api_duration_sec for e in experiments)

    # Compute hours by category
    commercial_exps = [e for e in experiments if e.exp_type == "commercial"]
    oss_exps = [e for e in experiments if e.exp_type == "oss"]

    lines.append("")
    lines.append(f"  {'Metric':<40} {'Commercial':>15} {'OSS':>15} {'Grand Total':>15}")
    lines.append(f"  {'─' * 90}")
    lines.append(
        f"  {'Total Debates':<40} "
        f"{format_number(sum(e.total_debates for e in commercial_exps)):>15} "
        f"{format_number(sum(e.total_debates for e in oss_exps)):>15} "
        f"{format_number(grand_debates):>15}"
    )
    lines.append(
        f"  {'Total API Calls':<40} "
        f"{format_number(sum(e.total_api_calls for e in commercial_exps)):>15} "
        f"{format_number(sum(e.total_api_calls for e in oss_exps)):>15} "
        f"{format_number(grand_api_calls):>15}"
    )
    lines.append(
        f"  {'Est. Input Tokens':<40} "
        f"{format_number(sum(e.total_input_tokens for e in commercial_exps)):>15} "
        f"{format_number(sum(e.total_input_tokens for e in oss_exps)):>15} "
        f"{format_number(grand_input):>15}"
    )
    lines.append(
        f"  {'Est. Output Tokens':<40} "
        f"{format_number(sum(e.total_output_tokens for e in commercial_exps)):>15} "
        f"{format_number(sum(e.total_output_tokens for e in oss_exps)):>15} "
        f"{format_number(grand_output):>15}"
    )
    lines.append(
        f"  {'Est. Total Tokens':<40} "
        f"{format_number(sum(e.total_tokens for e in commercial_exps)):>15} "
        f"{format_number(sum(e.total_tokens for e in oss_exps)):>15} "
        f"{format_number(grand_tokens):>15}"
    )
    lines.append(
        f"  {'Est. Cost (USD)':<40} "
        f"${sum(e.total_cost for e in commercial_exps):>14.2f} "
        f"${sum(e.total_cost for e in oss_exps):>14.2f} "
        f"${grand_cost:>14.2f}"
    )
    lines.append(
        f"  {'Total API Duration':<40} "
        f"{format_duration(sum(e.total_api_duration_sec for e in commercial_exps)):>15} "
        f"{format_duration(sum(e.total_api_duration_sec for e in oss_exps)):>15} "
        f"{format_duration(grand_api_sec):>15}"
    )

    # Compute hours
    lines.append("")
    lines.append(f"  Cumulative API compute time: {grand_api_sec/3600:.1f} hours")
    lines.append(f"  Estimated total wall-clock time (all experiments): ~{grand_api_sec/3600/4:.1f} hours")
    lines.append(f"    (API time / ~4 for concurrency of 30 parallel debates)")

    # -----------------------------------------------------------------------
    # Section 4: Per-Dataset Summary
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("")
    lines.append("4. PER-DATASET SUMMARY")
    lines.append("=" * W)

    for dataset in ["NLSY97", "COMPAS"]:
        ds_exps = [e for e in experiments if e.dataset == dataset]
        if not ds_exps:
            continue
        lines.append("")
        lines.append(f"  Dataset: {dataset}")
        lines.append(f"  {'─' * 90}")
        ds_debates = sum(e.total_debates for e in ds_exps)
        ds_calls = sum(e.total_api_calls for e in ds_exps)
        ds_tokens = sum(e.total_tokens for e in ds_exps)
        ds_cost = sum(e.total_cost for e in ds_exps)

        # Collect unique models
        ds_models = set()
        for e in ds_exps:
            ds_models.update(e.models.keys())

        lines.append(f"    Experiments: {len(ds_exps)}")
        lines.append(f"    Unique models: {len(ds_models)}")
        lines.append(f"    Total debates: {format_number(ds_debates)}")
        lines.append(f"    Total API calls: {format_number(ds_calls)}")
        lines.append(f"    Total tokens: {format_number(ds_tokens)}")
        lines.append(f"    Estimated cost: ${ds_cost:.2f}")
        lines.append(f"    Models: {', '.join(sorted(ds_models))}")

    # -----------------------------------------------------------------------
    # Section 5: Monitor Log Timing Analysis
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("")
    lines.append("5. MONITOR LOG TIMING ANALYSIS")
    lines.append("=" * W)

    monitor_data = parse_monitor_logs()
    if monitor_data:
        for label, mdata in sorted(monitor_data.items()):
            lines.append(f"")
            lines.append(f"  {label}")
            lines.append(f"  {'─' * 60}")
            lines.append(f"    Source: {mdata['log_file']}")
            lines.append(f"    Wall-clock time logged: {format_duration(mdata['wall_clock_sec'])}")
            lines.append(f"    Throughput at end: {mdata['avg_debates_per_min']} debates/min")
            lines.append(f"    Avg seconds/debate: {mdata['avg_sec_per_debate']:.1f}s")
            lines.append(f"    Progress at end: {mdata['final_done']}/{mdata['final_total']} "
                         f"({100*mdata['final_done']/mdata['final_total']:.1f}%)")
            # Extrapolate total time
            if mdata['final_done'] > 0 and mdata['final_total'] > 0:
                fraction = mdata['final_done'] / mdata['final_total']
                est_total = mdata['wall_clock_sec'] / fraction if fraction > 0 else 0
                lines.append(f"    Estimated total if completed: ~{format_duration(est_total)}")
    else:
        lines.append("  No monitor logs found.")

    # -----------------------------------------------------------------------
    # Section 6: Hardware/Infrastructure Notes
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("")
    lines.append("6. HARDWARE & INFRASTRUCTURE")
    lines.append("=" * W)
    lines.append("")
    lines.append("  Experiments run on macOS (Darwin 24.6.0), API calls to OpenAI, Google, xAI")
    lines.append("  (direct), and OpenRouter (for OSS models). Concurrency: 30 parallel debates.")
    lines.append("")
    lines.append("  Commercial models:")
    lines.append("    - gpt-4o-mini (OpenAI): $0.15/$0.60 per 1M tokens (input/output)")
    lines.append("    - gpt-4.1-mini (OpenAI): $0.40/$1.60 per 1M tokens")
    lines.append("    - gemini-2.5-flash (Google): $0.15/$0.60 per 1M tokens")
    lines.append("    - grok-4-1-fast (xAI): $0.60/$4.00 per 1M tokens")
    lines.append("")
    lines.append("  OSS models (via OpenRouter):")
    lines.append("    - 7-9B models: ~$0.065/$0.065 per 1M tokens")
    lines.append("    - 14B models (phi-4): ~$0.14/$0.14 per 1M tokens")
    lines.append("    - 24B models (dolphin3.0-mistral-24b): ~$0.14/$0.14 per 1M tokens")
    lines.append("    - 32B models (olmo-2-0325-32b): ~$0.20/$0.20 per 1M tokens")
    lines.append("")
    lines.append("  Debate protocol: 7 speaker turns + ~6 silent judge opinions = ~13 API calls/debate")
    lines.append("  Token estimation: tiktoken cl100k_base where metadata counts are unavailable")
    lines.append("")
    lines.append("  NOTE: Cost estimates are approximate. Models without raw API logs have")
    lines.append("  tokens estimated at ~900 input / ~400 output per API call based on observed")
    lines.append("  averages from logged models. OpenRouter pricing uses representative rates")
    lines.append("  which may differ from actual charges.")

    # -----------------------------------------------------------------------
    # Section 7: Compact Table for Paper Appendix
    # -----------------------------------------------------------------------
    lines.append("")
    lines.append("")
    lines.append("7. COMPACT TABLE FOR PAPER APPENDIX (LaTeX-ready)")
    lines.append("=" * W)
    lines.append("")

    # Group experiments by (dataset, type)
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[(exp.dataset, exp.exp_type)].append(exp)

    lines.append("  \\begin{table}[h]")
    lines.append("  \\centering")
    lines.append("  \\caption{Experiment resource summary.}")
    lines.append("  \\label{tab:experiment-costs}")
    lines.append("  \\begin{tabular}{llrrrrrr}")
    lines.append("  \\toprule")
    lines.append("  Dataset & Models & Debates & API Calls & Tokens (M) & Est. Cost (\\$) & s/call & s/debate$^\\dagger$ \\\\")
    lines.append("  \\midrule")

    for (dataset, exp_type), exps in sorted(grouped.items()):
        total_debates = sum(e.total_debates for e in exps)
        total_calls = sum(e.total_api_calls for e in exps)
        total_tokens = sum(e.total_tokens for e in exps)
        total_cost = sum(e.total_cost for e in exps)

        all_models = set()
        for e in exps:
            all_models.update(e.models.keys())

        # Avg seconds per API call (from logged durations)
        total_dur = sum(e.total_api_duration_sec for e in exps)
        total_calls_count = max(1, sum(e.total_api_calls for e in exps))
        avg_call_sec = total_dur / total_calls_count

        # Wall-clock s/debate: use monitor-log if available, else estimate
        spds = [e.avg_sec_per_debate for e in exps if e.avg_sec_per_debate]
        if spds:
            avg_spd = sum(spds) / len(spds)
            spd_str = f"{avg_spd:.1f}"
        else:
            # For commercial (concurrent), estimate wall-clock s/debate from avg API call time
            # With concurrency ~30, wall-clock ~ API_time_per_call * calls_per_debate / concurrency
            avg_spd = avg_call_sec * EXPECTED_API_CALLS_PER_DEBATE / 30 * 1.2  # 20% overhead
            spd_str = f"$\\approx${avg_spd:.1f}"

        type_label = "Commercial" if exp_type == "commercial" else "OSS"
        lines.append(
            f"  {dataset} ({type_label}) & {len(all_models)} & "
            f"{format_number(total_debates)} & {format_number(total_calls)} & "
            f"{total_tokens/1_000_000:.2f} & {total_cost:.2f} & {avg_call_sec:.1f} & {spd_str} \\\\"
        )

    # Grand total row
    total_dur_all = sum(e.total_api_duration_sec for e in experiments)
    total_calls_all = max(1, sum(e.total_api_calls for e in experiments))
    avg_call_all = total_dur_all / total_calls_all

    lines.append("  \\midrule")
    lines.append(
        f"  \\textbf{{Total}} & & "
        f"\\textbf{{{format_number(grand_debates)}}} & "
        f"\\textbf{{{format_number(grand_api_calls)}}} & "
        f"\\textbf{{{grand_tokens/1_000_000:.2f}}} & "
        f"\\textbf{{{grand_cost:.2f}}} & {avg_call_all:.1f} & \\\\"
    )
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\vspace{2pt}")
    lines.append("  \\footnotesize{$^\\dagger$Wall-clock seconds per debate with 30-way concurrency.")
    lines.append("  OSS values from monitor logs; commercial values estimated from API latencies.}")
    lines.append("  \\end{table}")

    lines.append("")
    lines.append("=" * W)
    lines.append("END OF REPORT")
    lines.append("=" * W)

    return "\n".join(lines)


def main():
    print("AgenticSimLaw Experiment Cost Calculator")
    print("=" * 50)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    print(f"\nProject directory: {PROJECT_DIR}")
    experiments = discover_experiments()
    print(f"Found {len(experiments)} experiment directories:")
    for exp in experiments:
        print(f"  - {exp.name} ({exp.dataset}, {exp.exp_type})")

    # Process
    experiments = process_experiments(experiments)

    # Generate report
    report = generate_report(experiments)

    # Write report
    OUTPUT_FILE.write_text(report)
    print(f"\n{'=' * 50}")
    print(f"Report written to: {OUTPUT_FILE}")
    print(f"{'=' * 50}")

    # Also print to stdout
    print("\n")
    print(report)


if __name__ == "__main__":
    main()
