#!/usr/bin/env python3
"""
Step 0: Ollama Model Management

Pulls and manages Ollama models from YAML configuration files.

Paper Reference: Section 3.2 "Experimental Setup"
- Models sourced from ollama.ai
- 4-bit quantized (q4_K_M) unless otherwise noted
- Three ensembles: size (14 models), OSS (8 models), all (78 models)

Configuration Files:
- config_ollama_models_size.yaml: Parameter size ensembles (1B-70B)
- config_ollama_models_oss.yaml: Open-source model set
- config_ollama_models_reasoning.yaml: Reasoning-focused models
- config_ollama_models_all.yaml: Complete model list

Usage:
    python step0_pull_rename_ollama_models_ver3.py --config size
    python step0_pull_rename_ollama_models_ver3.py --config all --dry-run
    python step0_pull_rename_ollama_models_ver3.py --model llama3.1:8b-instruct-q4_K_M
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG_DIR = "../configs"

CONFIG_FILES = {
    'size': 'config_ollama_models_size.yaml',
    'oss': 'config_ollama_models_oss.yaml',
    'reasoning': 'config_ollama_models_reasoning.yaml',
    'all': 'config_ollama_models_all.yaml',
}

# Small ensemble for quick testing (from paper Table 1)
SMALL_ENSEMBLE = [
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


# ============================================================================
# Logging
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: Path) -> List[str]:
    """Load model list from YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model list
    models = config.get('ollama_pull_models', [])

    if not models:
        raise ValueError(f"No models found in config: {config_path}")

    return models


def get_config_path(config_name: str, config_dir: Path) -> Path:
    """Get full path to config file."""
    if config_name in CONFIG_FILES:
        return config_dir / CONFIG_FILES[config_name]
    else:
        # Assume it's a direct path
        return Path(config_name)


# ============================================================================
# Ollama Operations
# ============================================================================

def check_ollama_installed() -> bool:
    """Check if Ollama CLI is installed."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_installed_models() -> List[str]:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return []

        # Parse output (skip header line)
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                # Model name is first column
                model_name = line.split()[0]
                models.append(model_name)

        return models

    except Exception:
        return []


def pull_model(model_name: str, logger: logging.Logger) -> bool:
    """Pull a single Ollama model."""
    logger.info(f"Pulling model: {model_name}")

    try:
        result = subprocess.run(
            ['ollama', 'pull', model_name],
            capture_output=False,  # Show progress
            text=True
        )

        if result.returncode == 0:
            logger.info(f"  ✓ Successfully pulled: {model_name}")
            return True
        else:
            logger.error(f"  ✗ Failed to pull: {model_name}")
            return False

    except Exception as e:
        logger.error(f"  ✗ Error pulling {model_name}: {e}")
        return False


def check_model_availability(model_name: str) -> Dict[str, any]:
    """Check if a model is available (installed or pullable)."""
    installed_models = get_installed_models()

    if model_name in installed_models:
        return {'status': 'installed', 'model': model_name}

    # Try to show model info (will fail if not available)
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return {'status': 'available', 'model': model_name}
        else:
            return {'status': 'not_found', 'model': model_name}

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'model': model_name}
    except Exception as e:
        return {'status': 'error', 'model': model_name, 'error': str(e)}


# ============================================================================
# Main Operations
# ============================================================================

def pull_models_from_config(
    config_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    skip_existing: bool = True
) -> Dict[str, List[str]]:
    """
    Pull all models from a config file.

    Returns:
        Dict with 'success', 'failed', 'skipped' model lists
    """
    models = load_config(config_path)
    logger.info(f"Found {len(models)} models in config: {config_path.name}")

    installed = get_installed_models() if skip_existing else []

    results = {
        'success': [],
        'failed': [],
        'skipped': [],
    }

    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Processing: {model}")

        # Skip if already installed
        if model in installed:
            logger.info(f"  → Already installed, skipping")
            results['skipped'].append(model)
            continue

        if dry_run:
            logger.info(f"  → Would pull (dry-run)")
            continue

        # Pull the model
        success = pull_model(model, logger)

        if success:
            results['success'].append(model)
        else:
            results['failed'].append(model)

    return results


def pull_single_model(
    model_name: str,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Pull a single model by name."""
    if dry_run:
        logger.info(f"Would pull: {model_name} (dry-run)")
        return True

    return pull_model(model_name, logger)


def list_models(logger: logging.Logger):
    """List all installed models."""
    models = get_installed_models()

    if not models:
        logger.info("No models installed")
        return

    logger.info(f"Installed models ({len(models)}):")
    for model in sorted(models):
        logger.info(f"  - {model}")


def check_ensemble(
    ensemble: List[str],
    logger: logging.Logger
) -> Dict[str, List[str]]:
    """Check which models from an ensemble are installed."""
    installed = get_installed_models()

    results = {
        'installed': [],
        'missing': [],
    }

    for model in ensemble:
        if model in installed:
            results['installed'].append(model)
        else:
            results['missing'].append(model)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Model Management for AgenticSimLaw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                         List installed models
  %(prog)s --config size                  Pull models from size ensemble config
  %(prog)s --config all --dry-run         Show what would be pulled
  %(prog)s --model llama3.1:8b            Pull a specific model
  %(prog)s --check-ensemble               Check if small ensemble is installed
        """
    )

    parser.add_argument('--config', type=str, choices=list(CONFIG_FILES.keys()),
                       help='Config preset to use (size, oss, reasoning, all)')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom config YAML file')
    parser.add_argument('--model', type=str,
                       help='Pull a specific model by name')
    parser.add_argument('--list', action='store_true',
                       help='List installed models')
    parser.add_argument('--check-ensemble', action='store_true',
                       help='Check if small ensemble models are installed')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    parser.add_argument('--force', action='store_true',
                       help='Pull even if model is already installed')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("Ollama Model Management")
    logger.info("=" * 60)

    # Check Ollama installation
    if not check_ollama_installed():
        logger.error("Ollama is not installed or not in PATH")
        logger.error("Install from: https://ollama.ai/")
        sys.exit(1)

    logger.info("Ollama CLI: OK")

    # Handle different operations
    if args.list:
        list_models(logger)
        return

    if args.check_ensemble:
        logger.info(f"\nChecking small ensemble ({len(SMALL_ENSEMBLE)} models):")
        results = check_ensemble(SMALL_ENSEMBLE, logger)

        logger.info(f"\nInstalled ({len(results['installed'])}):")
        for m in results['installed']:
            logger.info(f"  ✓ {m}")

        logger.info(f"\nMissing ({len(results['missing'])}):")
        for m in results['missing']:
            logger.info(f"  ✗ {m}")

        if results['missing']:
            logger.info(f"\nTo install missing models, run:")
            logger.info(f"  python {sys.argv[0]} --config size")
        return

    if args.model:
        success = pull_single_model(args.model, logger, args.dry_run)
        sys.exit(0 if success else 1)

    if args.config or args.config_file:
        # Determine config path
        script_dir = Path(__file__).parent
        config_dir = script_dir / DEFAULT_CONFIG_DIR

        if args.config_file:
            config_path = Path(args.config_file)
            if not config_path.is_absolute():
                config_path = script_dir / config_path
        else:
            config_path = get_config_path(args.config, config_dir)

        logger.info(f"Using config: {config_path}")

        results = pull_models_from_config(
            config_path,
            logger,
            dry_run=args.dry_run,
            skip_existing=not args.force
        )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if args.dry_run:
            logger.info("(Dry run - no models were actually pulled)")

        logger.info(f"Success: {len(results['success'])}")
        logger.info(f"Failed:  {len(results['failed'])}")
        logger.info(f"Skipped: {len(results['skipped'])}")

        if results['failed']:
            logger.warning("\nFailed models:")
            for m in results['failed']:
                logger.warning(f"  - {m}")

        sys.exit(0 if not results['failed'] else 1)

    # No operation specified
    parser.print_help()


if __name__ == "__main__":
    main()
