#!/usr/bin/env python3
"""
Generate all paper figures from data/results.

Reproduces every figure referenced in the TMLR 2026 paper:
  - figure_brd_comparison.pdf          (Table 10 BRD bar chart)
  - figure_commercial_comparison.pdf   (Table 7 commercial LLM comparison)
  - figure_cooperative_ablation.pdf    (Table 13 adversarial vs cooperative)
  - figure_random_flip_ablation.pdf    (Random-flip Monte Carlo ablation)

Figures that are NOT regenerated (conceptual/legacy):
  - figure1_madcal_flowchart.pdf       (hand-drawn architecture diagram)
  - figureC1/C2                        (NLSY97 OSS scatter plots from step4/step6)
  - figureB1/B2                        (wall-clock/token plots from step4)

Usage:
  python scripts/generate_all_figures.py

Output directory: figures/
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURE_SCRIPTS = [
    ("figure_brd_comparison.pdf", "generate_brd_figure.py"),
    ("figure_commercial_comparison.pdf", "generate_commercial_comparison_figure.py"),
    ("figure_cooperative_ablation.pdf", "generate_cooperative_figure.py"),
    ("figure_random_flip_ablation.pdf", "generate_random_flip_figure.py"),
]


def main():
    print("=" * 60)
    print("Generating all paper figures")
    print("=" * 60)

    success = 0
    failed = 0

    for fig_name, script_name in FIGURE_SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        print(f"\n[{success + failed + 1}/{len(FIGURE_SCRIPTS)}] {fig_name}")
        print(f"  Script: {script_name}")

        if not os.path.exists(script_path):
            print(f"  ERROR: Script not found: {script_path}")
            failed += 1
            continue

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                print(f"  OK")
                success += 1
            else:
                print(f"  FAILED (exit code {result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:500]}")
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>600s)")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Figure generation complete: {success}/{len(FIGURE_SCRIPTS)} succeeded")
    if failed:
        print(f"  {failed} failed")
    print(f"Output: figures/")
    print("=" * 60)

    # List generated figures
    fig_dir = os.path.join(SCRIPT_DIR, "..", "figures")
    if os.path.isdir(fig_dir):
        pdfs = sorted(f for f in os.listdir(fig_dir) if f.endswith(".pdf"))
        if pdfs:
            print(f"\nGenerated figures ({len(pdfs)}):")
            for f in pdfs:
                size = os.path.getsize(os.path.join(fig_dir, f))
                print(f"  {f} ({size:,} bytes)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
