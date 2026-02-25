#!/usr/bin/env python3
"""Compute sign test for debate calibration across model-dataset pairs.

Tests whether debate achieves lowest BRD (Base-Rate Deviation) across
model-dataset pairs more often than chance (H0: p=1/3 for 3 methods).

Usage:
  python scripts/compute_sign_test.py

  Fill in BRD values from experiment results before running.
"""
from scipy import stats

# ── BRD values per (model, dataset) ───────────────────
# Format: (model, dataset): {"zs": BRD, "sc": BRD, "debate": BRD}
# BRD = |YES% - base_rate|
#
# BASE RATES: nlsy97=0.36, compas=0.45, credit_default=0.22

RESULTS = {
    # ── Existing pairs (from Table 10 in paper) ──
    ("gpt-4o-mini",             "nlsy97"):  {"zs": 0.470, "sc": 0.490, "debate": 0.023},
    ("gpt-4.1-mini",            "nlsy97"):  {"zs": 0.360, "sc": 0.380, "debate": 0.303},
    ("gemini/gemini-2.5-flash", "nlsy97"):  {"zs": 0.460, "sc": 0.460, "debate": 0.191},
    ("gpt-4o-mini",             "compas"):  {"zs": 0.070, "sc": 0.080, "debate": 0.077},
    ("gpt-4.1-mini",            "compas"):  {"zs": 0.080, "sc": 0.060, "debate": 0.073},
    ("gemini/gemini-2.5-flash", "compas"):  {"zs": 0.110, "sc": 0.110, "debate": 0.098},
    # ── NEW pairs from C3 experiments ──
    ("gpt-4o-mini",             "credit_default"):  {"zs": 0.020, "sc": 0.020, "debate": 0.090},
    ("gpt-4.1-mini",            "credit_default"):  {"zs": 0.040, "sc": 0.040, "debate": 0.013},
    # gemini/gemini-2.5-flash credit_default: excluded (API limit + parsing failures)
}


def main():
    # Filter to pairs with complete data
    complete = {
        k: v for k, v in RESULTS.items()
        if all(val is not None for val in v.values())
    }

    n = len(complete)
    wins = sum(
        1 for brds in complete.values()
        if brds["debate"] < min(brds["zs"], brds["sc"])
    )
    ties = sum(
        1 for brds in complete.values()
        if brds["debate"] == min(brds["zs"], brds["sc"])
    )

    print(f"Complete pairs: {n}")
    print(f"Debate wins (lowest BRD): {wins}/{n}")
    print(f"Ties: {ties}/{n}")
    print()

    # One-sided binomial test: debate wins > chance (1/3)
    # Using exact binomial test
    p_binom = stats.binomtest(wins, n, 1/3, alternative="greater").pvalue
    print(f"Binomial test (H0: p=1/3): p = {p_binom:.4f}")

    # Also report simple sign test (debate < best-of-rest)
    p_sign = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    print(f"Sign test (H0: p=0.5):     p = {p_sign:.4f}")

    # Per-pair detail
    print("\nPer-pair results:")
    print(f"{'Model':<35} {'Dataset':<18} {'ZS':>6} {'SC':>6} {'Debate':>6}  Winner")
    print("-" * 95)
    for (model, ds), brds in sorted(complete.items()):
        winner = min(brds, key=brds.get)
        marker = " <<<" if winner == "debate" else ""
        print(f"{model:<35} {ds:<18} {brds['zs']:>6.3f} {brds['sc']:>6.3f} {brds['debate']:>6.3f}  {winner}{marker}")


if __name__ == "__main__":
    main()
