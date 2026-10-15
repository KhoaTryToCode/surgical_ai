"""
Universal Benchmark Aggregator CLI.
Scans all model results in TEMPORAL_EXP/results/ and compiles the global LEADERBOARD.csv and LEADERBOARD.md.

Usage:
  python aggregate.py
"""

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(EXP_DIR))

from core.aggregator import compile_leaderboard


def main():
    results_dir = str(EXP_DIR / "results")
    compile_leaderboard(results_dir)


if __name__ == '__main__':
    main()
