"""Runs everything in order: tests, E0, E1-E5, analysis.

    python run_all.py              # full run (about 1-2 minutes on a laptop)
    python run_all.py --seeds 20   # quicker run with fewer seeds

Works the same on Linux, macOS and Windows.
"""
import argparse
import subprocess
import sys


def step(*args):
    print("\n>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    a = p.parse_args()
    step("-m", "pytest", "-q", "tests")
    step("-m", "experiments.e0_xmono_check")
    step("-m", "experiments.run_experiments", "--exp", "all", "--seeds", str(a.seeds))
    step("-m", "analysis.analyze")
    print("\nDone. Tables: results/summary.md, figures: figures/")
