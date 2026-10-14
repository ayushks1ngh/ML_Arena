"""Classification leaderboard. Usage: python scripts/run_classification.py [--quick] [--models dt rf]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena.runner import run_baseline

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--models", nargs="*")
a = ap.parse_args()
run_baseline("clf", quick=a.quick, models=a.models)
