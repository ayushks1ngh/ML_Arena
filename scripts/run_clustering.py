"""Clustering arena + stability. Usage: python scripts/run_clustering.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import argparse
from arena.clustering import run_all
from arena.config import QUICK_SEEDS, SEEDS

pd.set_option("display.width", 180)
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
raw, sel, stab = run_all(QUICK_SEEDS if a.quick else SEEDS)
print(sel.round(3).to_string(index=False)); print(stab.round(3).to_string(index=False))
