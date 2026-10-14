"""Clustering arena + stability. Usage: python scripts/run_clustering.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from arena.clustering import run_all

pd.set_option("display.width", 180)
raw, sel, stab = run_all()
print(sel.round(3).to_string(index=False)); print(stab.round(3).to_string(index=False))
