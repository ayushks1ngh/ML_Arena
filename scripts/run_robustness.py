"""Robustness sweeps. Usage: python scripts/run_robustness.py --task clf|reg [--quick]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from arena.config import QUICK_SEEDS, RAW_DIR, SEEDS
from arena.robustness import sweep, write_summaries

ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["clf", "reg"])
ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
raw = pd.read_csv(RAW_DIR / f"{a.task}_baseline.csv")        # tuned params come from the full baseline run
seeds = QUICK_SEEDS if a.quick else SEEDS
models = [m for m in raw.model.unique() if m != "dummy"]
df = sweep(a.task, models, seeds, raw[raw.metric == ("f1_macro" if a.task == "clf" else "rmse")], quick=a.quick)
suffix = "_quick" if a.quick else ""
df.to_csv(RAW_DIR / f"robustness_{a.task}{suffix}.csv", index=False)
if not a.quick:
    write_summaries(a.task)
print("wrote", len(df), "rows")
