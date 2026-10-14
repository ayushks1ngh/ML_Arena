"""Classification leaderboard runs. Usage: python scripts/run_classification.py [--quick] [--models dt rf]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from arena.config import QUICK_SEEDS, RAW_DIR, SEEDS
from arena.evaluate import run_one, split, to_rows
from arena.models import CLF_MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--models", nargs="*", default=list(CLF_MODELS))
a = ap.parse_args()

rows = []
for seed in (QUICK_SEEDS if a.quick else SEEDS):
    data = split("clf", seed)          # same split for every model at this seed
    for m in a.models:
        res = run_one("clf", m, seed, quick=a.quick, data=data)
        rows += to_rows("clf", m, seed, res)
        print(f"seed={seed} {m:6s} acc={res.metrics['accuracy']:.3f} f1_macro={res.metrics['f1_macro']:.3f} "
              f"fit={res.fit_time:.1f}s", flush=True)
out = RAW_DIR / ("clf_baseline_quick.csv" if a.quick else "clf_baseline.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
