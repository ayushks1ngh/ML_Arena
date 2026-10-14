"""Shared baseline runner for classification and regression (the leaderboard experiments)."""
import pandas as pd

from .config import PRED_DIR, QUICK_SEEDS, RAW_DIR, SEEDS
from .evaluate import TASKS, run_one, split, to_rows


def run_baseline(task, quick=False, models=None, seeds=None):
    registry = TASKS[task][2]
    models = models or list(registry)
    seeds = seeds or (QUICK_SEEDS if quick else SEEDS)
    rows, preds = [], None
    for seed in seeds:
        data = split(task, seed)               # identical split for every model at this seed
        for m in models:
            res = run_one(task, m, seed, quick=quick, data=data)
            rows += to_rows(task, m, seed, res)
            main = "f1_macro" if task == "clf" else "rmse"
            print(f"{task} seed={seed} {m:6s} {main}={res.metrics[main]:.4f} fit={res.fit_time:.1f}s", flush=True)
            if seed == seeds[0]:               # keep seed-0 predictions for the Failure Investigator
                if preds is None:
                    preds = pd.DataFrame({"row_id": data[1].index, "y_true": data[3].values})
                preds[f"{m}_pred"] = res.pred
                if res.proba is not None:
                    preds[f"{m}_proba"] = res.proba
    suffix = "_quick" if quick else ""
    pd.DataFrame(rows).to_csv(RAW_DIR / f"{task}_baseline{suffix}.csv", index=False)
    preds.to_csv(PRED_DIR / f"{task}_predictions_seed0{suffix}.csv", index=False)
    return pd.DataFrame(rows)
