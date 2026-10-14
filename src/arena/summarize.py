"""Raw per-seed CSVs -> leaderboard tables (mean/std, rank, best flag, tie flag)."""
import pandas as pd

from .config import RAW_DIR, SUMMARY_DIR
from .evaluate import HIGHER_IS_BETTER


def leaderboard(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["model", "metric"]).agg(mean=("value", "mean"), std=("value", "std"),
                                              n_seeds=("seed", "nunique"), fit_time_s=("fit_time_s", "mean"))
    g = g.reset_index()
    out = []
    for metric, d in g.groupby("metric"):
        d = d.copy()
        asc = not HIGHER_IS_BETTER[metric]
        d["rank"] = d["mean"].rank(ascending=asc, method="min").astype(int)
        best = d.loc[d["rank"] == 1].iloc[0]
        d["is_best"] = d["rank"] == 1
        # "tie" = not the best, but within 1 std of the best model's mean -> not a real win
        d["tie_with_best"] = (~d["is_best"]) & ((d["mean"] - best["mean"]).abs() < best["std"])
        out.append(d)
    return pd.concat(out).sort_values(["metric", "rank"]).reset_index(drop=True)


def write_leaderboard(task: str, quick: bool = False) -> pd.DataFrame:
    suffix = "_quick" if quick else ""
    lb = leaderboard(pd.read_csv(RAW_DIR / f"{task}_baseline{suffix}.csv"))
    lb.to_csv(SUMMARY_DIR / f"leaderboard_{task}{suffix}.csv", index=False)
    return lb
