"""Shared plotting helpers (used by notebooks; the dashboard has Plotly equivalents)."""
import matplotlib.pyplot as plt

from .evaluate import HIGHER_IS_BETTER


def leaderboard_bar(lb, metric, ax=None, title=None):
    d = lb[lb.metric == metric].sort_values("mean", ascending=not HIGHER_IS_BETTER[metric])
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    colors = ["#2a9d8f" if b else ("#e9c46a" if t else "#8d99ae") for b, t in zip(d.is_best, d.tie_with_best)]
    ax.barh(d.model, d["mean"], xerr=d["std"].fillna(0), color=colors)
    ax.invert_yaxis(); ax.set_xlabel(metric + (" (higher is better)" if HIGHER_IS_BETTER[metric] else " (lower is better)"))
    ax.set_title(title or f"{metric}: green=best, yellow=tie within 1 std")
    return ax
