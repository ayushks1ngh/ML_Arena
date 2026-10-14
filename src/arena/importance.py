"""Grouped permutation importance: one-hot columns of the same original feature are shuffled together."""
import numpy as np

from .evaluate import score


def column_groups(pre):
    """Map preprocessed columns to their original feature (e.g. 'cat__Sex_male' -> 'Sex')."""
    names = pre.get_feature_names_out()
    groups = {}
    for i, n in enumerate(names):
        kind, rest = n.split("__", 1)
        groups.setdefault(rest if kind == "num" else rest.split("_")[0], []).append(i)
    return groups


def grouped_permutation_importance(task, est, Z, y, groups, rng, metric, n_repeats=3):
    """Importance = drop in `metric` (oriented so higher=better) when a group of columns is shuffled together."""
    sign = -1 if metric in ("rmse", "mae") else 1
    base = sign * score(task, est, Z, y)[0][metric]
    out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Zp = Z.copy()
            Zp[:, cols] = Z[rng.permutation(len(Z))][:, cols]
            drops.append(base - sign * score(task, est, Zp, y)[0][metric])
        out[g] = float(np.mean(drops))
    return out
