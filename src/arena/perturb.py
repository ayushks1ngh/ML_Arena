"""Perturbations used by the Robustness Lab.

Pure functions: explicit RNG, return NEW arrays, never touch the input.
They act on the PREPROCESSED matrix Z. Numeric columns are the first `n_num` columns (standardized), so
sigma is measured in training-standard-deviation units. One-hot columns are left intact by noise/missing.
"""
import numpy as np


def add_noise(Z, n_num, sigma, rng):
    out = Z.copy()
    out[:, :n_num] += rng.normal(0.0, sigma, size=(Z.shape[0], n_num))
    return out


def train_fill_values(pre):
    """Where a train-median-imputed value lands in standardized space (fit on TRAIN only)."""
    num = pre.named_transformers_["num"]
    return (num.named_steps["imp"].statistics_ - num.named_steps["sc"].mean_) / num.named_steps["sc"].scale_


def mask_missing(Z, n_num, p, rng, fill):
    """MCAR: each numeric entry is lost with prob p and replaced by the TRAIN median (never test statistics)."""
    out = Z.copy()
    mask = rng.random((Z.shape[0], n_num)) < p
    out[:, :n_num] = np.where(mask, fill[None, :], out[:, :n_num])
    return out


def ablate_columns(Z, cols, train_means):
    """Targeted feature loss: replace the given columns by their TRAIN mean."""
    out = Z.copy()
    out[:, cols] = train_means[cols]
    return out


def make_imbalanced(X, y, minority_share, rng):
    """Downsample the positive (minority) class of the TRAIN set so it is `minority_share` of the new set.
    Returns X, y unchanged if the class is already that rare."""
    pos, neg = np.where(y.values == 1)[0], np.where(y.values == 0)[0]
    if len(pos) > len(neg):
        pos, neg = neg, pos
    keep_n = int(round(minority_share / (1 - minority_share) * len(neg)))
    if keep_n >= len(pos):
        return X, y
    keep = np.concatenate([rng.choice(pos, keep_n, replace=False), neg])
    keep.sort()
    return X.iloc[keep], y.iloc[keep]
