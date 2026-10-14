"""Fit/tune/score one model for one seed. The unit of work for every experiment script."""
import json
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .config import INNER_CV, TEST_SIZE
from .data import load_california, load_titanic
from .models import CLF_MODELS, REG_MODELS
from .preprocess import build_preprocessor

TASKS = {
    "clf": ("titanic", load_titanic, CLF_MODELS, "f1_macro"),
    "reg": ("california", load_california, REG_MODELS, "neg_root_mean_squared_error"),
}
# Metrics where HIGHER is better; the rest (mae, rmse) are errors where lower is better.
HIGHER_IS_BETTER = {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "f1_macro": 1, "roc_auc": 1,
                    "r2": 1, "mae": 0, "rmse": 0}


def clf_metrics(y_true, y_pred, proba=None):
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, zero_division=0),
         "recall": recall_score(y_true, y_pred, zero_division=0),
         "f1": f1_score(y_true, y_pred, zero_division=0),
         "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    if proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, proba)
    return m


def reg_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred)}


def score(task, estimator, X, y):
    """Score a fitted estimator on already-transformed features X."""
    pred = estimator.predict(X)
    if task == "clf":
        proba = estimator.predict_proba(X)[:, 1] if hasattr(estimator, "predict_proba") else None
        return clf_metrics(y, pred, proba), pred, proba
    return reg_metrics(y, pred), pred, None


def split(task, seed):
    name, loader, _, _ = TASKS[task]
    X, y = loader()
    strat = y if task == "clf" else None
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=strat)


@dataclass
class FitResult:
    preprocessor: object      # fitted on TRAIN only
    estimator: object         # fitted on TRAIN only (tuned)
    n_numeric: int
    best_params: dict
    metrics: dict
    pred: np.ndarray
    proba: object
    X_test_index: object
    fit_time: float
    predict_time: float


def run_one(task, model_key, seed, quick=False, data=None):
    """Tune with inner CV on the training set only, refit, score on the held-out test set."""
    _, _, registry, tune_metric = TASKS[task]
    spec = registry[model_key]
    Xtr, Xte, ytr, yte = data if data is not None else split(task, seed)
    pre, n_num = build_preprocessor(Xtr)
    grid = {f"est__{k}": ([v[0]] if quick else v) for k, v in spec.grid.items()}
    pipe = Pipeline([("pre", pre), ("est", spec.factory(seed))])
    t0 = time.time()
    if grid and not quick:
        gs = GridSearchCV(pipe, grid, scoring=tune_metric, cv=INNER_CV, n_jobs=1)
        gs.fit(Xtr, ytr)
        fitted, params = gs.best_estimator_, {k[5:]: v for k, v in gs.best_params_.items()}
    else:
        params = {k[5:]: v[0] for k, v in grid.items()}
        fitted = pipe.set_params(**{k: v[0] for k, v in grid.items()}).fit(Xtr, ytr)
    fit_time = time.time() - t0
    pre_f, est_f = fitted.named_steps["pre"], fitted.named_steps["est"]
    t1 = time.time()
    Xte_t = pre_f.transform(Xte)
    metrics, pred, proba = score(task, est_f, Xte_t, yte)
    return FitResult(pre_f, est_f, n_num, params, metrics, pred, proba, Xte.index,
                     fit_time, time.time() - t1)


def to_rows(task, model_key, seed, res: FitResult):
    """Long-format rows matching the schema in docs/EXPERIMENT_DESIGN.md section 9."""
    name = TASKS[task][0]
    return [dict(task=task, dataset=name, model=model_key, seed=seed, metric=k, value=v,
                 fit_time_s=res.fit_time, predict_time_s=res.predict_time,
                 best_params=json.dumps(res.best_params, default=str))
            for k, v in res.metrics.items()]
