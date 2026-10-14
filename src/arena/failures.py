"""Failure Investigator: where models disagree / fail, subgroup errors, importance (permutation vs. native),
and the accuracy-robustness-complexity trade-off table. Uses seed-0 saved predictions + refits (fixed params)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import PRED_DIR, RAW_DIR, SUMMARY_DIR
from .data import load_california, load_titanic
from .evaluate import split
from .importance import column_groups, grouped_permutation_importance
from .models import CLF_MODELS, REG_MODELS
from .robustness import PRIMARY, _fit

SEED = 0


def _models(preds):
    return [c[:-5] for c in preds.columns if c.endswith("_pred") and c != "dummy_pred"]


def _native(est, groups, n_cols):
    """Native importance aggregated per original feature (tree importances or |coef|)."""
    if hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        v = np.abs(np.ravel(est.coef_))
    else:
        return {g: np.nan for g in groups}
    return {g: float(v[cols].sum()) for g, cols in groups.items()}


def _importance_table(task, models, base_raw, data, n_rows=None):
    Xtr, Xte, ytr, yte = data
    rows = []
    for m in models:
        res = _fit(task, m, SEED, base_raw, data)
        Z = res.preprocessor.transform(Xte)
        y = yte
        if n_rows:
            idx = np.random.default_rng(0).choice(len(Z), min(n_rows, len(Z)), replace=False)
            Z, y = Z[idx], yte.iloc[idx]
        groups = column_groups(res.preprocessor)
        perm = grouped_permutation_importance(task, res.estimator, Z, y, groups, np.random.default_rng(SEED),
                                              PRIMARY[task], n_repeats=5)
        nat = _native(res.estimator, groups, Z.shape[1])
        rows += [dict(model=m, feature=g, perm_importance=perm[g], native_importance=nat[g]) for g in groups]
    return pd.DataFrame(rows)


def classification_failures():
    preds = pd.read_csv(PRED_DIR / "clf_predictions_seed0.csv")
    models = _models(preds)
    X, _ = load_titanic()
    raw = X.loc[preds.row_id].reset_index(drop=True)
    wrong = pd.DataFrame({m: preds[f"{m}_pred"] != preds.y_true for m in models})
    cases = pd.concat([preds[["row_id", "y_true"]], raw, preds[[f"{m}_pred" for m in models]]], axis=1)
    cases["n_wrong"] = wrong.sum(axis=1)
    cases["category"] = np.where(cases.n_wrong == 0, "all_correct",
                                 np.where(cases.n_wrong == len(models), "hard_core", "contested"))
    cases.sort_values(["n_wrong"], ascending=False).to_csv(SUMMARY_DIR / "failures_clf_cases.csv", index=False)
    cm = [dict(model=m, true=t, pred=p, count=int(c)) for m in models
          for (t, p), c in np.ndenumerate(confusion_matrix(preds.y_true, preds[f"{m}_pred"], labels=[0, 1]))]
    pd.DataFrame(cm).to_csv(SUMMARY_DIR / "failures_clf_confusion.csv", index=False)
    age_band = pd.cut(raw.Age, [0, 12, 18, 60, 120], labels=["child", "teen", "adult", "senior"]).astype(object).fillna("unknown")
    sub = []
    for name, col in (("Sex", raw.Sex), ("Pclass", raw.Pclass.astype(str)), ("AgeBand", age_band)):
        for val in col.unique():
            mask = (col == val).to_numpy()
            for m in models:
                sub.append(dict(group=name, value=val, model=m, n=int(mask.sum()), error_rate=float(wrong.loc[mask, m].mean())))
    pd.DataFrame(sub).to_csv(SUMMARY_DIR / "failures_clf_subgroups.csv", index=False)
    base = pd.read_csv(RAW_DIR / "clf_baseline.csv"); base = base[base.metric == "f1_macro"]
    imp = _importance_table("clf", [m for m in ("xgb", "rf", "dt", "knn", "ann") if m in models], base, split("clf", SEED))
    imp.to_csv(SUMMARY_DIR / "failures_importance_clf.csv", index=False)
    return cases, imp


def regression_failures():
    preds = pd.read_csv(PRED_DIR / "reg_predictions_seed0.csv")
    models = _models(preds)
    X, _ = load_california()
    raw = X.loc[preds.row_id].reset_index(drop=True)
    bins = pd.cut(preds.y_true, [0, 1, 2, 3, 4, 4.99, 6], labels=["<1", "1-2", "2-3", "3-4", "4-5", "capped(5.0)"])
    rows = []
    for m in models:
        res = preds.y_true - preds[f"{m}_pred"]
        for b in bins.cat.categories:
            mask = (bins == b).to_numpy()
            rows.append(dict(model=m, target_range=b, n=int(mask.sum()), mae=float(res[mask].abs().mean()),
                             mean_residual=float(res[mask].mean())))
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "failures_reg_by_range.csv", index=False)
    best = "xgb" if "xgb" in models else models[0]
    err = (preds.y_true - preds[f"{best}_pred"]).abs()
    worst = pd.concat([preds[["row_id", "y_true", f"{best}_pred"]], raw], axis=1).assign(abs_error=err)
    worst.sort_values("abs_error", ascending=False).head(30).to_csv(SUMMARY_DIR / "failures_reg_worst.csv", index=False)
    base = pd.read_csv(RAW_DIR / "reg_baseline.csv"); base = base[base.metric == "rmse"]
    imp = _importance_table("reg", [m for m in ("xgb", "rf", "gb", "lin") if m in models], base, split("reg", SEED), n_rows=3000)
    imp.to_csv(SUMMARY_DIR / "failures_importance_reg.csv", index=False)
    return worst, imp


def tradeoff(task):
    """Accuracy (main metric) vs. robustness vs. complexity vs. fit time, one row per model."""
    main = PRIMARY[task]
    lb = pd.read_csv(SUMMARY_DIR / f"leaderboard_{task}.csv"); lb = lb[lb.metric == main].set_index("model")
    rb = pd.read_csv(SUMMARY_DIR / f"robustness_summary_{task}.csv")
    reg = CLF_MODELS if task == "clf" else REG_MODELS
    out = pd.DataFrame({"score": lb["mean"], "score_std": lb["std"], "fit_time_s": lb.fit_time_s})
    out["complexity"] = [reg[m].complexity for m in out.index]
    piv = rb.pivot(index="model", columns="family", values="mean_rel_change")
    out = out.join(piv.add_prefix("relchg_"))
    feat = [c for c in ("relchg_noise", "relchg_missing", "relchg_drop_feature") if c in out]
    out["feature_robustness"] = out[feat].mean(axis=1)
    out.loc["dummy"] = out.loc["dummy"] if "dummy" in out.index else None
    out = out.drop(index="dummy", errors="ignore")
    out.reset_index(names="model").to_csv(SUMMARY_DIR / f"tradeoff_{task}.csv", index=False)
    return out
