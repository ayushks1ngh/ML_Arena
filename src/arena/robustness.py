"""Robustness Lab: perturb the TEST set of a fixed trained model (noise / missing / drop_feature),
or the TRAIN set and retrain (imbalance). Records baseline, perturbed, delta, rel_change per metric."""
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .config import DROP_TOP_K, MINORITY_SHARES, MISSING_FRACS, NOISE_SIGMAS, RAW_DIR, SUMMARY_DIR
from .evaluate import HIGHER_IS_BETTER, TASKS, clf_metrics, run_one, score, split
from .importance import column_groups, grouped_permutation_importance
from .perturb import (ablate_columns, add_noise, make_imbalanced, mask_missing, train_fill_values)

METRICS = {"clf": ["accuracy", "f1_macro", "recall"], "reg": ["rmse", "mae", "r2"]}
PRIMARY = {"clf": "f1_macro", "reg": "rmse"}
FAMILY_CODE = {"noise": 1, "missing": 2, "drop_feature": 3, "imbalance": 4, "imbalance_matched": 5}
ORIG_SHARE = 0.38      # approx. Titanic positive share; severity axis for imbalance = ORIG_SHARE - share


def rel_change(metric, base, pert):
    """Negative = worse for EVERY metric (error metrics are sign-flipped)."""
    d = (pert - base) if HIGHER_IS_BETTER[metric] else (base - pert)
    return d / abs(base) if base != 0 else np.nan


def _rows(task, model, seed, family, level, base, pert, params):
    return [dict(task=task, dataset=TASKS[task][0], model=model, seed=seed, family=family, level=level,
                 metric=m, baseline_score=base[m], perturbed_score=pert[m], delta=pert[m] - base[m],
                 rel_change=rel_change(m, base[m], pert[m]), params_json=json.dumps(params)) for m in METRICS[task]]


def sweep(task, models, seeds, baseline_raw, quick=False):
    """baseline_raw: the tuned-baseline CSV (provides best_params per model/seed so we never re-tune)."""
    metric_main = PRIMARY[task]
    rows = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = split(task, seed)
        # reference importance (RF) on the TRAIN set decides which features 'drop_feature' removes
        ref = _fit(task, "rf", seed, baseline_raw, (Xtr, Xte, ytr, yte))
        Ztr = ref.preprocessor.transform(Xtr)
        groups = column_groups(ref.preprocessor)
        imp = grouped_permutation_importance(task, ref.estimator, Ztr[:2000], ytr.iloc[:2000], groups,
                                             np.random.default_rng(seed), metric_main)
        ranked = sorted(imp, key=imp.get, reverse=True)
        for model in models:
            res = _fit(task, model, seed, baseline_raw, (Xtr, Xte, ytr, yte))
            pre, est, n_num = res.preprocessor, res.estimator, res.n_numeric
            Zte, ytest = pre.transform(Xte), yte
            base = score(task, est, Zte, ytest)[0]
            fill, means = train_fill_values(pre), pre.transform(Xtr).mean(axis=0)
            for i, s in enumerate(NOISE_SIGMAS):
                rng = np.random.default_rng([seed, FAMILY_CODE["noise"], i])
                pert = score(task, est, add_noise(Zte, n_num, s, rng), ytest)[0]
                rows += _rows(task, model, seed, "noise", s, base, pert, {"sigma": s})
            for i, p in enumerate(MISSING_FRACS):
                rng = np.random.default_rng([seed, FAMILY_CODE["missing"], i])
                pert = score(task, est, mask_missing(Zte, n_num, p, rng, fill), ytest)[0]
                rows += _rows(task, model, seed, "missing", p, base, pert, {"p": p})
            for k in DROP_TOP_K:
                cols = [c for g in ranked[:k] for c in groups[g]]
                pert = score(task, est, ablate_columns(Zte, cols, means), ytest)[0]
                rows += _rows(task, model, seed, "drop_feature", k, base, pert, {"dropped": ranked[:k]})
            if task == "clf":          # imbalance: change TRAIN data, retrain with the same hyperparameters
                for i, share in enumerate(MINORITY_SHARES):
                    rng = np.random.default_rng([seed, FAMILY_CODE["imbalance"], i])
                    Xi, yi = make_imbalanced(Xtr, ytr, share, rng)
                    r2 = run_one(task, model, seed, data=(Xi, Xte, yi, yte), fixed_params=res.best_params)
                    rows += _rows(task, model, seed, "imbalance", share, base, r2.metrics,
                                  {"minority_share": share, "n_train": len(yi)})
                    # matched variant: the TEST set is skewed the same way (deployment prior = training prior),
                    # which is where accuracy can hide minority-class failure. Baseline = clean model on same rows.
                    rng2 = np.random.default_rng([seed, FAMILY_CODE["imbalance_matched"], i])
                    Xs, ys = make_imbalanced(Xte, yte, share, rng2)
                    pos = yte.index.get_indexer(ys.index)
                    base_m = clf_metrics(ys, res.pred[pos], res.proba[pos] if res.proba is not None else None)
                    pert_m = clf_metrics(ys, r2.pred[pos], r2.proba[pos] if r2.proba is not None else None)
                    rows += _rows(task, model, seed, "imbalance_matched", share, base_m, pert_m,
                                  {"minority_share": share, "n_test": len(ys)})
            print(f"{task} seed={seed} {model} done", flush=True)
    return pd.DataFrame(rows)


def _fit(task, model, seed, baseline_raw, data):
    p = baseline_raw[(baseline_raw.model == model) & (baseline_raw.seed == seed)].best_params.iloc[0]
    return run_one(task, model, seed, data=data, fixed_params=json.loads(p))


def severity(family, level):
    return ORIG_SHARE - level if family.startswith("imbalance") else level


def summarize_robustness(df, task):
    """Per model & family: mean rel_change, normalised degradation-curve area, worst drop.
    Plus rank stability per (family, level): Kendall tau vs. baseline ranking, rank flips, top-1 changes."""
    m = PRIMARY[task]
    d = df[df.metric == m]
    # ---- per-model summary
    srows = []
    for (model, fam), g in d.groupby(["model", "family"]):
        by_level = g.groupby("level").agg(rel=("rel_change", "mean"), base=("baseline_score", "mean"),
                                          pert=("perturbed_score", "mean")).sort_index()
        sev = np.array([0.0] + [severity(fam, l) for l in by_level.index])
        sev = sev
        ratio = (by_level.pert / by_level.base) if HIGHER_IS_BETTER[m] else (by_level.base / by_level.pert)
        curve = np.concatenate([[1.0], ratio.values])
        order = np.argsort(sev); span = sev.max() - sev.min()
        auc = np.trapezoid(curve[order], sev[order]) / span if span > 0 else np.nan
        srows.append(dict(model=model, family=fam, metric=m, mean_rel_change=by_level.rel.mean(),
                          auc_norm=auc, worst_drop=by_level.rel.min()))
    summary = pd.DataFrame(srows)
    # ---- rank stability
    base_scores = d[d.family == d.family.iloc[0]].groupby("model").baseline_score.mean()
    asc = not HIGHER_IS_BETTER[m]
    base_rank = base_scores.rank(ascending=asc, method="min")
    base_std = d[d.family == d.family.iloc[0]].groupby(["model", "seed"]).baseline_score.first().groupby("model").std()
    top_base = base_rank.idxmin()
    rrows, rank_rows = [], []
    for (fam, level), g in d.groupby(["family", "level"]):
        sc = g.groupby("model").perturbed_score.mean(); sd = g.groupby("model").perturbed_score.std()
        rk = sc.rank(ascending=asc, method="min")
        top = rk.idxmin()
        tau = kendalltau(base_rank.loc[sc.index], rk)[0] if len(sc) > 2 else np.nan
        rrows.append(dict(family=fam, level=level, metric=m, kendall_tau=tau,
                          rank_flips=int((rk != base_rank.loc[sc.index]).sum()), top1=top, baseline_top1=top_base,
                          top1_changed=top != top_base,
                          gap_vs_baseline_winner=abs(sc[top] - sc[top_base]), top1_std=sd[top]))
        for model in sc.index:
            rank_rows.append(dict(family=fam, level=level, model=model, mean_score=sc[model], std=sd[model],
                                  rank=int(rk[model]), baseline_rank=int(base_rank[model])))
    return summary, pd.DataFrame(rrows), pd.DataFrame(rank_rows)


def write_summaries(task):
    df = pd.read_csv(RAW_DIR / f"robustness_{task}.csv")
    summ, stab, ranks = summarize_robustness(df, task)
    summ.to_csv(SUMMARY_DIR / f"robustness_summary_{task}.csv", index=False)
    stab.to_csv(SUMMARY_DIR / f"robustness_rank_stability_{task}.csv", index=False)
    ranks.to_csv(SUMMARY_DIR / f"robustness_ranks_{task}.csv", index=False)
    return summ, stab, ranks
