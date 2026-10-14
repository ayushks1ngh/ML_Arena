"""Computes the verdict for H1-H6 from saved result tables, following the decision rules in
docs/EXPERIMENT_DESIGN.md section 7. Writes results/summary/hypotheses.csv."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .config import RAW_DIR, SUMMARY_DIR
from .models import CLF_MODELS, REG_MODELS

ENSEMBLE, DISTANCE = ["rf", "ada", "xgb"], ["knn", "nc"]


def _h1(task):
    st = pd.read_csv(SUMMARY_DIR / f"robustness_rank_stability_{task}.csv")
    changed = st[st.top1_changed]
    clear = changed[changed.gap_vs_baseline_winner > changed.top1_std]
    n = len(st)
    ev = (f"baseline winner={st.baseline_top1.iloc[0]}; top-1 changed in {len(changed)}/{n} (family, level) cells; "
          f"changed by more than the new winner's seed std in {len(clear)}/{n}. "
          + "; ".join(f"{r.family}@{r.level}: {r.top1}" for r in clear.itertuples()))
    v = "Supported" if len(clear) else ("Inconclusive" if len(changed) else "Not supported")
    return dict(id=f"H1 ({task})", claim="Best baseline model is not best under every perturbation", evidence=ev, verdict=v)


def _h2():
    rc = pd.read_csv(RAW_DIR / "robustness_clf.csv")
    d = rc[(rc.metric == "f1_macro") & (rc.family == "noise")]
    g = d.assign(group=np.where(d.model.isin(ENSEMBLE), "ensembles", np.where(d.model.isin(["knn", "ann"]), "knn_ann", "other")))
    t = g[g.group != "other"].groupby(["level", "group"]).rel_change.mean().unstack()
    wins = int((t.ensembles > t.knn_ann).sum())
    ev = "mean macro-F1 rel_change under noise (ensembles rf/ada/xgb vs knn/ann) per sigma: " + "; ".join(
        f"{lvl}: {r.ensembles:+.3f} vs {r.knn_ann:+.3f}" for lvl, r in t.iterrows()) + f". Ensembles better at {wins}/{len(t)} levels."
    v = "Supported" if wins > len(t) / 2 else ("Inconclusive" if wins == len(t) / 2 else "Not supported")
    return dict(id="H2", claim="Tree ensembles lose less than kNN/ANN under feature noise", evidence=ev, verdict=v)


def _h3():
    s = pd.read_csv(SUMMARY_DIR / "robustness_summary_clf.csv")
    ev, ok = [], True
    for fam in ("noise", "missing"):
        f = s[s.family == fam].sort_values("mean_rel_change")
        worst2 = list(f.model.head(2))
        ev.append(f"{fam}: most affected = {', '.join(f'{r.model} ({r.mean_rel_change:+.3f})' for r in f.head(3).itertuples())}; "
                  f"distance models (knn {f.set_index('model').mean_rel_change['knn']:+.3f}, nc {f.set_index('model').mean_rel_change['nc']:+.3f})")
        ok &= set(worst2) == set(DISTANCE)
    return dict(id="H3", claim="Distance-based models (kNN, NC) suffer the largest drops under noise and missing values",
                evidence=" | ".join(ev), verdict="Supported" if ok else "Not supported")


def _h4(family, label):
    rc = pd.read_csv(RAW_DIR / "robustness_clf.csv")
    low = min(rc[rc.family == family].level)
    d = rc[(rc.family == family) & (rc.level == low)]
    m = d.groupby(["model", "metric"]).agg(base=("baseline_score", "mean"), pert=("perturbed_score", "mean")).unstack()
    acc_drop = m["base"]["accuracy"] - m["pert"]["accuracy"]
    f1_drop = m["base"]["f1_macro"] - m["pert"]["f1_macro"]
    rec_drop = m["base"]["recall"] - m["pert"]["recall"]
    hidden = [mod for mod in m.index if acc_drop[mod] <= 0.02 and max(f1_drop[mod], rec_drop[mod]) >= 0.10]
    tau = kendalltau(m["pert"]["accuracy"], m["pert"]["f1_macro"])[0]
    ev = (f"{label}, minority share {low}: " + "; ".join(f"{mod}: acc {-acc_drop[mod]:+.3f}, macroF1 {-f1_drop[mod]:+.3f}, minority recall {-rec_drop[mod]:+.3f}"
                                              for mod in m.index) + f". Models with accuracy within 2pts but >=10pt F1/recall loss: {hidden or 'none'}. "
          f"Kendall tau(accuracy ranking, macro-F1 ranking) = {tau:.2f}.")
    v = "Supported" if hidden and tau < 1 else ("Inconclusive" if hidden or tau < 1 else "Not supported")
    return dict(id=f"H4 ({label})", claim="Under imbalance accuracy hides minority-class failure", evidence=ev, verdict=v)


def _h5(task):
    s = pd.read_csv(SUMMARY_DIR / f"robustness_summary_{task}.csv")
    reg = CLF_MODELS if task == "clf" else REG_MODELS
    avg = s.groupby("model").mean_rel_change.mean()
    cx = pd.Series({m: reg[m].complexity for m in avg.index})
    rho, p = spearmanr(cx, avg)
    ev = f"Spearman rho(complexity, mean rel_change across families) = {rho:+.2f} (p={p:.2f}, n={len(avg)} models; small n)."
    v = "Supported" if (abs(rho) < 0.5 or p > 0.05) else "Not supported"
    return dict(id=f"H5 ({task})", claim="Complexity does not predict robustness", evidence=ev, verdict=v)


def _h6():
    st = pd.read_csv(RAW_DIR / "clustering_stability.csv")
    out, ev = [], []
    for feat in ("2d", "3d"):
        a = st[st.features == feat].groupby(["algorithm", "sigma"]).ari.mean().unstack(0)
        lower = ((a.agg_single < a.kmeans).sum(), (a.meanshift < a.kmeans).sum())
        ev.append(f"{feat}: ARI(kmeans) vs single/meanshift per sigma = " + "; ".join(
            f"{s}: {r.kmeans:.2f} vs {r.agg_single:.2f}/{r.meanshift:.2f}" for s, r in a.iterrows()))
        out.append(all(x > len(a) / 2 for x in lower))
    nf = st[(st.algorithm == "dbscan") & (st.features == "2d")].groupby("sigma").noise_frac_noisy.mean()
    ev.append("DBSCAN noise fraction vs sigma (2d): " + ", ".join(f"{s}: {v:.2f}" for s, v in nf.items()))
    return dict(id="H6", claim="Single linkage and Mean Shift are less stable than K-means under noise",
                evidence=" | ".join(ev), verdict="Supported" if all(out) else ("Inconclusive" if any(out) else "Not supported"))


def run():
    rows = [_h1("clf"), _h1("reg"), _h2(), _h3(), _h4("imbalance", "train skewed, balanced test"), _h4("imbalance_matched", "train AND test skewed"), _h5("clf"), _h5("reg"), _h6()]
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_DIR / "hypotheses.csv", index=False)
    return df
