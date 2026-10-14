"""Clustering arena (Mall Customers): sweeps, internal metrics with noise handling, and stability under noise.
No ground truth exists, so we use silhouette / Davies-Bouldin / cluster counts and stability (ARI vs. clean)."""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import PRED_DIR, RAW_DIR, SEEDS, SUMMARY_DIR
from .data import load_mall

FEATURES = {"2d": ["Annual Income (k$)", "Spending Score (1-100)"],
            "3d": ["Annual Income (k$)", "Spending Score (1-100)", "Age"]}
STAB_SIGMAS = [0.1, 0.25, 0.5]
MAX_NOISE_FRAC = 0.30    # a DBSCAN config that labels >30% of points as noise is not selected


def get_X(variant):
    df = load_mall()
    return StandardScaler().fit_transform(df[FEATURES[variant]])


def make(algo, params, seed=0):
    if algo == "kmeans":
        return KMeans(n_clusters=params["k"], n_init=10, random_state=seed)
    if algo == "meanshift":
        return MeanShift(bandwidth=params["bandwidth"])
    if algo == "agg_single":
        return AgglomerativeClustering(n_clusters=params["k"], linkage="single")
    if algo == "dbscan":
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    raise ValueError(algo)


def internal_metrics(X, labels):
    """Silhouette / DB are computed on NON-noise points (label -1 = noise); NaN if fewer than 2 clusters."""
    noise = labels == -1
    core, lab = X[~noise], labels[~noise]
    k = len(set(lab))
    out = dict(n_clusters=k, noise_frac=float(noise.mean()), silhouette=np.nan, davies_bouldin=np.nan)
    if k >= 2 and len(lab) > k:
        out["silhouette"], out["davies_bouldin"] = silhouette_score(core, lab), davies_bouldin_score(core, lab)
    return out


def candidate_grid(algo, X):
    if algo in ("kmeans", "agg_single"):
        return [{"k": k} for k in range(2, 11)]
    if algo == "meanshift":
        return [{"bandwidth": float(estimate_bandwidth(X, quantile=q)), "quantile": q} for q in (0.1, 0.2, 0.3)]
    return [{"eps": e, "min_samples": m} for e in (0.2, 0.3, 0.4, 0.5, 0.6) for m in (3, 5)]


def run_arena(variant="2d", seeds=SEEDS):
    X = get_X(variant)
    rows, selected, labels_out = [], [], {}
    for algo in ("kmeans", "meanshift", "agg_single", "dbscan"):
        best = None
        for params in candidate_grid(algo, X):
            for seed in (seeds if algo == "kmeans" else [0]):     # only k-means is stochastic
                labels = make(algo, params, seed).fit_predict(X)
                met = internal_metrics(X, labels)
                rows.append(dict(dataset="mall", features=variant, algorithm=algo, params_json=json.dumps(params),
                                 seed=seed, **met))
                if seed == 0:
                    ok = not np.isnan(met["silhouette"]) and met["noise_frac"] <= MAX_NOISE_FRAC
                    if ok and (best is None or met["silhouette"] > best[1]["silhouette"]):
                        best = (params, met, labels)
        if best:
            selected.append(dict(features=variant, algorithm=algo, params_json=json.dumps(best[0]), **best[1]))
            labels_out[f"{algo}_{variant}"] = best[2]
    return pd.DataFrame(rows), pd.DataFrame(selected), labels_out, X


def stability(X, selected, seeds=SEEDS):
    """ARI between the clean clustering and the clustering of noised data (selected params held fixed)."""
    rows = []
    for _, s in selected.iterrows():
        params = json.loads(s.params_json)
        clean = make(s.algorithm, params, 0).fit_predict(X)
        sil_clean = internal_metrics(X, clean)["silhouette"]
        for sigma in STAB_SIGMAS:
            for seed in seeds:
                Xn = X + np.random.default_rng([seed, int(sigma * 100)]).normal(0, sigma, X.shape)
                noisy = make(s.algorithm, params, seed).fit_predict(Xn)
                m = internal_metrics(Xn, noisy)
                rows.append(dict(features=s.features, algorithm=s.algorithm, sigma=sigma, seed=seed,
                                 ari=adjusted_rand_score(clean, noisy), sil_clean=sil_clean,
                                 sil_noisy=m["silhouette"], n_clusters_noisy=m["n_clusters"],
                                 noise_frac_noisy=m["noise_frac"]))
    return pd.DataFrame(rows)


def run_all(seeds=SEEDS):
    raws, sels, stabs = [], [], []
    labels_df = load_mall()[["CustomerID"]].copy()
    for variant in ("2d", "3d"):
        raw, sel, labels, X = run_arena(variant, seeds)
        raws.append(raw); sels.append(sel); stabs.append(stability(X, sel, seeds))
        for k, v in labels.items():
            labels_df[k] = v
    raw, sel, stab = pd.concat(raws), pd.concat(sels), pd.concat(stabs)
    raw.to_csv(RAW_DIR / "clustering.csv", index=False)
    stab.to_csv(RAW_DIR / "clustering_stability.csv", index=False)
    sel.to_csv(SUMMARY_DIR / "clustering_selected.csv", index=False)
    labels_df.to_csv(PRED_DIR / "clustering_labels.csv", index=False)
    stab_sum = stab.groupby(["features", "algorithm", "sigma"]).agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
                                                                    sil_noisy=("sil_noisy", "mean")).reset_index()
    stab_sum.to_csv(SUMMARY_DIR / "clustering_stability_summary.csv", index=False)
    return raw, sel, stab_sum
