import numpy as np
from arena.data import load_titanic
from arena.evaluate import clf_metrics, reg_metrics, run_one, split
from arena.preprocess import build_preprocessor


def test_preprocessor_fit_on_train_only():
    Xtr, Xte, ytr, yte = split("clf", 0)
    pre, n_num = build_preprocessor(Xtr)
    pre.fit(Xtr)
    scaler = pre.named_transformers_["num"].named_steps["sc"]
    # train-only: the fitted scaler mean equals the train (imputed) mean, not the all-data mean
    X, _ = load_titanic()
    pre_all, _ = build_preprocessor(X); pre_all.fit(X)
    s_all = pre_all.named_transformers_["num"].named_steps["sc"]
    assert not np.allclose(scaler.mean_, s_all.mean_)


def test_split_is_stratified_and_reproducible():
    a = split("clf", 3); b = split("clf", 3)
    assert a[1].index.equals(b[1].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.02


def test_numeric_columns_first():
    Xtr, _, _, _ = split("clf", 0)
    pre, n_num = build_preprocessor(Xtr)
    Z = pre.fit_transform(Xtr)
    assert Z.shape[1] > n_num and np.allclose(Z[:, :n_num].mean(axis=0), 0, atol=0.2)


def test_metrics_known_values():
    m = clf_metrics(np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]))
    assert m["accuracy"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    r = reg_metrics(np.array([1., 2., 3.]), np.array([1., 2., 5.]))
    assert abs(r["mae"] - 2 / 3) < 1e-9 and abs(r["rmse"] - np.sqrt(4 / 3)) < 1e-9


def test_run_one_quick_beats_dummy():
    d = run_one("clf", "dummy", 0, quick=True).metrics["accuracy"]
    r = run_one("clf", "rf", 0, quick=True).metrics["accuracy"]
    assert r > d


def test_leaderboard_rank_and_tie():
    import pandas as pd
    from arena.summarize import leaderboard
    rows = []
    for model, vals in {"a": [0.80, 0.82], "b": [0.79, 0.81], "c": [0.50, 0.52]}.items():
        for seed, v in enumerate(vals):
            rows.append(dict(model=model, metric="accuracy", seed=seed, value=v, fit_time_s=0.1))
            rows.append(dict(model=model, metric="rmse", seed=seed, value=1 - v, fit_time_s=0.1))
    lb = leaderboard(pd.DataFrame(rows)).set_index(["metric", "model"])
    assert lb.loc[("accuracy", "a"), "rank"] == 1 and lb.loc[("rmse", "a"), "rank"] == 1  # lower rmse is better
    assert lb.loc[("accuracy", "b"), "tie_with_best"] and not lb.loc[("accuracy", "c"), "tie_with_best"]
