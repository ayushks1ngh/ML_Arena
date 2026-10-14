import numpy as np
import pandas as pd
from arena.perturb import ablate_columns, add_noise, make_imbalanced, mask_missing


def test_noise_std_and_no_mutation():
    Z = np.zeros((20000, 5)); Z0 = Z.copy()
    out = add_noise(Z, 3, 0.5, np.random.default_rng(0))
    assert np.array_equal(Z, Z0)                      # input untouched
    assert abs(out[:, :3].std() - 0.5) < 0.02 and np.all(out[:, 3:] == 0)  # numeric only, sigma in std units


def test_same_seed_same_output():
    Z = np.random.default_rng(1).normal(size=(50, 4))
    a = add_noise(Z, 4, 0.3, np.random.default_rng(7)); b = add_noise(Z, 4, 0.3, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_mask_fraction_and_fill():
    Z = np.ones((20000, 3)); fill = np.array([-5., -5., -5.])
    out = mask_missing(Z, 3, 0.3, np.random.default_rng(0), fill)
    assert abs((out == -5).mean() - 0.3) < 0.01


def test_ablate_uses_train_means():
    Z = np.ones((5, 4)); out = ablate_columns(Z, [1], np.array([0., 9., 0., 0.]))
    assert np.all(out[:, 1] == 9) and np.all(out[:, 0] == 1)


def test_imbalance_share_and_majority_kept():
    y = pd.Series([1] * 380 + [0] * 620); X = pd.DataFrame({"a": range(1000)})
    Xi, yi = make_imbalanced(X, y, 0.10, np.random.default_rng(0))
    assert abs(yi.mean() - 0.10) < 0.01 and (yi == 0).sum() == 620
    X2, y2 = make_imbalanced(X, y, 0.5, np.random.default_rng(0))   # already rarer than target -> unchanged
    assert len(y2) == 1000


def test_rel_change_sign_convention():
    from arena.robustness import rel_change
    assert rel_change("accuracy", 0.8, 0.6) < 0          # score fell -> negative
    assert rel_change("rmse", 0.5, 1.0) < 0              # error rose -> negative (worse)
    assert abs(rel_change("rmse", 0.5, 1.0) + 1.0) < 1e-9
