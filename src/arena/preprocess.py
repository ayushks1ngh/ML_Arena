"""Preprocessing. Built as an UNFITTED sklearn transformer: fitting happens on the training split only."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Median-impute + scale numeric columns, mode-impute + one-hot categoricals.

    Numeric columns come FIRST in the output, so numeric_idx = range(n_numeric);
    the robustness code relies on this to perturb numeric features only.
    """
    num = [c for c in X.columns if X[c].dtype.kind in "if"]
    cat = [c for c in X.columns if c not in num]
    steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)]
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(steps), len(num)
