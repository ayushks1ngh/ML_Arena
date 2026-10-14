"""Preprocessing. Built as an UNFITTED sklearn transformer: fitting happens on the training split only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


class Discretizer(BaseEstimator, TransformerMixin):
    """Turns Titanic features into integer-coded categories for ID3 (fit on TRAIN only).

    Age/Fare: median-impute, then 4 quantile bins (edges learned on train). Counts (SibSp, Parch, FamilySize):
    capped at 3. Everything else (Sex, Embarked, Title, Pclass, IsAlone): category codes learned on train;
    unseen categories become -1.
    """
    BINNED, CAPPED = ("Age", "Fare"), ("SibSp", "Parch", "FamilySize")

    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        self.median_ = {c: X[c].median() for c in self.BINNED}
        self.edges_ = {c: np.unique(np.quantile(X[c].fillna(self.median_[c]), [0.25, 0.5, 0.75])) for c in self.BINNED}
        self.cats_ = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                      for c in self.cols_ if c not in self.BINNED + self.CAPPED}
        return self

    def transform(self, X):
        out = {}
        for c in self.cols_:
            if c in self.BINNED:
                out[c] = np.digitize(X[c].fillna(self.median_[c]), self.edges_[c])
            elif c in self.CAPPED:
                out[c] = np.minimum(X[c].astype(int), 3)
            else:
                out[c] = X[c].astype(str).map(self.cats_[c]).fillna(-1).astype(int)
        return pd.DataFrame(out, index=X.index).to_numpy()

    def get_feature_names_out(self, names=None):
        return np.array(self.cols_)
