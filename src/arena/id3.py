"""ID3 decision tree from scratch (Quinlan): entropy, information gain, multiway splits on categorical features.

Input X is a 2-D array of integer-coded CATEGORICAL features. At each node we split on the feature with the
highest information gain; each distinct value gets its own branch. Stops when the node is pure, no features are
left, or max_depth is reached; leaves predict the majority class.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer  # noqa: F401  (used by the discretizer in preprocess.py)


def entropy(y):
    """H(y) = -sum p log2 p."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def information_gain(x, y):
    """Gain(y, feature x) = H(y) - sum_v |y_v|/|y| * H(y_v)."""
    h = entropy(y)
    for v in np.unique(x):
        m = x == v
        h -= m.mean() * entropy(y[m])
    return h


class ID3Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.tree_ = self._build(X, y, list(range(X.shape[1])), 0)
        return self

    def _leaf(self, y):
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        return {"leaf": True, "proba": counts / counts.sum(), "pred": self.classes_[counts.argmax()]}

    def _build(self, X, y, feats, depth):
        if len(np.unique(y)) == 1 or not feats or (self.max_depth is not None and depth >= self.max_depth):
            return self._leaf(y)
        gains = [information_gain(X[:, j], y) for j in feats]
        j = feats[int(np.argmax(gains))]
        if max(gains) <= 0:
            return self._leaf(y)
        rest = [f for f in feats if f != j]
        node = {"leaf": False, "feat": j, "default": self._leaf(y),     # default: unseen value -> node majority
                "children": {v: self._build(X[X[:, j] == v], y[X[:, j] == v], rest, depth + 1)
                             for v in np.unique(X[:, j])}}
        return node

    def _walk(self, row):
        node = self.tree_
        while not node["leaf"]:
            node = node["children"].get(row[node["feat"]], node["default"])
        return node

    def predict(self, X):
        return np.array([self._walk(r)["pred"] for r in np.asarray(X)])

    def predict_proba(self, X):
        return np.vstack([self._walk(r)["proba"] for r in np.asarray(X)])

    def export_text(self, feature_names, class_names=None, node=None, indent=""):
        """Readable printout of the learned tree (for the viva)."""
        node = node or self.tree_
        class_names = class_names or [str(c) for c in self.classes_]
        if node["leaf"]:
            return f"{indent}-> {class_names[int(np.where(self.classes_ == node['pred'])[0][0])]}\n"
        out = ""
        for v, child in sorted(node["children"].items()):
            out += f"{indent}{feature_names[node['feat']]} = {v}\n" + self.export_text(feature_names, class_names, child, indent + "    ")
        return out
