"""Model registries. Each ModelSpec = estimator factory + small tuning grid + complexity rank (for H5)."""
from dataclasses import dataclass, field

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelSpec:
    factory: callable                 # seed -> unfitted estimator
    grid: dict = field(default_factory=dict)   # keys WITHOUT the "est__" prefix
    complexity: int = 1               # documented ordinal 1 (simple) .. 5 (complex)


CLF_MODELS = {
    "dummy": ModelSpec(lambda s: DummyClassifier(strategy="most_frequent"), {}, 0),
    "dt": ModelSpec(lambda s: DecisionTreeClassifier(criterion="entropy", random_state=s),
                    {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]}, 2),
    "knn": ModelSpec(lambda s: KNeighborsClassifier(),
                     {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]}, 2),
    "nc": ModelSpec(lambda s: NearestCentroid(), {"metric": ["euclidean", "manhattan"]}, 1),
    "ann": ModelSpec(lambda s: MLPClassifier(max_iter=1000, early_stopping=True, random_state=s),
                     {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}, 5),
    "rf": ModelSpec(lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1),
                    {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]}, 4),
    "ada": ModelSpec(lambda s: AdaBoostClassifier(random_state=s),
                     {"n_estimators": [50, 200], "learning_rate": [0.1, 0.5, 1.0]}, 3),
    "xgb": ModelSpec(lambda s: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=s, n_jobs=4),
                     {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}, 5),
}

REG_MODELS = {
    "dummy": ModelSpec(lambda s: DummyRegressor(strategy="mean"), {}, 0),
    "lin": ModelSpec(lambda s: LinearRegression(), {}, 1),
    "dt": ModelSpec(lambda s: DecisionTreeRegressor(random_state=s),
                    {"max_depth": [5, 10, None], "min_samples_leaf": [5, 20]}, 2),
    "rf": ModelSpec(lambda s: RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=-1),
                    {"max_depth": [10, None], "min_samples_leaf": [1, 5]}, 4),
    "gb": ModelSpec(lambda s: GradientBoostingRegressor(n_estimators=200, random_state=s),
                    {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}, 4),
    "xgb": ModelSpec(lambda s: XGBRegressor(n_estimators=300, random_state=s, n_jobs=4),
                     {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}, 5),
}
