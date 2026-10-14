"""Model registries. Each ModelSpec = estimator factory + small tuning grid + complexity rank (for H5)."""
from dataclasses import dataclass, field

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class ModelSpec:
    factory: callable                 # seed -> unfitted estimator
    grid: dict = field(default_factory=dict)   # keys WITHOUT the "est__" prefix
    complexity: int = 1               # documented ordinal 1 (simple) .. 5 (complex)


CLF_MODELS = {
    "dummy": ModelSpec(lambda s: DummyClassifier(strategy="most_frequent"), {}, 0),
    "dt": ModelSpec(lambda s: DecisionTreeClassifier(criterion="entropy", random_state=s),
                    {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]}, 2),
    "rf": ModelSpec(lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1),
                    {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]}, 4),
}

REG_MODELS = {
    "dummy": ModelSpec(lambda s: DummyRegressor(strategy="mean"), {}, 0),
}
