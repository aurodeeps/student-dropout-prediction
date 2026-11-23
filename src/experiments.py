"""Shared definitions for Milestone 5: models, search spaces and the equal tuning budget."""
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.models import make_pipeline

SEED = 42
N_ITER = 15            # identical tuning budget for every model: 15 random configurations x 5-fold CV
MODELS = ["logreg", "tree", "rf", "hgb"]
LABELS = {"majority": "Majority class", "stratified_random": "Stratified random", "logreg": "Logistic regression",
          "tree": "Decision tree", "rf": "Random forest", "hgb": "Gradient boosting"}

SPACES = {
    "logreg": {"model__C": loguniform(1e-3, 1e2), "model__class_weight": [None, "balanced"]},
    "tree": {"model__max_depth": [3, 4, 5, 6, 8, 10, 12, None], "model__min_samples_leaf": [1, 2, 5, 10, 20, 40],
             "model__criterion": ["gini", "entropy"], "model__class_weight": [None, "balanced"]},
    "rf": {"model__max_depth": [None, 8, 12, 16], "model__min_samples_leaf": [1, 2, 3, 5, 8],
           "model__max_features": ["sqrt", 0.2, 0.4], "model__class_weight": [None, "balanced", "balanced_subsample"]},
    "hgb": {"model__learning_rate": loguniform(0.02, 0.3), "model__max_depth": [2, 3, 4, 6, None],
            "model__max_leaf_nodes": [8, 15, 31], "model__l2_regularization": loguniform(1e-3, 10),
            "model__min_samples_leaf": [10, 20, 40], "model__class_weight": [None, "balanced"]},
}


def estimator(name, seed=SEED, **params):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified_random":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=3000, **params)
    if name == "tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=10, random_state=seed, **params)
    raise ValueError(name)


def pipeline(name, cols, seed=SEED, **params):
    """params are given without the 'model__' prefix."""
    return make_pipeline(cols, estimator(name, seed, **params), dense=(name == "hgb"))
