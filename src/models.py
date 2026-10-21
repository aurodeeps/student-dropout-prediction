"""Preprocessing + model pipelines shared by every experiment (same preprocessing for all models)."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import CATEGORICAL


def make_pipeline(cols, model):
    """One-hot encode integer-coded categoricals, standardize the rest; fit on training data only."""
    cat = [c for c in CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler(), num)])
    return Pipeline([("prep", prep), ("model", model)])


def build_model(name, seed=42, **kw):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified_random":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=kw.get("max_iter", 3000), C=kw.get("C", 1.0))
    if name == "tree_depth5":
        return DecisionTreeClassifier(max_depth=5, random_state=seed)
    raise ValueError(name)
