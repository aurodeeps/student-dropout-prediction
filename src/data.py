"""Data loading and the fixed, documented train/validation/test split (Milestone 2)."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "split_indices.json"
TARGET = "Target"

CATEGORICAL = ["Marital status", "Application mode", "Course", "Previous qualification",
               "Nacionality", "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation"]
SECOND_SEM = lambda cols: [c for c in cols if "2nd sem" in c]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]  # one header has a trailing tab
    return df


def feature_sets(df):
    """'early' = information available at/after the end of semester 1; 'full' = after semester 2."""
    X_cols = [c for c in df.columns if c != TARGET]
    early = [c for c in X_cols if "2nd sem" not in c]
    return {"early": early, "full": X_cols}


def make_splits(df, seed=SEED, save=True):
    """Stratified 70/15/15 split. Test set is touched only for the final evaluation."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    if save:
        SPLIT_PATH.write_text(json.dumps({"seed": seed, "train": X_tr.index.tolist(),
                                          "val": X_va.index.tolist(), "test": X_te.index.tolist()}))
    return X_tr, X_va, X_te, y_tr, y_va, y_te
