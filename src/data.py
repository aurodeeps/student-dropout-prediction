"""Data loading and the fixed, documented train/validation/test split."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA_PATH = ROOT / "data" / "data.csv"
SPLIT_PATH = ROOT / "data" / "split_indices.json"
TARGET = "Target"

CATEGORICAL = ["Marital status", "Application mode", "Course", "Previous qualification",
               "Nacionality", "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation"]


def file_md5(path=DATA_PATH):
    """MD5 of the file with line endings normalised to LF.

    The original download uses CRLF (raw MD5 ee29c9ee...), while Git on some setups stores LF (raw MD5 9b2b50f4...).
    The content is identical, so we checksum the LF-normalised bytes (9b2b50f41d0cefbbc3e8222d60032d91 for both).
    """
    return hashlib.md5(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]  # one header has a trailing tab
    return df


def feature_sets(df):
    """'early' = enrollment info + semester 1 (30 features); 'full' adds semester 2 (36 features)."""
    X_cols = [c for c in df.columns if c != TARGET]
    early = [c for c in X_cols if "2nd sem" not in c]
    return {"early": early, "full": X_cols}


def make_splits(df, seed=SEED, save=False):
    """Stratified 70/15/15 split. The test set is reserved for the final evaluation (Milestone 5/6)."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    if save:
        SPLIT_PATH.write_text(json.dumps({"seed": seed, "train": X_tr.index.tolist(),
                                          "val": X_va.index.tolist(), "test": X_te.index.tolist()}))
    return X_tr, X_va, X_te, y_tr, y_va, y_te
