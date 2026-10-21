"""Subgroup view of the logistic-regression baseline (early features).

    python scripts/subgroup_baseline.py

Uses 5-fold out-of-fold predictions on TRAIN+VALIDATION rows only (the test set is never used), so every
student is predicted by a model that did not see them. Groups follow the codings in the dataset descriptor
(Realinho et al., 2022, Appendix A): Gender 1=male, 0=female; Daytime/evening attendance 1=daytime, 0=evening;
yes/no variables 1=yes. These are the same variables explored in the group EDA notebook.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets
from src.models import make_pipeline, build_model

df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df)
cols = feature_sets(df)["early"]
X = pd.concat([Xtr, Xva]); y = pd.concat([ytr, yva])
oof = pd.Series(cross_val_predict(make_pipeline(cols, build_model("logreg")), X[cols], y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=42)), index=X.index)
groups = {
    "All (train+val)": pd.Series(True, index=X.index),
    "Female": X["Gender"] == 0, "Male": X["Gender"] == 1,
    "Evening attendance": X["Daytime/evening attendance"] == 0, "Daytime attendance": X["Daytime/evening attendance"] == 1,
    "Scholarship holder": X["Scholarship holder"] == 1, "No scholarship": X["Scholarship holder"] == 0,
    "Tuition up to date": X["Tuition fees up to date"] == 1, "Tuition NOT up to date": X["Tuition fees up to date"] == 0,
    "Debtor": X["Debtor"] == 1, "Not debtor": X["Debtor"] == 0,
    "Displaced": X["Displaced"] == 1, "Not displaced": X["Displaced"] == 0,
    "Age >= 25 at enrollment": X["Age at enrollment"] >= 25, "Age < 25 at enrollment": X["Age at enrollment"] < 25,
    "International": X["International"] == 1,
}
rows = []
for name, m in groups.items():
    yt, yp = y[m], oof[m]
    rows.append({"group": name, "n": int(m.sum()), "dropout_rate": (yt == "Dropout").mean(),
                 "accuracy": (yt == yp).mean(), "macro_F1": f1_score(yt, yp, average="macro", labels=["Dropout", "Enrolled", "Graduate"], zero_division=0),
                 "dropout_recall": recall_score(yt, yp, labels=["Dropout"], average=None, zero_division=0)[0],
                 "dropout_precision": precision_score(yt, yp, labels=["Dropout"], average=None, zero_division=0)[0]})
out = pd.DataFrame(rows); out.round(4).to_csv(ROOT / "results" / "subgroup_baseline.csv", index=False)
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))

# figure: Dropout recall by group (shows which students the baseline misses)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pairs = [("Female", "Male"), ("Daytime attendance", "Evening attendance"), ("No scholarship", "Scholarship holder"),
         ("Tuition up to date", "Tuition NOT up to date"), ("Not debtor", "Debtor"), ("Not displaced", "Displaced"),
         ("Age < 25 at enrollment", "Age >= 25 at enrollment")]
o = out.set_index("group"); fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.38
for i, (a, b) in enumerate(pairs):
    ax.bar(i - w/2, o.loc[a, "dropout_recall"], w, color="#4c72b0", label="first group in label" if i == 0 else None)
    ax.bar(i + w/2, o.loc[b, "dropout_recall"], w, color="#c44e52", label="second group in label" if i == 0 else None)
short = ["Female\nvs male", "Daytime\nvs evening", "No scholarship\nvs scholarship", "Fees paid\nvs not paid", "Not debtor\nvs debtor", "Not displaced\nvs displaced", "Age < 25\nvs age >= 25"]
ax.set_xticks(range(len(pairs))); ax.set_xticklabels(short, fontsize=7)
ax.axhline(o.loc["All (train+val)", "dropout_recall"], color="k", ls="--", lw=0.8); ax.set_ylim(0, 1.05)
ax.set_ylabel("Dropout recall"); ax.set_title("Logistic regression (early features): Dropout recall by group, out-of-fold", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(ROOT / "figures" / "subgroup_dropout_recall.png", dpi=170)
