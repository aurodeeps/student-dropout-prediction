"""Milestone 4: preliminary tuning of the decision tree and random forest, compared with the baselines.

    python scripts/run_models.py

Same split, preprocessing, metric and CV protocol as run_baseline.py. Small preliminary grids (full tuning is Milestone 5). CV: 5-fold x 3 repeats for logistic regression and tree, 5-fold x 1 for the forest.
Uses TRAIN (CV) and VALIDATION only; the test set is never evaluated.
Outputs: results/m4_models_table.csv, results/m4_feature_importance.csv, results/m4_run_log.json,
         figures/m4_learning_curves.png, figures/m4_feature_importance.png, figures/m4_confusion_matrices.png
"""
import json, sys, time, platform
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5
from src.models import make_pipeline

SEED = 42; CLASSES = ["Dropout", "Enrolled", "Graduate"]; t0 = time.time()
cfg = json.loads((ROOT / "configs" / "baseline.json").read_text()); assert file_md5() == cfg["dataset_md5"]
df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df); fs = feature_sets(df)
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

grids = {  # preliminary search spaces: 5 / 15 / 6 configurations (forest kept small: the run machine has one CPU core)
    "Logistic regression": (LogisticRegression(max_iter=3000), {"model__C": [0.01, 0.1, 1, 10, 100]}),
    "Decision tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [3, 5, 7, 10, None], "model__min_samples_leaf": [1, 5, 20]}),
    "Random forest": (RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
                      {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 3, 5]}),
}
rows, fitted = [], {}
for fname in ["early", "full"]:
    cols = fs[fname]
    for mname, (est, grid) in grids.items():
        gs = GridSearchCV(make_pipeline(cols, est), grid, scoring="f1_macro", cv=cv5, n_jobs=1)
        t1 = time.time(); gs.fit(Xtr[cols], ytr); search_s = time.time() - t1
        best = gs.best_estimator_; fitted[(fname, mname)] = (best, cols)
        t2 = time.time(); best.fit(Xtr[cols], ytr); fit_s = time.time() - t2
        ptr, pva = best.predict(Xtr[cols]), best.predict(Xva[cols])
        cv = cross_val_score(best, Xtr[cols], ytr, scoring="f1_macro", cv=(cv5 if mname == "Random forest" else rcv), n_jobs=1)
        rec = recall_score(yva, pva, labels=CLASSES, average=None)
        rows.append({"feature_set": fname, "model": mname, "n_configs": len(gs.cv_results_["params"]), "best_params": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()}),
                     "train_macro_F1": f1_score(ytr, ptr, average="macro"), "val_macro_F1": f1_score(yva, pva, average="macro"), "val_accuracy": accuracy_score(yva, pva),
                     "val_recall_Dropout": rec[0], "val_recall_Enrolled": rec[1], "val_recall_Graduate": rec[2],
                     "cv_macro_F1_mean": cv.mean(), "cv_macro_F1_sd": cv.std(), "fit_seconds": fit_s, "search_seconds": search_s,
                     "n_encoded_features": int(best.named_steps["prep"].transform(Xva[cols].head(2)).shape[1])})
        print(fname, mname, round(rows[-1]["val_macro_F1"], 3), "cv", round(cv.mean(), 3), "train", round(rows[-1]["train_macro_F1"], 3), flush=True)
tab = pd.DataFrame(rows); tab.round(4).to_csv(ROOT / "results" / "m4_models_table.csv", index=False)

# learning curves (early set): bias/variance evidence
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for a, mname in zip(ax, grids):
    best, cols = fitted[("early", mname)]
    sizes, trs, vas = learning_curve(best, Xtr[cols], ytr, train_sizes=([0.4, 0.7, 1.0] if mname == "Random forest" else [0.2, 0.4, 0.6, 0.8, 1.0]), cv=cv5, scoring="f1_macro", n_jobs=1, random_state=SEED, shuffle=True)
    a.plot(sizes, trs.mean(1), "o-", color="#4c72b0", label="train"); a.plot(sizes, vas.mean(1), "o-", color="#c44e52", label="CV validation")
    a.fill_between(sizes, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), color="#c44e52", alpha=.15); a.set_title(mname, fontsize=9); a.set_xlabel("training rows", fontsize=8)
ax[0].set_ylabel("macro-F1"); ax[0].legend(fontsize=7); plt.suptitle("Learning curves (early features, tuned settings)", fontsize=10); plt.tight_layout(); plt.savefig(ROOT / "figures" / "m4_learning_curves.png", dpi=170); plt.close()

# permutation importance of the tuned random forest on the VALIDATION set (original, un-encoded features)
best, cols = fitted[("early", "Random forest")]
pi = permutation_importance(best, Xva[cols], yva, scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("importance", ascending=False)
imp.round(4).to_csv(ROOT / "results" / "m4_feature_importance.csv", index=False)
top = imp.head(10).iloc[::-1]; fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.barh(top["feature"], top["importance"], xerr=top["sd"], color="#4c72b0")
ax.set_xlabel("drop in validation macro-F1 when shuffled"); ax.set_title("Random forest (early features): top 10 permutation importances", fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); plt.savefig(ROOT / "figures" / "m4_feature_importance.png", dpi=170); plt.close()

# confusion matrices, early features, validation
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
for a, mname in zip(ax, grids):
    best, cols = fitted[("early", mname)]; cm = confusion_matrix(yva, best.predict(Xva[cols]), labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=["Drop", "Enr", "Grad"]).plot(ax=a, cmap="Blues", colorbar=False); a.set_title(mname, fontsize=9)
plt.tight_layout(); plt.savefig(ROOT / "figures" / "m4_confusion_matrices.png", dpi=170); plt.close()

(ROOT / "results" / "m4_run_log.json").write_text(json.dumps({"script": "scripts/run_models.py", "seed": SEED, "dataset_md5": file_md5(), "test_set_used": False,
    "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__}, "runtime_seconds": round(time.time() - t0, 1)}, indent=2))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(tab[["feature_set", "model", "n_configs", "best_params", "train_macro_F1", "val_macro_F1", "val_accuracy", "val_recall_Dropout", "val_recall_Enrolled", "cv_macro_F1_mean", "cv_macro_F1_sd", "fit_seconds", "n_encoded_features"]].round(3).to_string(index=False))
print(imp.head(10).round(4).to_string(index=False)); print("runtime", round(time.time() - t0, 1))
