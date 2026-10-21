"""Milestone 3: regenerate the baseline table end to end.

    python scripts/run_baseline.py

Outputs (all under results/ and figures/): baseline_table.csv/.md, reproduction_check.csv,
discrepancy_probes.csv, run_log.json, confusion_matrix_logreg_early.png.
Only the TRAIN and VALIDATION sets are used. The test set is never evaluated here.
"""
import json, sys, time, warnings, platform
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn, scipy
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5, TARGET, SPLIT_PATH
from src.models import make_pipeline, build_model

cfg = json.loads((ROOT / "configs" / "baseline.json").read_text())
SEED = cfg["seed"]; CLASSES = ["Dropout", "Enrolled", "Graduate"]
t0 = time.time()

# ---- 1. data version + fixed split -------------------------------------------------------------
md5 = file_md5()
assert md5 == cfg["dataset_md5"], f"data.csv differs from the approved version ({md5})"
df = load_data()
Xtr, Xva, Xte, ytr, yva, yte = make_splits(df, seed=SEED)
if SPLIT_PATH.exists():                                  # the stored split must match what we regenerate
    stored = json.loads(SPLIT_PATH.read_text())
    assert stored["train"] == Xtr.index.tolist() and stored["val"] == Xva.index.tolist() \
        and stored["test"] == Xte.index.tolist(), "stored split differs from regenerated split"
else:
    make_splits(df, seed=SEED, save=True)
fs = feature_sets(df)
print(f"data ok (md5 {md5[:8]}...), split train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)} (test untouched)")

def val_metrics(pipe, cols):
    pipe.fit(Xtr[cols], ytr); p = pipe.predict(Xva[cols])
    out = {"macro_F1": f1_score(yva, p, average="macro"), "accuracy": accuracy_score(yva, p)}
    for c, f, r in zip(CLASSES, f1_score(yva, p, average=None, labels=CLASSES), recall_score(yva, p, average=None, labels=CLASSES)):
        out[f"F1_{c}"] = f; out[f"recall_{c}"] = r
    return out, p

rskf = RepeatedStratifiedKFold(n_splits=cfg["cv"]["folds"], n_repeats=cfg["cv"]["repeats"], random_state=SEED)
rows, tuned_C, preds = [], {}, {}
for fname in cfg["feature_sets"]:
    cols = fs[fname]
    # ---- 2. naive benchmarks and course-level baselines ----------------------------------------
    specs = [("Majority class", "majority", {}), ("Stratified random (seed 42)", "stratified_random", {}),
             ("Logistic regression (C=1)", "logreg", {}), ("Decision tree (depth 5)", "tree_depth5", {})]
    # tuned logistic regression: C chosen by 5-fold CV on TRAIN only
    gs = GridSearchCV(make_pipeline(cols, build_model("logreg")), {"model__C": cfg["logreg_C_grid"]},
                      scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    gs.fit(Xtr[cols], ytr); bestC = gs.best_params_["model__C"]; tuned_C[fname] = bestC
    specs.insert(3, (f"Logistic regression (tuned, C={bestC:g})", "logreg", {"C": bestC}))
    for label, key, kw in specs:
        pipe = make_pipeline(cols, build_model(key, SEED, **kw))
        m, p = val_metrics(pipe, cols)
        cv = cross_val_score(make_pipeline(cols, build_model(key, SEED, **kw)), Xtr[cols], ytr, scoring="f1_macro", cv=rskf)
        rows.append({"feature_set": fname, "n_features": len(cols), "model": label, "key": key, **m,
                     "cv_macro_F1_mean": cv.mean(), "cv_macro_F1_sd": cv.std()})
        preds[(fname, key, tuple(sorted(kw.items())))] = p
tab = pd.DataFrame(rows)
tab.round(4).to_csv(ROOT / "results" / "baseline_table.csv", index=False)

# markdown version for the repo/README
def md_table(d, cols):
    h = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return h + "\n".join("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for _, r in d.iterrows())
(ROOT / "results" / "baseline_table.md").write_text(
    "# Baseline table (validation set; CV on training set)\n\n" +
    md_table(tab, ["feature_set", "model", "macro_F1", "accuracy", "recall_Dropout", "recall_Enrolled", "recall_Graduate", "cv_macro_F1_mean", "cv_macro_F1_sd"]) + "\n")

# ---- 3. reproduction check against the values reported in the Milestone 2 memo -------------------
rep = []
for _, r in tab.iterrows():
    exp = cfg["expected_from_M2_memo"].get(r["key"], {}).get(r["feature_set"])
    if exp and not r["model"].startswith("Logistic regression (tuned"):
        rep.append({"feature_set": r["feature_set"], "model": r["model"], "expected_macro_F1": exp[0], "reproduced_macro_F1": round(r["macro_F1"], 3),
                    "delta_macro_F1": round(r["macro_F1"] - exp[0], 3), "expected_acc": exp[1], "reproduced_acc": round(r["accuracy"], 3),
                    "delta_acc": round(r["accuracy"] - exp[1], 3)})
rep = pd.DataFrame(rep); rep.to_csv(ROOT / "results" / "reproduction_check.csv", index=False)

# ---- 4. discrepancy probes (what changes the number, and by how much?) --------------------------
probe = []
cols = fs["early"]; base = tab[(tab.feature_set == "early") & (tab.key == "logreg") & ((tab.model == "Logistic regression (C=1)"))].iloc[0]["macro_F1"]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m, _ = val_metrics(make_pipeline(cols, build_model("logreg", max_iter=100)), cols)
    nconv = sum("converge" in str(x.message).lower() for x in w)
probe.append({"probe": "LogReg max_iter=100 (sklearn default) instead of 3000", "macro_F1": m["macro_F1"], "delta_vs_pipeline": m["macro_F1"] - base, "note": f"{nconv} convergence warning(s)"})
# scaler fit on ALL rows (train+val+test) = a leakage mistake; we only measure the size of the effect
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.data import CATEGORICAL
cat = [c for c in CATEGORICAL if c in cols]; num = [c for c in cols if c not in cat]
ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)]).fit(df[cols])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=3000).fit(ct.transform(Xtr[cols]), ytr)
leak = f1_score(yva, lr.predict(ct.transform(Xva[cols])), average="macro")
probe.append({"probe": "Preprocessing fit on ALL rows (leakage, do not use)", "macro_F1": leak, "delta_vs_pipeline": leak - base, "note": "shows leakage effect is small here but is still a protocol error"})
sr = []
for s in cfg["stratified_random_seeds"]:
    sr.append(f1_score(yva, build_model("stratified_random", s).fit(Xtr[cols], ytr).predict(Xva[cols]), average="macro"))
probe.append({"probe": "Stratified random over 10 seeds (mean)", "macro_F1": float(np.mean(sr)), "delta_vs_pipeline": float(np.mean(sr)) - 0.313, "note": f"sd {np.std(sr):.3f}, range {min(sr):.3f}-{max(sr):.3f}; seed 42 gives 0.313"})
probe = pd.DataFrame(probe); probe.round(4).to_csv(ROOT / "results" / "discrepancy_probes.csv", index=False)

# ---- 5. confusion matrix figure (LR C=1, early set, validation) ---------------------------------
p = preds[("early", "logreg", ())]
cm = confusion_matrix(yva, p, labels=CLASSES)
fig, ax = plt.subplots(figsize=(4.2, 3.6)); ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Logistic regression (early set), validation"); plt.tight_layout(); plt.savefig(ROOT / "figures" / "confusion_matrix_logreg_early.png", dpi=170); plt.close()

# ---- 6. run log ----------------------------------------------------------------------------------
log = {"script": "scripts/run_baseline.py", "seed": SEED, "dataset_md5": md5, "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
       "test_set_used": False, "tuned_C_by_feature_set": tuned_C, "cv": cfg["cv"],
       "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
       "runtime_seconds": round(time.time() - t0, 1)}
(ROOT / "results" / "run_log.json").write_text(json.dumps(log, indent=2))

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\nBASELINE TABLE"); print(tab[["feature_set", "model", "macro_F1", "accuracy", "recall_Dropout", "recall_Enrolled", "recall_Graduate", "cv_macro_F1_mean", "cv_macro_F1_sd"]].round(3).to_string(index=False))
print("\nREPRODUCTION CHECK vs M2 memo"); print(rep.to_string(index=False))
print("\nDISCREPANCY PROBES"); print(probe.round(4).to_string(index=False))
print(f"\nruntime {log['runtime_seconds']}s")
