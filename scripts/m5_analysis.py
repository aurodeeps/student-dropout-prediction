"""Milestone 5, step 2: uncertainty, ablation, sensitivity, robustness and error analysis.

    python scripts/m5_analysis.py

Uses ONLY the development rows (train + validation = 3,760 students) and the train/validation split for fit-then-check
experiments. The TEST set is never used here. Hyperparameters come from results/m5_best_params.json (step 1).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split)

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5, CATEGORICAL
from src.experiments import MODELS, LABELS, SEED, pipeline

cfg = json.loads((ROOT / "configs" / "baseline.json").read_text()); assert file_md5() == cfg["dataset_md5"]
best = json.loads((ROOT / "results" / "m5_best_params.json").read_text())
tune = pd.read_csv(ROOT / "results" / "m5_tuning_records.csv")
df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df); fs = feature_sets(df)
Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva]); CL = ["Dropout", "Enrolled", "Graduate"]
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED); rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
R = ROOT / "results"; F = ROOT / "figures"; t00 = time.time()
def P(m, fname, **over): return pipeline(m, fs[fname], SEED, **{**best[f"{fname}|{m}"], **over})
def log(*a): print(f"[{time.time()-t00:5.0f}s]", *a, flush=True)

# selected model = highest tuning-CV macro-F1 on the EARLY set (rule fixed before the analysis)
top = tune[tune.feature_set == "early"].groupby("model").cv_macro_F1_mean.max().sort_values(ascending=False)
SELECTED = top.index[0]; SECOND = top.index[1]; log("selected", SELECTED, "second", SECOND, top.round(4).to_dict())

# ---- C: repeated CV (5-fold x 3 repeats) on development rows, all models, both feature sets ----------------------------
rows = []
for fname in ["early", "full"]:
    cols = fs[fname]
    for m in ["majority", "stratified_random"] + MODELS:
        est = pipeline(m, cols, SEED) if m in ("majority", "stratified_random") else P(m, fname)
        cvr = cross_validate(est, Xdev[cols], ydev, scoring=["f1_macro", "accuracy"], cv=rcv, n_jobs=1)
        for k, (a, b) in enumerate(zip(cvr["test_f1_macro"], cvr["test_accuracy"])):
            rows.append({"run_id": f"C-{fname}-{m}", "feature_set": fname, "model": m, "fold": k, "macro_F1": a, "accuracy": b})
        log("CV", fname, m, round(np.mean(cvr["test_f1_macro"]), 4), round(np.std(cvr["test_f1_macro"]), 4))
pd.DataFrame(rows).round(5).to_csv(R / "m5_cv_repeated_raw.csv", index=False)

# ---- B: ablations (early features), selected and second model -------------------------------------------------------------
groups = {"Full early set (30)": [],
          "- financial (debtor, tuition up to date, scholarship)": ["Debtor", "Tuition fees up to date", "Scholarship holder"],
          "- macroeconomic (unemployment, inflation, GDP)": ["Unemployment rate", "Inflation rate", "GDP"],
          "- financial and macroeconomic": ["Debtor", "Tuition fees up to date", "Scholarship holder", "Unemployment rate", "Inflation rate", "GDP"],
          "- personal attributes (gender, age, marital, nationality, international, displaced, special needs)":
              ["Gender", "Age at enrollment", "Marital status", "Nacionality", "International", "Displaced", "Educational special needs"],
          "- parental background (qualification, occupation)": ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]}
AB_MODELS = [SELECTED, SECOND] + (["logreg"] if "logreg" not in (SELECTED, SECOND) else [])
ab = []
for m in AB_MODELS:
    base = None
    for name, drop in groups.items():
        cols = [c for c in fs["early"] if c not in drop]
        cvr = cross_validate(pipeline(m, cols, SEED, **best[f"early|{m}"]), Xdev[cols], ydev, scoring="f1_macro", cv=rcv, n_jobs=1)["test_score"]
        base = cvr if base is None else base
        ab.append({"run_id": f"A-{m}-{len(ab)%len(groups)+1}", "model": m, "variant": name, "n_features": len(cols), "macro_F1_mean": cvr.mean(), "macro_F1_sd": cvr.std(),
                   "delta_vs_full": (cvr - base).mean(), "delta_sd": (cvr - base).std()})
        log("ablation", m, name[:30], round(cvr.mean(), 4), round((cvr - base).mean(), 4))
ab = pd.DataFrame(ab); ab.round(5).to_csv(R / "m5_ablation.csv", index=False)

# ---- S: sensitivity ---------------------------------------------------------------------------------------------------------
cols = fs["early"]; sens = []
for cw in [None, "balanced"]:
    for C in np.logspace(-3, 2, 11):
        s = cross_validate(pipeline("logreg", cols, SEED, C=float(C), class_weight=cw), Xdev[cols], ydev, scoring="f1_macro", cv=cv5, n_jobs=1)["test_score"]
        sens.append({"run_id": f"S-C-{len(sens)+1:02d}", "study": "logreg_C", "class_weight": str(cw), "value": float(C), "macro_F1_mean": s.mean(), "macro_F1_sd": s.std()})
pd.DataFrame(sens).round(5).to_csv(R / "m5_sensitivity_C.csv", index=False); log("sensitivity C done")
cwr = []
for m in MODELS:
    for cw in [None, "balanced"]:
        p = {**best[f"early|{m}"], "class_weight": cw}
        pred = cross_val_predict(pipeline(m, cols, SEED, **p), Xdev[cols], ydev, cv=cv5)
        rec = recall_score(ydev, pred, labels=CL, average=None)
        cwr.append({"run_id": f"S-CW-{m}-{cw}", "model": m, "class_weight": str(cw), "macro_F1": f1_score(ydev, pred, average="macro"), "recall_Dropout": rec[0], "recall_Enrolled": rec[1], "recall_Graduate": rec[2]})
    log("class weight", m)
pd.DataFrame(cwr).round(4).to_csv(R / "m5_sensitivity_class_weight.csv", index=False)
proba = cross_val_predict(P("logreg", "early"), Xdev[cols], ydev, cv=cv5, method="predict_proba"); boost = []
for f in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    pred = np.array(CL)[np.argmax(proba * np.array([1, f, 1]), axis=1)]; rec = recall_score(ydev, pred, labels=CL, average=None)
    boost.append({"run_id": f"S-BOOST-{f}", "enrolled_factor": f, "macro_F1": f1_score(ydev, pred, average="macro"), "recall_Dropout": rec[0], "recall_Enrolled": rec[1], "recall_Graduate": rec[2]})
boost = pd.DataFrame(boost); boost.round(4).to_csv(R / "m5_sensitivity_enrolled_boost.csv", index=False); log("boost done")

# ---- R: robustness (fit on TRAIN, evaluate on VALIDATION) ----------------------------------------------------------------
num = [c for c in cols if c not in CATEGORICAL]; med = Xtr[cols].median(); mode = Xtr[cols].mode().iloc[0]
def corrupt(X, rate, rng):
    X = X.copy()
    for c in cols:
        mask = rng.random(len(X)) < rate; X.loc[mask, c] = med[c] if c in num else mode[c]
    return X
rob = []; fitted = {m: P(m, "early").fit(Xtr[cols], ytr) for m in MODELS}
for m, est in fitted.items():
    for rate in [0.0, 0.1, 0.2, 0.3]:
        sc = [f1_score(yva, est.predict(corrupt(Xva[cols], rate, np.random.default_rng(100 + d)) if rate else Xva[cols]), average="macro") for d in range(5 if rate else 1)]
        rob.append({"run_id": f"R-miss-{m}-{rate}", "study": "missing_at_prediction", "model": m, "level": rate, "macro_F1_mean": np.mean(sc), "macro_F1_sd": np.std(sc)})
    for frac in [0.25, 0.5, 1.0]:
        sc = []
        for d in range(3 if frac < 1 else 1):
            sub = Xtr[cols] if frac == 1.0 else train_test_split(Xtr[cols], train_size=frac, stratify=ytr, random_state=d)[0]
            sc.append(f1_score(yva, P(m, "early").fit(sub, ytr.loc[sub.index]).predict(Xva[cols]), average="macro"))
        rob.append({"run_id": f"R-size-{m}-{frac}", "study": "training_size", "model": m, "level": frac, "macro_F1_mean": np.mean(sc), "macro_F1_sd": np.std(sc)})
    log("robustness", m)
rob = pd.DataFrame(rob); rob.round(4).to_csv(R / "m5_robustness.csv", index=False)

# ---- E: error analysis with out-of-fold predictions of the selected model -----------------------------------------------
oof = pd.Series(cross_val_predict(P(SELECTED, "early"), Xdev[cols], ydev, cv=cv5), index=Xdev.index)
cm = pd.DataFrame(confusion_matrix(ydev, oof, labels=CL), index=[f"true {c}" for c in CL], columns=[f"pred {c}" for c in CL]); cm.to_csv(R / "m5_oof_confusion.csv")
feat = ["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Tuition fees up to date", "Debtor", "Scholarship holder", "Age at enrollment", "Admission grade"]
prof = []
for t, p_ in [("Enrolled", "Enrolled"), ("Enrolled", "Graduate"), ("Enrolled", "Dropout"), ("Dropout", "Graduate"), ("Graduate", "Dropout"), ("Dropout", "Dropout"), ("Graduate", "Graduate")]:
    mk = (ydev == t) & (oof == p_); prof.append({"true": t, "predicted": p_, "n": int(mk.sum()), **Xdev.loc[mk, feat].mean().round(2).to_dict()})
pd.DataFrame(prof).to_csv(R / "m5_error_profiles.csv", index=False)
binv = []
for m in [SELECTED, "logreg"]:
    pr = cross_val_predict(P(m, "early"), Xdev[cols], ydev, cv=cv5, method="predict_proba")[:, 0]; y_bin = (ydev == "Dropout").astype(int)
    binv.append({"run_id": f"E-bin-{m}", "model": m, "roc_auc_dropout_vs_rest": roc_auc_score(y_bin, pr), "pr_auc_dropout_vs_rest": average_precision_score(y_bin, pr), "dropout_prevalence": y_bin.mean()})
pd.DataFrame(binv).round(4).to_csv(R / "m5_dropout_vs_rest.csv", index=False); log("error analysis done")

# ---- figures -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2)); w = 0.8 / len(AB_MODELS); v = [k for k in groups][1:]
for j, m in enumerate(AB_MODELS):
    d = ab[(ab.model == m) & (ab.variant != "Full early set (30)")]
    ax.barh(np.arange(len(v)) + (j - (len(AB_MODELS) - 1) / 2) * w, d.delta_vs_full, w, xerr=d.delta_sd, label=LABELS[m], color=["#4c72b0", "#dd8452", "#55a868"][j])
ax.set_yticks(range(len(v))); ax.set_yticklabels([x.split(" (")[0] for x in v], fontsize=7); ax.axvline(0, color="k", lw=.8); ax.invert_yaxis()
ax.set_xlabel("change in CV macro-F1 vs. full early set (paired, +/- sd over 15 folds)", fontsize=8); ax.legend(fontsize=7, loc="lower left"); ax.set_title("Ablation: removing feature groups (early features)", fontsize=9)
plt.tight_layout(); plt.savefig(F / "m5_ablation.png", dpi=170); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3)); s = pd.DataFrame(sens)
for cw, c in [("None", "#4c72b0"), ("balanced", "#c44e52")]:
    d = s[s.class_weight == cw]; ax[0].semilogx(d.value, d.macro_F1_mean, "o-", color=c, label=f"class_weight={cw}"); ax[0].fill_between(d.value, d.macro_F1_mean - d.macro_F1_sd, d.macro_F1_mean + d.macro_F1_sd, color=c, alpha=.12)
ax[0].set_xlabel("C (inverse regularisation)", fontsize=8); ax[0].set_ylabel("CV macro-F1", fontsize=8); ax[0].legend(fontsize=7); ax[0].set_title("Logistic regression: sensitivity to C", fontsize=9)
for c, col in [("recall_Enrolled", "#dd8452"), ("recall_Dropout", "#c44e52"), ("macro_F1", "#4c72b0")]: ax[1].plot(boost.enrolled_factor, boost[c], "o-", label=c.replace("_", " "), color=col)
ax[1].set_xlabel("multiplier on Enrolled probability", fontsize=8); ax[1].legend(fontsize=7); ax[1].set_title("Boosting Enrolled (OOF, logistic regression)", fontsize=9)
for m in MODELS:
    d = rob[(rob.study == "missing_at_prediction") & (rob.model == m)]; ax[2].errorbar(d.level, d.macro_F1_mean, d.macro_F1_sd, marker="o", label=LABELS[m], capsize=2)
ax[2].set_xlabel("share of feature values missing at prediction", fontsize=8); ax[2].set_ylabel("validation macro-F1", fontsize=8); ax[2].legend(fontsize=7); ax[2].set_title("Robustness to missing values", fontsize=9)
plt.tight_layout(); plt.savefig(F / "m5_sensitivity_robustness.png", dpi=170); plt.close()
json.dump({"selected": SELECTED, "second": SECOND, "tuning_cv_top": top.round(4).to_dict(), "runtime_seconds": round(time.time() - t00, 1), "test_set_used": False}, open(R / "m5_analysis_log.json", "w"), indent=2)
log("ALL DONE")
