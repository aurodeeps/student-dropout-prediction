"""Milestone 5, step 3: the single, confirmatory evaluation on the held-out TEST set.

    python scripts/m5_final_test.py --confirm-final

Run ONLY after tuning and all analyses are finished. Hyperparameters are frozen in results/m5_best_params.json (its SHA-256
is written to the log). Final models are trained on train + validation (3,760 rows) and evaluated once on the 664 test rows.
A lock file stops accidental re-runs; the test set must not be used to change any decision.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5
from src.experiments import MODELS, LABELS, SEED, pipeline

ap = argparse.ArgumentParser(); ap.add_argument("--confirm-final", action="store_true"); ap.add_argument("--rerun-and-admit-test-reuse", action="store_true"); a = ap.parse_args()
LOCK = ROOT / "results" / "m5_FINAL_TEST_DONE.lock"
if not a.confirm_final: sys.exit("Refusing to run: pass --confirm-final (this uses the test set).")
if LOCK.exists() and not a.rerun_and_admit_test_reuse: sys.exit(f"Test set already used ({LOCK.read_text().strip()}). Re-running counts as reuse; see docs/EXPERIMENT_LOG.md.")
cfg = json.loads((ROOT / "configs" / "baseline.json").read_text()); assert file_md5() == cfg["dataset_md5"]
best_path = ROOT / "results" / "m5_best_params.json"; best = json.loads(best_path.read_text()); alog = json.loads((ROOT / "results" / "m5_analysis_log.json").read_text())
SELECTED = alog["selected"]; df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df); fs = feature_sets(df)
Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva]); CL = ["Dropout", "Enrolled", "Graduate"]; SEEDS = [0, 1, 2, 3, 4]; t0 = time.time()
R, F = ROOT / "results", ROOT / "figures"; rows, preds, probs = [], {}, {}
for fname in ["early", "full"]:
    cols = fs[fname]
    for m in ["majority", "stratified_random"] + MODELS:
        for s in SEEDS:
            est = pipeline(m, cols, s) if m in ("majority", "stratified_random") else pipeline(m, cols, s, **best[f"{fname}|{m}"])
            t1 = time.time(); est.fit(Xdev[cols], ydev); fit_s = time.time() - t1; p = est.predict(Xte[cols]); rec = recall_score(yte, p, labels=CL, average=None)
            rows.append({"run_id": f"F-{fname}-{m}-s{s}", "feature_set": fname, "model": m, "seed": s, "macro_F1": f1_score(yte, p, average="macro"), "accuracy": accuracy_score(yte, p),
                         "recall_Dropout": rec[0], "recall_Enrolled": rec[1], "recall_Graduate": rec[2], "fit_seconds": fit_s})
            if s == 0:
                preds[(fname, m)] = p
                if hasattr(est, "predict_proba"): probs[(fname, m)] = est.predict_proba(Xte[cols])[:, 0]
    print("done", fname, f"{time.time()-t0:.0f}s", flush=True)
raw = pd.DataFrame(rows); raw.round(5).to_csv(R / "m5_final_test_per_seed.csv", index=False)

# bootstrap uncertainty over test rows (same resamples for every model -> paired differences)
rng = np.random.default_rng(SEED); n = len(yte); B = 1000; idx = rng.integers(0, n, size=(B, n)); yv = yte.to_numpy(); tab, diffs = [], []
boot = {k: np.array([f1_score(yv[i], p[i], average="macro") for i in idx]) for k, p in preds.items()}
for fname in ["early", "full"]:
    ref = boot[(fname, SELECTED)]
    for m in ["majority", "stratified_random"] + MODELS:
        d = raw[(raw.feature_set == fname) & (raw.model == m)]; b = boot[(fname, m)]; pb = probs.get((fname, m))
        yb = (yte == "Dropout").astype(int)
        tab.append({"feature_set": fname, "model": LABELS[m], "macro_F1_mean": d.macro_F1.mean(), "macro_F1_sd_over_seeds": d.macro_F1.std(ddof=1), "ci95_low": np.percentile(b, 2.5), "ci95_high": np.percentile(b, 97.5),
                    "accuracy_mean": d.accuracy.mean(), "recall_Dropout": d.recall_Dropout.mean(), "recall_Enrolled": d.recall_Enrolled.mean(), "recall_Graduate": d.recall_Graduate.mean(),
                    "roc_auc_dropout_vs_rest": roc_auc_score(yb, pb) if pb is not None else np.nan, "pr_auc_dropout_vs_rest": average_precision_score(yb, pb) if pb is not None else np.nan, "fit_seconds_mean": d.fit_seconds.mean()})
        if m != SELECTED:
            dd = ref - b; diffs.append({"feature_set": fname, "comparison": f"{LABELS[SELECTED]} minus {LABELS[m]}", "mean_diff": dd.mean(), "ci95_low": np.percentile(dd, 2.5), "ci95_high": np.percentile(dd, 97.5), "ci_excludes_zero": bool(np.percentile(dd, 2.5) > 0 or np.percentile(dd, 97.5) < 0)})
tab = pd.DataFrame(tab); tab.round(4).to_csv(R / "m5_final_test_table.csv", index=False); pd.DataFrame(diffs).round(4).to_csv(R / "m5_final_test_paired_bootstrap.csv", index=False)

# subgroups for the selected model (same frozen predictions; descriptive only)
p = pd.Series(preds[("early", SELECTED)], index=Xte.index); g = {"All": Xte["Gender"] >= 0, "Female": Xte["Gender"] == 0, "Male": Xte["Gender"] == 1, "No scholarship": Xte["Scholarship holder"] == 0, "Scholarship holder": Xte["Scholarship holder"] == 1,
    "Fees up to date": Xte["Tuition fees up to date"] == 1, "Fees NOT up to date": Xte["Tuition fees up to date"] == 0, "Age < 25": Xte["Age at enrollment"] < 25, "Age >= 25": Xte["Age at enrollment"] >= 25}
sub = [{"group": k, "n": int(m.sum()), "dropout_rate": (yte[m] == "Dropout").mean(), "macro_F1": f1_score(yte[m], p[m], average="macro", labels=CL, zero_division=0),
        "dropout_recall": recall_score(yte[m], p[m], labels=["Dropout"], average=None, zero_division=0)[0]} for k, m in g.items()]
pd.DataFrame(sub).round(4).to_csv(R / "m5_final_test_subgroups.csv", index=False)

# figures
for fname in ["early", "full"]:
    d = tab[tab.feature_set == fname]; fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(len(d))
    ax.bar(x, d.macro_F1_mean, yerr=[d.macro_F1_mean - d.ci95_low, d.ci95_high - d.macro_F1_mean], capsize=3, color=["#9aa5b1", "#9aa5b1", "#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    ax.set_xticks(x); ax.set_xticklabels(d.model, fontsize=7.5, rotation=15); ax.set_ylabel("test macro-F1 (95% bootstrap CI)", fontsize=8); ax.set_ylim(0, 0.85)
    for xi, v in zip(x, d.macro_F1_mean): ax.text(xi, v + 0.045, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_title(f"Final test results, {fname} features (trained on train + validation)", fontsize=9); plt.tight_layout(); plt.savefig(F / f"m5_main_figure_{fname}.png", dpi=170); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
for a_, m in zip(ax, MODELS):
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[("early", m)], labels=CL), display_labels=["Drop", "Enr", "Grad"]).plot(ax=a_, cmap="Blues", colorbar=False); a_.set_title(LABELS[m], fontsize=8)
plt.tight_layout(); plt.savefig(F / "m5_test_confusion_matrices.png", dpi=170); plt.close()
stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
json.dump({"selected_model": SELECTED, "best_params_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(), "trained_on_rows": len(Xdev), "test_rows": len(Xte), "seeds": SEEDS, "bootstrap_resamples": B,
           "software": __import__("sklearn").__version__, "run_utc": stamp, "runtime_seconds": round(time.time() - t0, 1)}, open(R / "m5_final_test_log.json", "w"), indent=2)
LOCK.write_text(f"final test evaluation run at {stamp}\n"); print(tab.round(3).to_string(index=False)); print(pd.DataFrame(diffs).round(3).to_string(index=False))
