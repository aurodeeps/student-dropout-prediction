"""Milestone 5: assemble the one-row-per-run experiment log (Excel + CSV) from the saved result files.

    python scripts/m5_build_log.py
"""
import json, sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
tune = pd.read_csv(R / "m5_tuning_records.csv"); cvr = pd.read_csv(R / "m5_cv_repeated_raw.csv"); ab = pd.read_csv(R / "m5_ablation.csv")
sc = pd.read_csv(R / "m5_sensitivity_C.csv"); cw = pd.read_csv(R / "m5_sensitivity_class_weight.csv"); bo = pd.read_csv(R / "m5_sensitivity_enrolled_boost.csv")
rob = pd.read_csv(R / "m5_robustness.csv"); binv = pd.read_csv(R / "m5_dropout_vs_rest.csv"); alog = json.loads((R / "m5_analysis_log.json").read_text())
final_path = R / "m5_final_test_per_seed.csv"; fin = pd.read_csv(final_path) if final_path.exists() else None
flog = json.loads((R / "m5_final_test_log.json").read_text()) if final_path.exists() else {}
SPLIT = "70/15/15, seed 42"; rows = []
def add(rid, date, stage, model, fs, hp, seed, split, val, test, rt, obs): rows.append([rid, date, stage, model, fs, hp, seed, split, val, test, rt, obs])
D = "2026-11-20"
best = tune.loc[tune.groupby(["feature_set", "model"]).cv_macro_F1_mean.idxmax(), "run_id"].tolist()
for _, r in tune.iterrows():
    add(r.run_id, D, "Tuning (random search, 5-fold CV on train)", r.model, r.feature_set, r.params, 42, SPLIT + " (train rows only)",
        f"{r.cv_macro_F1_mean:.4f} +/- {r.cv_macro_F1_sd:.4f}", "-", round(r.fit_seconds_mean, 2), "SELECTED (best CV macro-F1 for this model and feature set)" if r.run_id in best else f"train macro-F1 {r.train_macro_F1_mean:.3f}")
for (rid, m), g in cvr.groupby(["run_id", "model"]):
    add(rid, D, "Repeated CV (5x3, train+validation rows)", m, g.feature_set.iloc[0], "tuned settings from step 1" if m not in ("majority", "stratified_random") else "none", 42, "5-fold x 3 repeats on 3,760 development rows",
        f"{g.macro_F1.mean():.4f} +/- {g.macro_F1.std():.4f}", "-", "-", f"accuracy {g.accuracy.mean():.3f}")
for _, r in ab.iterrows():
    add(r.run_id, D, "Ablation (early features, 5x3 CV)", r.model, "early", r.variant, 42, "5-fold x 3 repeats on development rows", f"{r.macro_F1_mean:.4f} +/- {r.macro_F1_sd:.4f}", "-", "-", f"delta vs. full early set {r.delta_vs_full:+.4f} (paired sd {r.delta_sd:.4f})")
for _, r in sc.iterrows():
    add(r.run_id, D, "Sensitivity: logistic regression C", "logreg", "early", f"C={r.value:.4g}, class_weight={r.class_weight}", 42, "5-fold CV on development rows", f"{r.macro_F1_mean:.4f} +/- {r.macro_F1_sd:.4f}", "-", "-", "")
for _, r in cw.iterrows():
    add(r.run_id, D, "Sensitivity: class weighting", r.model, "early", f"class_weight={r.class_weight}", 42, "5-fold out-of-fold predictions", f"{r.macro_F1:.4f}", "-", "-", f"recall Dropout {r.recall_Dropout:.3f}, Enrolled {r.recall_Enrolled:.3f}, Graduate {r.recall_Graduate:.3f}")
for _, r in bo.iterrows():
    add(r.run_id, D, "Sensitivity: Enrolled probability multiplier", "logreg", "early", f"multiplier={r.enrolled_factor}", 42, "5-fold out-of-fold probabilities", f"{r.macro_F1:.4f}", "-", "-", f"recall Dropout {r.recall_Dropout:.3f}, Enrolled {r.recall_Enrolled:.3f}, Graduate {r.recall_Graduate:.3f}")
for _, r in rob.iterrows():
    add(r.run_id, D, f"Robustness: {r.study}", r.model, "early", f"level={r.level}", "5 draws" if r.study == "missing_at_prediction" and r.level > 0 else "3 draws" if r.study == "training_size" and r.level < 1 else 42, "fit on train, scored on validation", f"{r.macro_F1_mean:.4f} +/- {r.macro_F1_sd:.4f}", "-", "-", "")
for _, r in binv.iterrows():
    add(r.run_id, D, "Error analysis: Dropout vs. rest (out-of-fold)", r.model, "early", "tuned settings", 42, "5-fold out-of-fold probabilities", f"ROC-AUC {r.roc_auc_dropout_vs_rest:.3f}", "-", "-", f"PR-AUC {r.pr_auc_dropout_vs_rest:.3f} (prevalence {r.dropout_prevalence:.3f})")
if fin is not None:
    for _, r in fin.iterrows():
        add(r.run_id, flog.get("run_utc", D)[:10], "FINAL test evaluation (single use of the test set)", r.model, r.feature_set, "frozen tuned settings (m5_best_params.json)", int(r.seed), "train+validation fit, test 664 rows", "-", f"{r.macro_F1:.4f}", round(r.fit_seconds, 2), f"accuracy {r.accuracy:.3f}; recall D/E/G {r.recall_Dropout:.2f}/{r.recall_Enrolled:.2f}/{r.recall_Graduate:.2f}")
cols = ["Run ID", "Date", "Stage", "Model", "Feature set", "Hyperparameters / variant", "Seed", "Data split", "Val / CV macro-F1", "Test macro-F1", "Runtime (s)", "Observations"]
log = pd.DataFrame(rows, columns=cols); log.to_csv(R / "m5_experiment_log.csv", index=False)
import platform, sklearn, pandas, numpy, scipy
env = [["Item", "Value"], ["Python", platform.python_version()], ["scikit-learn", sklearn.__version__], ["pandas / numpy / scipy", f"{pandas.__version__} / {numpy.__version__} / {scipy.__version__}"], ["Hardware", "CPU only, single core; tuning about 10 minutes, analyses about 10 minutes, final test run about 2 minutes"],
       ["Dataset", "UCI Predict Students' Dropout and Academic Success; MD5 (line endings normalised) 9b2b50f41d0cefbbc3e8222d60032d91"], ["Split", "Stratified 70/15/15, seed 42 (data/split_indices.json): 3,096 / 664 / 664"],
       ["Primary metric", "macro-F1 (fixed in Milestone 2)"], ["Tuning budget", "15 random configurations x 5-fold stratified CV per model and feature set (identical for all four models)"],
       ["Selection rule", "highest mean CV macro-F1 on the early feature set (fixed before analysis); selected model: " + alog.get("selected", "")],
       ["Test set", "used once, in scripts/m5_final_test.py, after all tuning and analyses; trained on train + validation rows; frozen settings SHA-256 " + flog.get("best_params_sha256", "(not run yet)")],
       ["Commands", "python scripts/m5_tune.py ; python scripts/m5_analysis.py ; python scripts/m5_final_test.py --confirm-final ; python scripts/m5_build_log.py"]]
wb = Workbook(); ws = wb.active; ws.title = "Experiment log"; F = "Arial"
ws.append(cols)
for r in rows: ws.append(r)
for c in ws[1]: c.font = Font(name=F, bold=True, color="FFFFFF", size=10); c.fill = PatternFill("solid", fgColor="1F3A5F"); c.alignment = Alignment(wrap_text=True, vertical="center")
for row in ws.iter_rows(min_row=2):
    for c in row: c.font = Font(name=F, size=9); c.alignment = Alignment(wrap_text=True, vertical="top")
for i, w in enumerate([22, 11, 34, 17, 10, 46, 9, 26, 20, 12, 11, 52], 1): ws.column_dimensions[get_column_letter(i)].width = w
ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
we = wb.create_sheet("Environment and protocol")
for r in env: we.append(r)
we["A1"].font = we["B1"].font = Font(name=F, bold=True)
for row in we.iter_rows(min_row=2):
    for c in row: c.font = Font(name=F, size=10); c.alignment = Alignment(wrap_text=True, vertical="top")
we.column_dimensions["A"].width = 18; we.column_dimensions["B"].width = 120
wb.save(R / "M5_Experiment_Log.xlsx"); print(len(log), "log rows;", log.Stage.nunique(), "stages")
