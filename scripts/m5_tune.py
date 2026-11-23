"""Milestone 5, step 1: equal-budget random search (15 configurations x 5-fold CV) for every model and feature set.

    python scripts/m5_tune.py

Uses the TRAIN rows only (validation and test rows are not touched). Every configuration tried is saved.
Selection rule (fixed before looking at results): highest mean CV macro-F1 per model and feature set.
"""
import json, sys, time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5
from src.experiments import MODELS, SPACES, N_ITER, SEED, estimator
from src.models import make_pipeline

cfg = json.loads((ROOT / "configs" / "baseline.json").read_text()); assert file_md5() == cfg["dataset_md5"]
df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df); fs = feature_sets(df)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
records, best = [], {}
for fname in ["early", "full"]:
    cols = fs[fname]
    for m in MODELS:
        t0 = time.time()
        rs = RandomizedSearchCV(make_pipeline(cols, estimator(m), dense=(m == "hgb")), SPACES[m], n_iter=N_ITER, scoring="f1_macro", cv=cv,
                                random_state=SEED, n_jobs=1, refit=False, return_train_score=True)
        rs.fit(Xtr[cols], ytr)
        res = pd.DataFrame(rs.cv_results_)
        clean = lambda d: {p.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for p, v in d.items()}
        for k, r in res.iterrows():
            records.append({"run_id": f"T-{fname}-{m}-{k+1:02d}", "feature_set": fname, "model": m, "params": json.dumps(clean(r["params"])),
                            "cv_macro_F1_mean": r["mean_test_score"], "cv_macro_F1_sd": r["std_test_score"],
                            "train_macro_F1_mean": r["mean_train_score"], "fit_seconds_mean": r["mean_fit_time"]})
        i = int(res["mean_test_score"].idxmax()); best[f"{fname}|{m}"] = clean(res.loc[i, "params"])
        print(fname, m, "best cv", round(res.loc[i, "mean_test_score"], 4), best[f"{fname}|{m}"], f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(records).round(5).to_csv(ROOT / "results" / "m5_tuning_records.csv", index=False)
        (ROOT / "results" / "m5_best_params.json").write_text(json.dumps(best, indent=2))
print("done")
