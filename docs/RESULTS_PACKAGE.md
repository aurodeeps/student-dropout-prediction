# Milestone 5 results package: contents and how to regenerate

| Checklist item | Where it is | Source run IDs |
|---|---|---|
| Final tables and figures with captions | `results/m5_final_test_table.csv`, `m5_final_test_paired_bootstrap.csv`; `figures/m5_main_figure_early.png`, `m5_main_figure_full.png`, `m5_ablation.png`, `m5_sensitivity_robustness.png`, `m5_test_confusion_matrices.png`; captions in the draft Results section | F-*, A-*, S-*, R-* |
| Raw per-seed results | `results/m5_final_test_per_seed.csv` (test, 5 seeds), `results/m5_cv_repeated_raw.csv` (every CV fold) | F-*, C-* |
| Tuning records (every configuration tried) | `results/m5_tuning_records.csv` (120 runs), `results/m5_best_params.json` | T-* |
| Error analysis | `results/m5_oof_confusion.csv`, `m5_error_profiles.csv`, `m5_dropout_vs_rest.csv`, `m5_final_test_subgroups.csv` | E-* |
| Sensitivity, robustness, ablation | `results/m5_sensitivity_C.csv`, `m5_sensitivity_class_weight.csv`, `m5_sensitivity_enrolled_boost.csv`, `m5_robustness.csv`, `m5_ablation.csv` | S-*, R-*, A-* |
| Experiment log | `results/M5_Experiment_Log.xlsx` (+ `m5_experiment_log.csv`), 278 runs | all |
| Code to reproduce the main table | `scripts/m5_tune.py`, `m5_analysis.py`, `m5_final_test.py`, `m5_build_log.py`; `src/` | - |

## Regenerate (order matters)
```bash
pip install -r requirements.txt
python scripts/m5_tune.py            # equal-budget random search, train rows only (~10 min on one core)
python scripts/m5_analysis.py        # CV, ablation, sensitivity, robustness, error analysis; no test set (~9 min)
python scripts/m5_final_test.py --confirm-final   # ONE use of the test set (~2 min); writes results/m5_FINAL_TEST_DONE.lock
python scripts/m5_build_log.py       # experiment log (xlsx + csv)
```
`m5_final_test.py` refuses to run if the lock file exists. Deleting the lock to re-run counts as reusing the test set and must be disclosed.
