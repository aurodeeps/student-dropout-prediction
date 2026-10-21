# Experiment log

All runs use `data/data.csv` (MD5 after normalising line endings: `9b2b50f41d0cefbbc3e8222d60032d91`), stratified 70/15/15 split with seed 42, and the validation set unless stated. The test set has not been used. Environment: Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.

| ID | Date | What | Result | Conclusion |
|---|---|---|---|---|
| E1 | 2026-10-01 | Milestone 2 feasibility check: majority, stratified random, logistic regression (C=1), tree (depth 5), early and full sets | LR macro-F1 0.651 early / 0.705 full | Task is learnable; benchmarks clearly beaten |
| E2 | 2026-10-20 | `scripts/run_baseline.py`: reproduces E1 with the shared pipeline | All 8 rows match E1 to 3 decimals | Pipeline reproduces Milestone 2 numbers |
| E3 | 2026-10-20 | 5-fold CV grid for logistic regression C in {0.01, 0.1, 1, 10, 100}, train only | C = 1 for early and full | Default C is already best on this grid; "tuned" = default |
| E4 | 2026-10-20 | 5-fold x 3-repeat stratified CV on train for each model | LR 0.659 +/- 0.011 (early), 0.697 +/- 0.015 (full) | Validation scores are consistent with CV |
| E5 | 2026-10-20 | Probe: LR with sklearn default max_iter=100 | macro-F1 0.651, no convergence warning (no change) | No effect; we keep max_iter=3000 |
| E6 | 2026-10-20 | Probe: scaler/encoder fit on ALL rows (leakage, not used) | 0.649 (-0.002) | Small effect, still a protocol error; keep train-only fitting |
| E7 | 2026-10-20 | Probe: stratified random over seeds 0-9 | mean 0.337, range 0.316-0.372 (seed 42: 0.313) | Naive benchmark is seed-sensitive; report the mean |
| E8 | 2026-10-21 | `scripts/subgroup_baseline.py`: LR (early) out-of-fold predictions on train+val, by group (variables from the group EDA notebook) | Dropout recall 0.98 when fees not up to date vs 0.63 when up to date; 0.45 for scholarship holders; 0.88 age >= 25 vs 0.64 younger | Baseline leans on financial variables; test without them in M5 |
| E9 | 2026-10-21 | Integration check on the committed `data.csv` (LF line endings) | Raw MD5 differs from the original CRLF download (9b2b50f4... vs ee29c9ee...); identical after normalising; all results unchanged | Checksum now ignores line endings |

Negative or inconclusive results are kept here on purpose (E3, E5, E6).
