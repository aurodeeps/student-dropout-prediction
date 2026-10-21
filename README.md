# Predicting Student Dropout and Academic Success (CSCI 3052U, Group 21 - ML Musketeers)

3-class supervised classification (Dropout / Enrolled / Graduate) on the UCI
*Predict Students' Dropout and Academic Success* dataset (Realinho et al., 2022).

**Status:** Milestone 3 (baseline reproduction) - release `v0.3-baseline`.

## Structure
```
configs/baseline.json      seeds, expected values (from the Milestone 2 memo), dataset checksum, CV settings
data/                      data.csv (dataset, semicolon-separated), split_indices.json (fixed split, seed 42)
src/                       data.py (loading, split, feature sets, checksum), models.py (pipelines)
scripts/run_baseline.py    regenerates the baseline table end to end
scripts/subgroup_baseline.py   baseline behaviour by group (gender, scholarship, tuition, ...)
notebooks/01_m2_eda.ipynb  group exploratory analysis (crosstabs and boxplots by Target)
notebooks/02_m3_baseline.ipynb   Milestone 3 walkthrough of the results
tests/test_pipeline.py     data, split and pipeline checks
results/                   baseline_table.csv/.md, reproduction_check.csv, discrepancy_probes.csv, subgroup_baseline.csv, run_log.json
figures/                   generated figures
reports/                   proposal and milestone memos (PDF)
docs/                      DEMO.md (lab demo run sheet), EXPERIMENT_LOG.md
```

## Setup and reproduce
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # checks
python scripts/run_baseline.py                         # regenerates baseline results and figures (about 30 s)
python scripts/subgroup_baseline.py                    # subgroup table and figure (a few seconds)
```
The scripts stop if `data/data.csv` differs from the approved version. The checksum ignores line endings
(the original download uses CRLF, the copy stored in Git may use LF; the content is identical).

## Main result so far (validation set, macro-F1)
| Model | Early features (30) | Full features (36) |
|---|---|---|
| Majority class | 0.222 | 0.222 |
| Stratified random (seed 42) | 0.313 | 0.313 |
| Logistic regression (C=1) | 0.651 | 0.705 |
| Decision tree (depth 5) | 0.549 | 0.602 |

The test set has not been used. Full table with accuracy, per-class recall and CV results: `results/baseline_table.md`.

## Data and references
`data/data.csv` is included in this repository, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Dataset: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Dataset descriptor: V. Realinho, J. Machado, L. Baptista, M. V. Martins (2022). "Predicting student dropout and academic success", *Data* 7(11):146. https://doi.org/10.3390/data7110146

Introductory paper: M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, V. Realinho. (2021) "Early prediction of student's performance in higher education: a case study", Trends and Applications in Information Systems and Technologies, vol. 1, Advances in Intelligent Systems and Computing series. Springer. DOI: 10.1007/978-3-030-72657-7_16

## Team and process
Work is tracked with GitHub Issues/Project board; changes go through branches and pull requests
(each member: at least one PR and one code review). Add `CONTRIBUTIONS.md` and `AI_USE.md` as the project progresses.
