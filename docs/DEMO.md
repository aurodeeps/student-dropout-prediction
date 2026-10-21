# Milestone 3 lab demonstration - run sheet (any team member can follow this)

**Goal (about 3 minutes):** show that the baseline can be regenerated from scratch and explain what it does.

## 1. Run it
```bash
git clone https://github.com/aurodeeps/student-dropout-prediction.git && cd student-dropout-prediction
git checkout v0.3-baseline
python -m venv .venv && .venv\Scripts\activate      # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q                                   # 6 checks should pass
python scripts/run_baseline.py                        # about 30 s
python scripts/subgroup_baseline.py                   # optional: baseline by group
```
Expected: first line `data ok ... split train/val/test = 3096/664/664 (test untouched)`, then the baseline table. Logistic regression (C=1): macro-F1 0.651 (early) and 0.705 (full). Files appear in `results/` and `figures/`.

## 2. What to say (one sentence each)
- **Task:** predict Dropout / Enrolled / Graduate for 4,424 students (UCI dataset, CC BY 4.0).
- **Split:** stratified 70/15/15, seed 42, stored in `data/split_indices.json`. The test set is not touched until the final evaluation.
- **Pipeline:** one-hot encode the 9 coded categorical variables, standardize the rest, inside a scikit-learn `Pipeline` fit on training data only (no leakage).
- **Naive benchmarks:** majority class (macro-F1 0.222, accuracy 0.50) and stratified random (0.313).
- **Baseline:** multinomial logistic regression; macro-F1 0.651 early features, 0.705 full features.
- **Why macro-F1:** classes are imbalanced (50/32/18%), and accuracy hides failure on Enrolled.
- **Reproduction:** `results/reproduction_check.csv` shows all 8 rows match the Milestone 2 memo.
- **Biggest weakness:** Enrolled class (recall 0.32 early).
- **Group check (builds on the EDA notebook):** Dropout recall is 0.98 for students with tuition not up to date but 0.63 for those up to date, so the baseline leans on financial variables; we test this in Milestone 5.

## 3. Questions to be ready for
- *What does logistic regression do here?* It computes one linear score per class from the features, turns the three scores into probabilities with softmax, and is trained by minimizing cross-entropy loss with L2 regularization (C=1).
- *Why is the early set smaller?* It drops the 6 semester-2 course columns, so 36 - 6 = 30 features. The 3 macroeconomic columns are kept. It reflects what an advisor knows after semester 1.
- *Why not just report accuracy?* Always predicting Graduate already gets 50% accuracy.
- *What changes if the seed changes?* Deterministic models do not change; the stratified-random benchmark does (mean 0.337 over seeds 0-9).
- *Is there leakage?* Preprocessing is fit on training data only; financial variables (tuition up to date, debtor) may be recorded late, which we will test in Milestone 5.
