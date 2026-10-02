# Predicting Student Dropout and Academic Success (CSCI 3052U, Group 21 - ML Musketeers)

3-class supervised classification (Dropout / Enrolled / Graduate) on the UCI
*Predict Students' Dropout and Academic Success* dataset (Realinho et al., 2022).

## Structure
- `data/` - `data.csv` (dataset, semicolon-separated), `split_indices.json` (fixed split, seed 42)
- `src/data.py` - loading, feature sets (early = 30 features, full = 36), stratified 70/15/15 split
- `notebooks/01_m2_eda.ipynb` - Milestone 2 data card, EDA, split and feasibility check
- `figures/` - generated figures
- `reports/` - Milestone 1 proposal and Milestone 2 memo (PDF)

## Setup and reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace notebooks/01_m2_eda.ipynb
```
Data: download from the UCI Machine Learning Repository (doi 10.24432/C5MC89) and place the CSV at
`data/data.csv` if it is not already there. Check the dataset license before making this repository public.

## Team and process
Work is tracked with GitHub Issues/Project board; changes go through branches and pull requests
(each member: at least one PR and one code review). Add `CONTRIBUTIONS.md` and `AI_USE.md` as the project progresses.
