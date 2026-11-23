# AI use record (DRAFT - every member must review and correct this before it is relied on)

Course rules require that AI assistance is disclosed and that every member can explain the submitted work.

| Item | AI tool | What it was used for | How it was checked by the team |
|---|---|---|---|
| Milestones 1-2 (proposal, approval memo) | Claude (Anthropic) | Drafting text, computing dataset statistics, building the data card and EDA notebook | [Team: describe who reviewed the numbers and sources] |
| Milestone 3 (baseline pipeline, memo) | Claude (Anthropic) | Writing `src/`, `scripts/`, `tests/`, the technical memo, and Git/GitHub instructions | Results regenerate with `python scripts/run_baseline.py`; [Team: add reviewer names and dates] |
| Milestone 4 (models, report, slides, script) | Claude (Anthropic) | Writing `scripts/run_models.py`, drafting the progress report, slides and video script | Results regenerate with `python scripts/run_models.py`; [Team: add reviewer names and dates] |
| Milestone 5 (tuning, analyses, final test, log, draft Results section) | Claude (Anthropic) | Writing `scripts/m5_*.py` and `src/experiments.py`, running the experiments, building the experiment log and drafting the Results section | Results regenerate with the commands in `docs/RESULTS_PACKAGE.md`; [Team: add reviewer names and dates] |

References were looked up and checked by the team: [Team: confirm which references each member verified yourselves].
Do not use AI to hide unequal contribution or to produce results that were not actually run.

