# Project: Robustness Evaluation Framework for ML Models

## Overview
A system that trains ML models, stress-tests them under parameterized data
perturbations (noise, missing values, outliers, covariate shift), and produces
a statistically averaged robustness index per model — not a single before/after
snapshot. See `robustness_project_workplan.md` for the day-by-day build phases.

## Tech stack
Python, pandas, numpy, scikit-learn, matplotlib/seaborn, streamlit, joblib.

## Dataset
**Adult Census Income (UCI)** — not Titanic, not Student Performance. Mixed
categorical/numeric features, real class imbalance. All perturbation severity
scaling should be computed from this dataset's actual feature statistics
(std, range), not fixed magic numbers.

## Models
Logistic Regression, Random Forest, Gradient Boosting. Do not substitute
Decision Tree for Gradient Boosting — DT and RF are the same family and don't
produce a meaningfully different robustness comparison.

## Non-negotiable design rules
- Every stress-test function signature is `fn(X, severity, seed)` — never a
  bare `fn(data)` with no severity/seed parameters. This is the single most
  important rule in this project; without it every downstream metric is a
  single noisy sample instead of an average.
- Severities and seeds live only in `config.py`
  (`SEVERITIES = [0.1, 0.2, 0.3, 0.4, 0.5]`, `SEEDS = [0, 1, 2, 3, 4]`).
  No module hardcodes its own severity/seed values.
- The sweep runs every (model × perturbation × severity × seed) combination
  and writes one row per run to `results.csv`. All aggregation, the
  robustness index, plots, and insights read from this file — nothing is
  computed from a single run.
- `shift_distribution` simulates **covariate shift only** (shifts one
  feature's mean by `severity × std`). It does not simulate label shift or
  concept shift. State this explicitly in any docstring, README, or report
  text that mentions it.
- Scope is **Track A**: train on clean data, evaluate on perturbed data
  (measures generalization to real-world corruption). Do not silently mix in
  Track B (training on perturbed data) — if it gets added later, it's a
  separate, clearly labeled experiment, not a variant of the same one.
- Robustness index:
  `retention(severity) = metric(severity) / metric(severity=0)`,
  `robustness_index = mean(retention across tested severities)`,
  averaged across perturbation types for one overall score per model.
- Insight generator statements must be produced from computed thresholds
  against real values in `results.csv` — never hardcoded/templated strings
  printed regardless of outcome.

## Folder structure
```
project/
│── data/
│── models/
│── stress_tests/
│── evaluation/
│── visualization/
│── insights/
│── config.py
│── app.py
│── main.py
│── tests/
│── requirements.txt
```

## Commands
- `python main.py` — runs the full pipeline end to end.
- `streamlit run app.py` — launches the dashboard.
- `pytest tests/` — runs unit tests (severity=0 must return unperturbed data;
  results must be reproducible given the same seed).

## Reference
Full phase-by-phase build plan: `robustness_project_workplan.md`.
