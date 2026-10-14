# Acceptance Criteria (objective, checkable)

## Reproducibility & structure
- [x] Fresh clone + new venv + `pip install -r requirements.txt` works (versions pinned).
- [x] `python scripts/run_all.py --quick` finishes and creates every CSV listed in `EXPERIMENT_DESIGN.md` §9.
- [x] `python scripts/run_all.py` (full, ≥5 seeds) finishes in < ~30 min on a laptop CPU.
- [x] Re-running with the same seeds reproduces identical metric values (spot-check 3 rows).
- [x] `pytest` passes (leakage, perturbations, metrics/rel_change sign, ID3 PlayTennis gain 0.247 (0.2467) for Outlook).

## Data
- [x] 3 datasets load; source and licence note recorded in `DATASETS.md` (VERIFY items resolved except the Mall licence, which the Kaggle API does not expose: documented as unverified).
- [x] Preprocessing documented in notebook 01; imputer/scaler/encoder fitted on train only.

## Performance Arena
- [x] Classification leaderboard includes: DT, ID3, kNN, Nearest Centroid, ANN, RF, AdaBoost, XGBoost (+ Dummy baseline).
- [x] Regression leaderboard includes: Linear, DT, RF, GB, XGBoost (+ Dummy baseline).
- [x] Clustering table includes: K-means, Mean Shift, single-linkage agglomerative (+ DBSCAN if kept) with silhouette, n_clusters, noise fraction.
- [x] Each leaderboard has mean ± std over ≥5 seeds (≥3 only if a scope cut was logged), best-per-metric marked, ties flagged.
- [x] All real models beat the Dummy baseline (or the exception is explained).

## Robustness Lab
- [x] Families executed: noise, missing, imbalance for classification; noise + missing for regression; noise stability for clustering.
- [x] Every row stores baseline, perturbed, delta, rel_change, params, seed.
- [x] Summary table shows, per model: mean rel_change, AUC, worst drop, rank flips, Kendall τ.
- [x] Explicit answer to "is the baseline winner the most robust?" with numbers.
- [x] H1–H6 each have a verdict (Supported / Not supported / Inconclusive) with the supporting numbers in `docs/REPORT.md`.

## Failure Investigator
- [x] Confusion matrix for every classifier; disagreement table; ≥10 inspected misclassified examples with raw features.
- [x] Feature importance (native + permutation) for ≥3 models; regression residual analysis.
- [x] Written accuracy / stability / complexity trade-off discussion backed by a plot.

## Dashboard
- [x] `streamlit run app/Home.py` launches; all pages render with real data, no exceptions; dataset/task and metric selectors work.
- [x] Missing results files produce a friendly message, not a stack trace.

## Syllabus
- [x] Every item in `SYLLABUS_MAP.md` checklist ticked with executed notebook outputs (Labs A–G + main platform).

## Documentation
- [x] README: overview, setup, usage (each script, dashboard), structure, real headline results.
- [x] `docs/REPORT.md`: question, methodology, datasets, results tables, robustness findings, failures, limitations (small Titanic test set, synthetic Mall data, capped CA target), conclusions.
- [x] `docs/VIVA_PREP.md`: ≥40 Q&A covering each algorithm, protocol, leakage, metrics, robustness, and project decisions.
- [x] Every number in README/REPORT traces to a file in `results/`; no claim of a winner without evidence.
- [ ] The student can walk through each module and explain it without help. (only the student can confirm; use docs/VIVA_PREP.md)

## Verification log (2026-10-15, Day 7)
- Full `run_all.py`: 554 s total (limit ~30 min). Re-run vs. previous results: 29 CSVs identical to 1e-9 (timing columns excluded).
- `pytest`: 24 passed (includes AppTest of all dashboard pages and widget walks). Fresh clone + new venv + `pip install -r requirements.txt` + `run_all.py --quick` + pytest: passed.
- Not done: dashboard screenshots (no browser available in the build environment); user walk-through of the code.
