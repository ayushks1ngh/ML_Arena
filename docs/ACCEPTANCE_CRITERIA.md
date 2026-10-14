# Acceptance Criteria (objective, checkable)

## Reproducibility & structure
- [ ] Fresh clone + new venv + `pip install -r requirements.txt` works (versions pinned).
- [ ] `python scripts/run_all.py --quick` finishes and creates every CSV listed in `EXPERIMENT_DESIGN.md` §9.
- [ ] `python scripts/run_all.py` (full, ≥5 seeds) finishes in < ~30 min on a laptop CPU.
- [ ] Re-running with the same seeds reproduces identical metric values (spot-check 3 rows).
- [ ] `pytest` passes (leakage, perturbations, metrics/rel_change sign, ID3 PlayTennis gain 0.246 for Outlook).

## Data
- [ ] 3 datasets load; source and licence note recorded in `DATASETS.md` (VERIFY items resolved).
- [ ] Preprocessing documented in notebook 01; imputer/scaler/encoder fitted on train only.

## Performance Arena
- [ ] Classification leaderboard includes: DT, ID3, kNN, Nearest Centroid, ANN, RF, AdaBoost, XGBoost (+ Dummy baseline).
- [ ] Regression leaderboard includes: Linear, DT, RF, GB, XGBoost (+ Dummy baseline).
- [ ] Clustering table includes: K-means, Mean Shift, single-linkage agglomerative (+ DBSCAN if kept) with silhouette, n_clusters, noise fraction.
- [ ] Each leaderboard has mean ± std over ≥5 seeds (≥3 only if a scope cut was logged), best-per-metric marked, ties flagged.
- [ ] All real models beat the Dummy baseline (or the exception is explained).

## Robustness Lab
- [ ] Families executed: noise, missing, imbalance for classification; noise + missing for regression; noise stability for clustering.
- [ ] Every row stores baseline, perturbed, delta, rel_change, params, seed.
- [ ] Summary table shows, per model: mean rel_change, AUC, worst drop, rank flips, Kendall τ.
- [ ] Explicit answer to "is the baseline winner the most robust?" with numbers.
- [ ] H1–H6 each have a verdict (Supported / Not supported / Inconclusive) with the supporting numbers in `docs/REPORT.md`.

## Failure Investigator
- [ ] Confusion matrix for every classifier; disagreement table; ≥10 inspected misclassified examples with raw features.
- [ ] Feature importance (native + permutation) for ≥3 models; regression residual analysis.
- [ ] Written accuracy / stability / complexity trade-off discussion backed by a plot.

## Dashboard
- [ ] `streamlit run app/Home.py` launches; all pages render with real data, no exceptions; dataset/task and metric selectors work.
- [ ] Missing results files produce a friendly message, not a stack trace.

## Syllabus
- [ ] Every item in `SYLLABUS_MAP.md` checklist ticked with executed notebook outputs (Labs A–G + main platform).

## Documentation
- [ ] README: overview, setup, usage (each script, dashboard), structure, real headline results.
- [ ] `docs/REPORT.md`: question, methodology, datasets, results tables, robustness findings, failures, limitations (small Titanic test set, synthetic Mall data, capped CA target), conclusions.
- [ ] `docs/VIVA_PREP.md`: ≥40 Q&A covering each algorithm, protocol, leakage, metrics, robustness, and project decisions.
- [ ] Every number in README/REPORT traces to a file in `results/`; no claim of a winner without evidence.
- [ ] The student can walk through each module and explain it without help.
