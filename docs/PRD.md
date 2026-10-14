# ML Arena — Product Requirements Document

## 1. Vision
A small, reproducible ML research lab: compare algorithms across classification, regression and clustering; stress-test them with controlled data degradation; investigate where and why they fail; present everything in an interactive dashboard. It must read as a mini research project, not a notebook of accuracy scores.

**Central question:** Which algorithms perform best per task, and are they still the best when the data becomes harder?

## 2. Feasibility verdict (one week)
**Feasible, with discipline.** Reasons: all three datasets are small (891 / 20,640 / 200 rows), so full experiments run in minutes; scikit-learn + XGBoost cover almost every algorithm; Streamlit reads CSVs. The risk is breadth, not difficulty. Hence:
- One generic experiment runner serves all models, instead of per-model scripts.
- Syllabus items that are not benchmarkable (NAND perceptron, Q-learning, ML/MAP, distances) are **separate small labs**, not forced into the leaderboards.
- Dashboard reads precomputed CSVs only (no live training).
- A ranked scope-cut list exists (`docs/PLAN.md`).

## 3. Distinctive angle (what makes it more than a benchmark)
**"Does the leaderboard winner stay the winner?"** Most projects stop at clean-data accuracy. ML Arena adds a *rank-stability* analysis: compute the model ranking on clean data, re-compute it under each perturbation and severity, and quantify how much the ranking moves (rank flips, Kendall's τ, relative performance drop, degradation-curve area). Also the *metric-trust* angle: under class imbalance, accuracy can crown a model that ignores the minority class; macro-F1 / minority recall disagree. Combined with failure forensics (where models disagree), this gives real findings, including potential negative ones.

## 4. Research questions and hypotheses
Hypotheses are falsifiable and tested, never assumed. Each has a pass/fail rule defined in `docs/EXPERIMENT_DESIGN.md` §7.

- **RQ1** Which algorithm is best per task and metric on clean data? (No hypothesis; measured.)
- **RQ2** Is the clean-data winner also the most robust?
  - **H1:** The best baseline model is not the best under every perturbation (at least one rank change at the top in at least one perturbation family).
- **RQ3** Which model families degrade most under feature noise / missing data?
  - **H2:** Tree ensembles (RF, AdaBoost, XGBoost) lose less performance under Gaussian feature noise than kNN and the ANN.
  - **H3:** Distance-based models (kNN, Nearest Centroid) suffer the largest relative drop under noise and missing values.
- **RQ4** Does imbalance make accuracy misleading?
  - **H4:** As the minority share shrinks, accuracy stays high or rises while macro-F1 and minority recall fall; the accuracy ranking and macro-F1 ranking diverge.
- **RQ5** Does complexity buy robustness?
  - **H5:** Model complexity (proxy defined in design doc) is not monotonically related to robustness (|Spearman ρ| small or non-significant across models).
- **RQ6 (clustering)** Which clustering algorithm gives stable structure under noise on Mall Customers?
  - **H6:** Single-linkage and Mean Shift are less stable (lower ARI vs. clean clustering) than K-means under added noise; DBSCAN labels increasing fractions as noise.

Caveat to state in the report: Titanic's test set is small (~179 rows), so differences smaller than the seed-to-seed std are reported as ties, not wins.

## 5. Scope

### In scope (core platform)
- **Performance Arena:** leaderboards for 3 tasks, best model per metric, all results in CSV.
- **Robustness Lab:** Gaussian noise, missing-feature masking, targeted feature removal, class imbalance (classification), clustering stability under noise.
- **Failure Investigator:** model disagreement, misclassified examples, confusion matrices, residual analysis (regression), feature importance (permutation + native).
- **Dashboard:** Streamlit, 5–6 pages (see `ARCHITECTURE.md`).
- **Syllabus labs:** separate notebooks (see `SYLLABUS_MAP.md`).
- **Deliverables:** notebooks, reusable `src/arena/` modules, CSVs, tables, figures, README, methodology+findings report, viva material.

### Out of scope
Databases, APIs, auth, cloud deploy, custom frontend, deep-learning frameworks (ANN = scikit-learn MLP), SHAP (permutation importance instead, fewer dependency risks), AutoML, hyperparameter search beyond small grids, live model training in the dashboard.

## 6. Functional requirements
| ID | Requirement |
|---|---|
| F1 | Load/prepare the 3 datasets with documented, leakage-free preprocessing |
| F2 | Train and evaluate all listed algorithms with the same splits per seed |
| F3 | Save raw per-seed results and aggregated mean±std tables as CSV |
| F4 | Auto-identify best model per task/metric (with tie flag when within 1 std) |
| F5 | Run perturbation experiments; store baseline, perturbed, delta, settings, seed |
| F6 | Compute robustness summaries (relative drop, degradation-curve area, rank flips, Kendall τ) |
| F7 | Save per-sample predictions (seed 0) for failure analysis |
| F8 | Implement ID3 from scratch (or documented ID3-style) and compare to sklearn entropy tree |
| F9 | Dashboard exposing leaderboards, robustness, failures, clustering, methodology |
| F10 | Syllabus labs as notebooks with real outputs |
| F11 | `python scripts/run_all.py` regenerates all results; README explains setup |
| F12 | Methodology and findings report + viva Q&A material |

## 7. Non-functional requirements
Reproducible (fixed seeds, pinned deps); full pipeline < ~30 min on a laptop CPU; readable code (modules < ~300 lines, comments on why); tests for leakage-critical and metric code; no invented numbers.

## 8. Users
Primary: the student (builds, explains, defends in viva). Secondary: examiner browsing the dashboard and notebooks.

## 9. Success definition
All items in `docs/ACCEPTANCE_CRITERIA.md` pass, and the student can explain every module and every finding.
