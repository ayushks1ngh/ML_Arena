# Architecture

Principle: the simplest thing that works. A Python package + runner scripts + CSVs + Streamlit app. No services, no DB.

## 1. Folder structure
```
ML_Arena/
├── CLAUDE.md                 # session bootstrap
├── README.md                 # setup, usage, results overview
├── requirements.txt          # pinned deps
├── pyproject.toml            # (optional) makes `pip install -e .` work for src/arena
├── data/
│   ├── raw/                  # downloaded originals (gitignored if license restricts)
│   └── processed/            # optional cleaned copies
├── src/arena/
│   ├── config.py             # SEEDS, paths, perturbation levels, test size
│   ├── data.py               # load_titanic / load_california / load_mall (+ fallbacks)
│   ├── preprocess.py         # build_preprocessor(task) -> ColumnTransformer
│   ├── models.py             # registries: CLF_MODELS, REG_MODELS (estimator + grid + complexity rank)
│   ├── id3.py                # from-scratch ID3
│   ├── evaluate.py           # metric functions; fit/tune/score one model one seed
│   ├── perturb.py            # noise, missing, drop_feature, imbalance
│   ├── robustness.py         # run perturbation sweeps; compute rel_change, AUC, rank flips, tau
│   ├── clustering.py         # run clustering algos, silhouette w/ noise handling, stability (ARI)
│   ├── failures.py           # disagreement, hard cases, permutation importance
│   ├── summarize.py          # raw CSV -> summary CSVs + best-model/tie logic
│   └── viz.py                # shared plotting helpers (used by notebooks AND dashboard)
├── scripts/
│   ├── run_classification.py # baseline leaderboard + predictions
│   ├── run_regression.py
│   ├── run_clustering.py     # + stability
│   ├── run_robustness.py     # --task clf|reg  (noise/missing/drop_feature)
│   ├── run_imbalance.py
│   ├── run_failures.py
│   ├── run_all.py            # everything in order, with --quick flag (2 seeds, small grids)
│   └── check_env.py          # verifies imports, dataset access
├── results/{raw,summary,predictions,figures}/
├── notebooks/
│   ├── 01_data_and_preprocessing.ipynb
│   ├── 02_classification_arena.ipynb
│   ├── 03_regression_arena.ipynb
│   ├── 04_clustering_arena.ipynb
│   ├── 05_robustness_lab.ipynb
│   ├── 06_failure_investigator.ipynb
│   └── syllabus/ (lab_a_regression_types … lab_g_single_linkage)
├── app/
│   ├── Home.py               # Streamlit entry (findings overview)
│   └── pages/ 1_Leaderboards.py 2_Robustness_Lab.py 3_Failure_Investigator.py 4_Clustering.py 5_Methodology.py
├── tests/                    # pytest: leakage, perturb, metrics, id3, rel_change sign
└── docs/                     # planning + REPORT.md (final) + VIVA_PREP.md
```

## 2. Key interfaces (keep stable)
```python
# data.py
load_titanic() -> (X: DataFrame, y: Series)
load_california() -> (X, y)
load_mall() -> DataFrame

# models.py
CLF_MODELS: dict[str, ModelSpec]     # ModelSpec(estimator_factory(seed), grid, complexity_rank, needs_discretized=False)
REG_MODELS:  dict[str, ModelSpec]

# evaluate.py
run_one(task, model_key, seed) -> FitResult(pipeline, best_params, metrics: dict, fit_time, predict_time, split_idx)

# perturb.py   (pure functions, explicit rng, return copies)
add_noise(X, numeric_idx, sigma, rng); mask_missing(X, numeric_idx, p, rng, fill_values)
ablate_features(X, idx, fill_values); make_imbalanced(X, y, minority_share, rng)

# robustness.py
sweep(task, model_keys, seeds, families) -> long DataFrame (schema in EXPERIMENT_DESIGN §9)
summarize_robustness(df) -> per-model summary
```
Design notes:
- To perturb after preprocessing, the pipeline is split into `preprocessor` (fitted) and `estimator`: the sweep calls `preprocessor.transform(X_test)` once, perturbs the numeric columns, then `estimator.predict`. `evaluate.run_one` therefore returns both pieces plus the numeric column indices and train fill values.
- `ModelSpec.complexity_rank` is a documented ordinal used for H5.
- Parallelism: `joblib` over (model, seed) is optional; default serial with `--n_jobs`.
- Results are written incrementally per (model, seed) so a crash does not lose everything; scripts skip already-finished rows when `--resume` is passed.

## 3. Dashboard (Streamlit, reads `results/` only)
| Page | Contents |
|---|---|
| Home | Dataset/task selector summary, headline findings (auto-generated from summary CSVs: best model, most robust model, biggest rank flip), hypothesis scoreboard (H1–H6 verdicts) |
| 1 Leaderboards | Task selector; metric selector; sortable mean±std table; best-per-metric badges + tie flag; bar chart with error bars; fit-time column |
| 2 Robustness Lab | Family + metric selectors; degradation curves per model; baseline-vs-perturbed table; rank-bump chart (rank vs. level); rank-stability (τ) table; imbalance panel with accuracy vs. macro-F1 |
| 3 Failure Investigator | Confusion matrices per model; disagreement table with raw feature values; "hard core" examples; permutation importance comparison; regression residual plots |
| 4 Clustering | Algorithm sweep plots, scatter of chosen clusterings, stability (ARI) vs σ, noise fraction for DBSCAN |
| 5 Methodology | Protocol, datasets, limitations, links to notebooks (rendered from docs) |
Use `st.cache_data` on CSV loads; Plotly for charts (interactive); fail gracefully with a "run scripts first" message if a results file is missing.

## 4. Dependencies (pin versions on Day 1)
numpy, pandas, scikit-learn, xgboost, scipy, matplotlib, seaborn, plotly, streamlit, jupyter/ipykernel, nbformat, pytest, joblib. (No torch/tf, no shap.)

## 5. Testing strategy (lightweight, high value)
- Preprocessor fitted on train only (assert no test statistics leak: fit on train vs. fit on all gives different params).
- Perturbation functions: shapes preserved, original untouched, same seed → same output, mask fraction ≈ p, imbalance share ≈ target, test untouched.
- `rel_change` sign convention for error vs. score metrics.
- ID3: entropy/information-gain on a known textbook example (e.g. PlayTennis: gain(Outlook)=0.246).
- Smoke test: `run_all.py --quick` completes and produces every expected CSV.
