# ML Arena — Roadmap (7 days)

Rules: finish each day's **gate** before moving on; the user confirms before the next phase begins. Details of *what* to compute live in `EXPERIMENT_DESIGN.md`; module layout in `ARCHITECTURE.md`. Track progress in `STATUS.md`.

Time budget assumes ~6–8 focused hours/day. Order is deliberate: core results first, polish later; every day ends with something demo-able.

## Day 1 — Setup, data, end-to-end skeleton
- Check `.venv`, install/pin deps (`requirements.txt`), create folder skeleton, `config.py`, `check_env.py`.
- Verify the 3 datasets (shape, columns, licence, access). Implement `data.py` loaders + fallbacks.
- `preprocess.py` (train-only fit), `evaluate.py` (metrics), `models.py` with **two** classifiers (DT, RF) to prove the pipeline.
- Notebook 01 (data & preprocessing). Tests: leakage + metrics.
- **Gate 1:** `python scripts/run_classification.py --quick` writes a real CSV for 2 models × 2 seeds; tests pass; user can explain the pipeline.

## Day 2 — Leaderboards (classification + regression)
- Add all classifiers (kNN, NC, ANN, AdaBoost, XGB, Dummy) and all regressors; grids + inner CV.
- `summarize.py`: mean±std, ranks, best-per-metric, tie flag. Save seed-0 predictions.
- Notebooks 02, 03. Start `VIVA_PREP.md`.
- **Gate 2:** `results/summary/leaderboard_clf.csv` and `_reg.csv` exist with all models × 5 seeds; Dummy baseline is beaten by all real models (sanity); no leakage tests green.

## Day 3 — Robustness engine + clustering
- `perturb.py` (+tests), `robustness.py`: noise / missing / drop_feature sweeps for clf & reg; summaries (rel_change, AUC, flips, τ).
- `clustering.py`: K-means/Mean Shift/single linkage/DBSCAN sweeps, silhouette w/ noise handling, stability ARI.
- Notebooks 04, 05 (first version).
- **Gate 3:** `robustness_{clf,reg}.csv` and `clustering*.csv` produced; at least one degradation-curve plot per task looks sensible (performance decreases with severity; if not, debug); perturbation tests green.

## Day 4 — Imbalance, failures, ID3, first syllabus labs
- `run_imbalance.py`; macro-F1 vs. accuracy analysis.
- `id3.py` + test + comparison; `failures.py` + notebook 06.
- Syllabus labs D (NAND), F (distances), G (single linkage) — the quick ones.
- **Gate 4:** H1–H6 each have computed evidence in a results table (verdicts may be "inconclusive"); ID3 test passes (gain(Outlook)=0.246); failure tables generated.

## Day 5 — Dashboard + remaining labs
- Streamlit app: Home, Leaderboards, Robustness Lab, Failure Investigator, Clustering, Methodology.
- Labs A, B, C, E.
- **Gate 5:** `streamlit run app/Home.py` loads every page without errors using real CSVs; all 7 labs executed with outputs; screenshots saved for report.

## Day 6 — Docs, report, viva
- `README.md` (setup, usage, structure, results highlights from real CSVs), `docs/REPORT.md` (methodology + findings + limitations), finish `VIVA_PREP.md`, hypotheses verdict table.
- Fresh-clone test: new venv, `pip install -r requirements.txt`, `python scripts/run_all.py --quick`, app launches.
- **Gate 6:** acceptance checklist (`ACCEPTANCE_CRITERIA.md`) all ticked except polish items.

## Day 7 — Buffer
Bug fixes, full non-quick run (`run_all.py`), regenerate figures, rehearse viva with the Q&A, final commit/tag. Do not start new features.

## Scope cuts (apply in this order if behind)
1. Dashboard Methodology page → render a static markdown only.
2. DBSCAN and the 3-feature clustering variant.
3. Seeds 5 → 3 (never below 3).
4. Lab E (Q-learning) simplified to a 1-D corridor / 3×3 grid.
5. `drop_feature` perturbation family.
6. Regression robustness (keep clf robustness + imbalance: the signature).
7. Inner-CV tuning → fixed sensible defaults for all models (same for everyone; documented).
**Never cut:** train-only preprocessing, XGBoost, ID3, noise+missing+imbalance on classification, executed results, README, report.

## Checkpoint protocol (every phase)
1. Implement the smallest slice. 2. Run it for real and show output. 3. Run tests. 4. Brief walkthrough for the user + viva Q&A. 5. Update `STATUS.md`. 6. Stop and wait for the user's "go".
