# STATUS — live tracker (update at the end of every phase)

**Last updated:** 2026-10-15
**Mode:** AUTONOMOUS (user approved plan + told me to run Days 1-7 without confirmation).
**Current phase:** Day 3 done (Gate 3 passed). **Next action:** Day 4 (imbalance analysis/H1-H6 verdicts, ID3, failures.py + notebook 06, syllabus labs D/F/G).

## Phase checklist
- [x] Repo scaffold (README, .gitignore, initial PLAN)
- [x] Planning docs: CLAUDE.md, PRD, PLAN, EXPERIMENT_DESIGN, ARCHITECTURE, DATASETS, SYLLABUS_MAP, ACCEPTANCE_CRITERIA, RISKS, STATUS, VIVA_PREP (seed)
- [x] **Plan approved by user** (autonomous mode)
- [x] Day 1 — Setup, data, skeleton (Gate 1: 5 tests pass; quick run wrote results/raw/clf_baseline_quick.csv)
- [x] Day 2 — Leaderboards clf + reg (Gate 2: 8 clf models (ID3 added Day 4) + 6 reg models x 5 seeds, all beat dummy, 6 tests pass)
- [x] Day 3 — Robustness engine + clustering (Gate 3: 12 tests pass; refit baselines reproduce tuned ones exactly; degradation monotone in severity)
- [ ] Day 4 — Imbalance, failures, ID3, quick labs (Gate 4)
- [ ] Day 5 — Dashboard + remaining labs (Gate 5)
- [ ] Day 6 — README, REPORT, viva, fresh-clone test (Gate 6)
- [ ] Day 7 — Buffer, full run, final polish

## Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-10-15 | Use scikit-learn MLP for ANN; no torch/TF | Simplicity, fast, syllabus only needs fully connected net |
| 2026-10-15 | Permutation importance instead of SHAP | Fewer dependency risks, easier to explain |
| 2026-10-15 | California Housing from scikit-learn, not Kaggle CSV | No login, reproducible, cleaner variant |
| 2026-10-15 | Titanic: Kaggle file if user has it, else OpenML fallback; never commit raw | Licence/access uncertainty (VERIFY Day 1) |
| 2026-10-15 | Robustness = test-time perturbation of a fixed trained model (noise/missing/drop); imbalance retrains | Clean separation of "deployed model robustness" vs. "training-data shift" |
| 2026-10-15 | Syllabus items without fair comparison are separate labs (A–G) | Avoid invalid comparisons |
| 2026-10-15 | Tuning metric for clf = macro-F1 | Prevents majority-class bias in tuning |

## Open questions for the user
1. Do you approve the plan as written (7 days, scope cuts, labs A–G)? Any syllabus item you want treated differently?
2. Do you have a Kaggle account / the Titanic `train.csv` and `Mall_Customers.csv` downloaded? (Otherwise fallbacks are used.)
3. Preferred seeds: 5 (default) or 10?
4. Deadline: which day does the project have to be *submitted/presented* (to fit Day 7 buffer)?
5. Is a written report in Markdown fine, or do you need a Word/PDF (can be exported later)?

## Environment notes
- `.venv/` exists at repo root (not tracked). Day 1: check what's installed (`pip list`), then pin `requirements.txt`.

## Session handoff log
- 2026-10-15: Planning docs created. Nothing implemented. Resume by reading `CLAUDE.md` → this file → `PLAN.md`.
- 2026-10-15 (Day 1): Kaggle API worked (titanic comp + vjchoudhary7/customer-segmentation-tutorial-in-python). Data in data/raw/ (gitignored). Titanic 891x12, Mall 200x5, California 20640x8 all verified. Deps pinned in requirements.txt (pandas 3.0, sklearn 1.9, xgboost 3.4). Built config/data/preprocess/models/evaluate + check_env + run_classification + notebook 01 + tests. Design: GridSearchCV over full Pipeline(pre, est) then split fitted steps, so tuning is leak-free and perturbation can act between pre and est. Mall licence still to check (Day 6 report).
- 2026-10-15 (Day 2): added knn/nc/ann/ada/xgb + reg models, runner.py, summarize.py, viz.py, notebooks 02/03 (executed), seed-0 predictions saved. Full runs: clf ~2 min, reg ~10 min. Measured: clf macro-F1 best=xgb 0.834+-0.017 (rf, dt tie within 1 std); reg RMSE best=xgb 0.449 (gb 0.465, rf 0.503). Quick-run outputs are gitignored.
- 2026-10-15 (Day 3): perturb.py, importance.py (grouped permutation importance), robustness.py (+summaries: robustness_summary/rank_stability/ranks), clustering.py, notebooks 04/05 executed. Imbalance (clf) was built into the robustness sweep (retrains with same hyperparameters). Observations (not yet formal verdicts): clf top-1 changes under almost every perturbation (xgb baseline winner; rf wins noise/missing; nc/dt/ann win some); NC loses least under imbalance/noise but has the lowest baseline; reg: linear least hurt by noise, boosted trees (xgb/gb) hurt most. Clustering: single linkage selects k=2 (chaining), k-means most stable (ARI .95->.50 as sigma .1->.5), single-linkage ARI erratic. ID3 NOT yet in clf arena or robustness (Day 4). Decision: drop_feature ablates whole original features (all one-hot columns) ranked by grouped permutation importance of RF on TRAIN rows.
