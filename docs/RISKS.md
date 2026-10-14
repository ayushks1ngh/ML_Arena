# Risks, mitigations, fallbacks

| # | Risk | Likelihood / impact | Mitigation | Fallback |
|---|---|---|---|---|
| 1 | Too much scope for 7 days | High / High | Generic runner + registry; labs kept small; strict gates; ordered scope-cut list in `PLAN.md` | Cut in listed order; never cut the "never cut" items |
| 2 | Titanic Kaggle download needs login / licence blocks committing | Med / Med | Loader fallback chain (local file → OpenML → instructions); never commit raw | Use OpenML version consistently; log source |
| 3 | Titanic test set tiny → noisy rankings, false "winners" | High / Med | ≥5 seeds, mean±std, tie flag when gap < 1 std; report ties honestly; paired seed comparisons | Raise to 10 seeds (cheap) |
| 4 | Tuning budget unfair across models | Med / Med | Same inner-CV, small comparable grids, same scoring; document | Fixed defaults for all |
| 5 | Data leakage (scaling/imputing before split, tuning on test) | Med / Critical | Pipeline fitted on train only; unit tests; code review at Gate 1 | — |
| 6 | Perturbation done in the wrong space (noise on raw vs. scaled) producing meaningless σ | Med / Med | Define σ in std units of standardized numeric columns; test that noise std ≈ σ | — |
| 7 | Missing-value perturbation: re-imputing with test statistics | Low / High | Impute with train medians only (fill values passed in) | — |
| 8 | Clustering "winner" claimed from silhouette alone | High / Med | Report n_clusters, noise fraction, sizes, scatter, stability ARI; state no ground truth | Present descriptively, not as a ranking |
| 9 | Silhouette undefined (1 cluster / all noise in DBSCAN, Mean Shift single cluster) | Med / Low | Return NaN + flag; exclude from means; count occurrences | — |
| 10 | Robustness curves non-monotonic due to single noise draw | Med / Low | 5 seeds × levels; average; show std bands | Add 3 noise repetitions per seed |
| 11 | XGBoost install/runtime issues in WSL | Low / Med | Verify import on Day 1 (`check_env.py`) | `HistGradientBoostingClassifier` as documented stand-in (explain deviation; XGBoost is required, so resolve) |
| 12 | MLP convergence warnings / unstable ANN | Med / Low | `early_stopping`, `max_iter=1000`, scaled inputs, several seeds | Smaller network |
| 13 | California runtime (tuning RF/GB/XGB on 20k rows × 5 seeds × perturbations) | Med / Med | Small grids, `n_jobs=-1`, tune on subsample if needed, `--quick` mode | Fewer seeds for regression |
| 14 | Dashboard time sink | Med / Med | Reads CSVs only, Plotly defaults, 5 pages max | Static tables + matplotlib figures |
| 15 | Student cannot explain generated code | Med / High | Viva Q&A per phase; short modules; walkthrough after every phase; stop for confirmation | Slow down; simplify code |
| 16 | Session expires mid-work / context lost | Med / Med | `CLAUDE.md` + `docs/STATUS.md` kept current; phases end with a commit when asked; all decisions logged | New session reads CLAUDE.md → STATUS.md |
| 17 | Mall dataset is small/synthetic; conclusions overreach | High / Low | State limitation; stability-based reasoning; avoid strong claims | Add make_blobs/moons sanity study (labelled synthetic) |
| 18 | Hypotheses turn out false (e.g. ensembles not more robust) | Med / Low | This is fine: negative results are reported and explained | — |
