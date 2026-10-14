# ML Arena: Machine Learning Benchmarking & Reliability Lab

Compares ML algorithms across classification, regression and clustering, then tests
how reliable they are when data gets harder (noise, missing features, class imbalance).

**Central question:** Which algorithms perform best per task, and does the best model on
clean data stay the best when the data degrades?

## Status
Planning complete; **no code yet, awaiting plan approval**. Start with `CLAUDE.md`, then `docs/STATUS.md`.

## Documentation
`docs/PRD.md` (vision, hypotheses) · `docs/PLAN.md` (7-day roadmap) · `docs/EXPERIMENT_DESIGN.md` (algorithms, metrics, perturbations) · `docs/ARCHITECTURE.md` · `docs/DATASETS.md` · `docs/SYLLABUS_MAP.md` · `docs/ACCEPTANCE_CRITERIA.md` · `docs/RISKS.md` · `docs/VIVA_PREP.md`

## Planned datasets
| Task | Dataset |
|---|---|
| Classification | Titanic (Kaggle) |
| Regression | California Housing (Kaggle / scikit-learn) |
| Clustering | Mall Customer Segmentation (Kaggle) |

## Planned structure
`src/arena/` reusable modules, `scripts/` runners, `results/` CSVs, `notebooks/`, `app/` Streamlit dashboard, `tests/`, `docs/`.
