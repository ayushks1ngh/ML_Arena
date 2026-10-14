# ML Arena: Machine Learning Benchmarking & Reliability Lab

Compares ML algorithms across classification, regression and clustering, then tests
how reliable they are when data gets harder (noise, missing features, class imbalance).

**Central question:** Which algorithms perform best per task, and does the best model on
clean data stay the best when the data degrades?

## Status
Planning complete; implementation in progress (see `docs/PLAN.md`).

## Planned datasets
| Task | Dataset |
|---|---|
| Classification | Titanic (Kaggle) |
| Regression | California Housing (Kaggle / scikit-learn) |
| Clustering | Mall Customer Segmentation (Kaggle) |

## Planned structure
`src/arena/` reusable modules, `scripts/` runners, `results/` CSVs, `notebooks/`, `app/` Streamlit dashboard, `tests/`, `docs/`.
