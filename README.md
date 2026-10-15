# ML Arena: Machine Learning Benchmarking & Reliability Lab

Compares ML algorithms across classification, regression and clustering, then tests how reliable they are when data gets harder (noise, missing features, class imbalance). Includes a failure investigator, an interactive Streamlit dashboard, and a set of syllabus labs.

**Central question:** Which algorithms perform best per task, and does the best model on clean data stay the best when the data degrades?

## Headline results (5 seeds; full details in `docs/REPORT.md`)
| Task | Dataset | Best on clean data | Note |
|---|---|---|---|
| Classification | Titanic | XGBoost, macro-F1 0.834 ± 0.017 | Random Forest (0.828) and Decision Tree (0.825) are within 1 std: ties |
| Regression | California Housing | XGBoost, RMSE 0.449 ± 0.008 | gaps exceed std; ordering XGB > GB > RF > Tree > Linear |
| Clustering | Mall Customers | K-means k=5, silhouette 0.555 | most stable under noise; single linkage chains (k=2) |

The clean-data winner is **not** the most robust: a different model is #1 in 13 of 16 classification and 9 of 10 regression perturbation cells. Hypothesis verdicts (H1-H6) are in `docs/REPORT.md` and `results/summary/hypotheses.csv`; some hypotheses were **not** supported.

## New here? Start with the dashboard
`streamlit run app/Home.py` opens a guided tour: what the project is, a live "overfitting" playground (*Learn the Basics*), and every result explained in plain English. No ML background needed.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data (not committed): Titanic and Mall Customers come from Kaggle; California Housing downloads automatically via scikit-learn.
```bash
# needs ~/.kaggle/kaggle.json (and accepting the Titanic competition rules on kaggle.com once)
mkdir -p data/raw/titanic data/raw/mall
kaggle competitions download -c titanic -p data/raw/titanic && unzip -o data/raw/titanic/titanic.zip -d data/raw/titanic
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data/raw/mall && unzip -o data/raw/mall/*.zip -d data/raw/mall
python scripts/check_env.py        # verifies imports and data
```
If `data/raw/titanic/train.csv` is missing the loader falls back to OpenML.

## Usage
| Command | What it does |
|---|---|
| `python scripts/run_all.py` | Regenerates every result in `results/` (5 seeds, tuned; ~20-30 min) |
| `python scripts/run_all.py --quick` | Smoke test (2 seeds, no tuning, ~1.5 min) writing to `results_quick/` |
| `python scripts/run_all.py --only robustness-clf` | Run selected steps (names: classification, regression, leaderboards, robustness-clf, robustness-reg, clustering, failures+hypotheses) |
| `streamlit run app/Home.py` | Beginner-friendly interactive dashboard: explains the project, has a live playground, plain-English takeaways (reads `results/` only) |
| `pytest` | 24 tests: leakage, perturbations, metrics, ID3, dashboard pages |

## Structure
```
src/arena/   data, preprocess, models, evaluate, perturb, robustness, clustering, failures, hypotheses, id3, summarize
scripts/     run_all.py + one runner per experiment, notebook builders (make_*.py)
results/     raw/ (per-seed CSVs), summary/ (leaderboards, robustness, failures, hypotheses), predictions/, figures/
notebooks/   01-07 experiment notebooks; syllabus/ labs A-G
app/         Streamlit dashboard (Home + Learn the Basics + 5 result pages)
tests/       pytest suite
docs/        PRD, PLAN, EXPERIMENT_DESIGN, ARCHITECTURE, DATASETS, SYLLABUS_MAP, ACCEPTANCE_CRITERIA, RISKS, REPORT, VIVA_PREP, STATUS
```

## Syllabus coverage
Main platform: Decision Tree, **ID3 (own implementation)**, kNN, Nearest Centroid, ANN, Random Forest, AdaBoost, **XGBoost**, Linear/DT/RF/GB/XGB regression, K-means, Mean Shift, single linkage, DBSCAN.
Labs (`notebooks/syllabus/`): A regression types, B activation functions, C ML vs MAP, D NAND perceptron, E Q-learning, F Manhattan/Chebyshev distances, G single-linkage calculations.

## Documentation
Start with `docs/REPORT.md` (findings), `docs/EXPERIMENT_DESIGN.md` (protocol), `docs/VIVA_PREP.md` (Q&A). Working notes for the AI assistant: `CLAUDE.md`, `docs/STATUS.md`.

## Limitations
Small Titanic test set (many gaps are ties); small tuning grids; Mall data is small, unlabelled and its licence is unverified; see `docs/REPORT.md` section 8.
