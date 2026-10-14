# Datasets: validation and preparation

> Facts below are from prior knowledge and **must be verified on Day 1** (`scripts/check_env.py`, open each file, confirm shape/columns/licence text). Items marked **VERIFY** are unconfirmed.

## 1. Titanic (classification) — keep, with a loader fallback
- Source: Kaggle "Titanic: Machine Learning from Disaster" `train.csv`. 891 rows × 12 cols: PassengerId, Survived (target), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked. Survival rate ≈ 38%.
- Missingness: Age ≈ 20% (177), Cabin ≈ 77% (687), Embarked 2.
- **Licence/access (VERIFY):** Kaggle competition data requires a login and acceptance of competition rules; redistribution in the repo may not be allowed. → Do **not** commit the raw file. Loader order: (1) `data/raw/titanic.csv` if the user placed it; (2) `sklearn.datasets.fetch_openml("titanic", version=1, as_frame=True)` (1,309 rows, same family of columns, public); (3) clear error message with manual download instructions. Record which source was used in results metadata; mixing sources breaks reproducibility, so pick one and log it in STATUS.
- Features used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, + engineered `FamilySize = SibSp+Parch+1`, `IsAlone`, optional `Title` (from Name; collapse rare titles). Dropped: PassengerId, Name (after Title), Ticket, Cabin (too sparse; optionally `HasCabin`).
- Suitability: well-known, small, fast, mixed types, real missing values (good for imputation + missing-feature robustness). Limitation: tiny test set (~179), so seed variance is high → use ≥5 seeds, treat gaps < 1 std as ties. Strong signals (Sex, Pclass) mean many models plateau around 78–83% accuracy (**expectation, not a result**), so the robustness analysis is where models should differ.

## 2. California Housing (regression) — keep, load via scikit-learn
- Source: `sklearn.datasets.fetch_california_housing()` (1990 US census block groups; Pace & Barry 1997). 20,640 rows × 8 numeric features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude. Target `MedHouseVal` in units of $100,000.
- No missing values. Target is **capped at 5.0** ($500k) → ceiling effect: note in failure analysis; do not remove.
- **Licence/access:** fetched and cached by scikit-learn (needs internet once). The Kaggle "California Housing Prices" `housing.csv` is a *different variant* (has `ocean_proximity`, totals instead of averages, 207 missing `total_bedrooms`). Recommendation: use the scikit-learn version for reproducibility and zero login; cite the Kaggle variant only as an alternative. If offline: save a CSV copy once at `data/raw/california.csv`.
- Suitability: big enough for meaningful RF/GB/XGB differences, small enough to run in minutes. Outliers in AveRooms/AveOccup/Population are real; leave them (robustness is the point). Spatial features (Lat/Lon) favour tree models over linear: expected, to be measured.

## 3. Mall Customer Segmentation (clustering) — keep, with realistic caveats
- Source: Kaggle "Mall Customer Segmentation Data" (`Mall_Customers.csv`). 200 rows × 5 cols: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1–100). No missing values.
- **Licence (VERIFY):** listed on Kaggle as public domain/CC0 by the uploader, but confirm on the dataset page; if confirmed, a copy may be committed (it is ~4 KB), otherwise use a download-instructions loader.
- Suitability: standard for clustering demos; 2-D income×spending view shows ~5 visible groups, so results can be sanity-checked visually. Limitations to state in the report: **small (n=200)**, widely believed to be synthetic, **no ground-truth labels** (only internal metrics + stability), so conclusions about "best clustering algorithm" are tentative.
- Preparation: drop CustomerID; features = Annual Income, Spending Score (primary) and + Age (secondary); standardize; exclude Gender (binary one-hot distorts Euclidean distance; document).
- Fallback if unusable: `sklearn.datasets.make_blobs`/`make_moons` mix (clearly labelled synthetic) or Iris features (no species used for fitting; species could serve as a pseudo ground truth for ARI): log choice in STATUS.

## 4. Preparation outputs
- `notebooks/01_data_and_preprocessing.ipynb`: shape, dtypes, missingness, target distribution, correlations, chosen features, rationale.
- Each loader returns raw (unprocessed) frames; all fitting happens in `preprocess.py` inside the train-only pipeline.
