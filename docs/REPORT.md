# ML Arena: Methodology and Findings Report

*Every number below was produced by `python scripts/run_all.py` and is stored in `results/` (5 seeds, mean ± std). Nothing is quoted from memory. Notebooks 02-07 show the same tables with code.*

## 1. Question
Which machine-learning algorithms perform best for classification, regression and clustering, and **does the best model on clean data stay the best when the data gets harder** (noise, missing values, lost features, class imbalance)?

## 2. Data
| Task | Dataset | Size | Notes |
|---|---|---|---|
| Classification | Titanic (Kaggle `train.csv`) | 891 × 12, 38.4% survived | Age 177 missing, Cabin 687 missing (dropped). Engineered: FamilySize, IsAlone, Title |
| Regression | California Housing (scikit-learn) | 20,640 × 8 | Target = median value in $100k, **capped at 5.0** |
| Clustering | Mall Customers (Kaggle) | 200 × 5, no labels | Features: Annual Income + Spending Score (2d), + Age (3d) |

Licence of the Mall dataset could not be verified through the Kaggle API; the raw file is not committed.

## 3. Method
- **Seeds / splits.** Seeds 0-4. Each seed gives one 80/20 split (stratified for classification) shared by all models.
- **Leakage control.** Imputation, scaling and one-hot encoding are fitted on the training split only, inside a `Pipeline`. Hyper-parameters are tuned with 3-fold CV on the training split only (`GridSearchCV` wraps the whole pipeline); the tuning metric for classification is macro-F1.
- **Ties.** A model within one std of the best is reported as a tie, not a win.
- **Models.** Classification: Decision Tree, ID3 (own implementation), kNN, Nearest Centroid, ANN (MLP), Random Forest, AdaBoost, XGBoost (+ dummy). Regression: Linear, Decision Tree, Random Forest, Gradient Boosting, XGBoost (+ dummy). Clustering: K-means, Mean Shift, single-linkage agglomerative, DBSCAN.
- **Robustness.** Test-time perturbations of a fixed trained model, in standardized feature space (σ in training-std units, numeric columns only): Gaussian noise σ ∈ {0.1, 0.25, 0.5, 1.0}; missing values p ∈ {0.1, 0.2, 0.3, 0.5} (MCAR, filled with the **train** median); removal of the top-1 / top-2 features (whole original feature, ranked by grouped permutation importance on train rows). **Imbalance** perturbs the train set (minority share 20/10/5%, model retrained with the same hyper-parameters); a second variant also skews the test set. `rel_change` is oriented so negative = worse for every metric (for RMSE, `rel_change = -1` means the error doubled).

## 4. Performance Arena
### Classification (Titanic, 5 seeds)
| Model | Accuracy | F1 (survived) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **0.846 ± 0.016** | 0.791 ± 0.021 | **0.834 ± 0.017** | 0.877 ± 0.034 |
| Random Forest | 0.839 ± 0.006 | 0.783 ± 0.011 | 0.828 ± 0.007 | 0.870 ± 0.033 |
| Decision Tree | 0.839 ± 0.022 | 0.777 ± 0.034 | 0.825 ± 0.025 | 0.855 ± 0.039 |
| Tree on discretized features (sklearn) | 0.828 ± 0.014 | 0.751 ± 0.039 | 0.809 ± 0.022 | 0.860 ± 0.032 |
| AdaBoost | 0.821 ± 0.025 | 0.760 ± 0.036 | 0.809 ± 0.028 | 0.868 ± 0.029 |
| kNN | 0.822 ± 0.011 | 0.754 ± 0.023 | 0.808 ± 0.014 | 0.858 ± 0.042 |
| ID3 (own) | 0.816 ± 0.020 | 0.733 ± 0.039 | 0.796 ± 0.024 | 0.862 ± 0.041 |
| ANN (MLP) | 0.803 ± 0.047 | 0.731 ± 0.070 | 0.788 ± 0.052 | 0.849 ± 0.046 |
| Nearest Centroid | 0.754 ± 0.034 | 0.683 ± 0.038 | 0.741 ± 0.034 | 0.841 ± 0.034 |
| Dummy (majority) | 0.615 | 0.000 | 0.381 | 0.500 |

XGBoost has the best mean on every headline metric, but Random Forest and the Decision Tree are within one std of it (flagged ties), so **the data does not support a clear winner among the top three**. All models beat the dummy baseline. ID3 and sklearn's entropy tree on the *same* discretized features differ by 0.013 macro-F1 (within one std); both are about 0.015-0.03 below the tree on continuous features, i.e. discretization costs a little.

### Regression (California Housing, 5 seeds)
| Model | RMSE (×$100k) | MAE | R² |
|---|---|---|---|
| **XGBoost** | **0.449 ± 0.008** | 0.293 ± 0.004 | 0.847 ± 0.004 |
| Gradient Boosting | 0.465 ± 0.005 | 0.309 ± 0.003 | 0.836 ± 0.003 |
| Random Forest | 0.503 ± 0.009 | 0.327 ± 0.007 | 0.808 ± 0.008 |
| Decision Tree | 0.598 ± 0.014 | 0.398 ± 0.010 | 0.729 ± 0.013 |
| Linear Regression | 0.729 ± 0.004 | 0.533 ± 0.003 | 0.596 ± 0.003 |
| Dummy (mean) | 1.148 ± 0.008 | 0.908 | 0.000 |

Here the gaps exceed the std: XGBoost > Gradient Boosting > Random Forest > Tree > Linear is a reliable ordering. XGBoost also trains more than 20× faster than scikit-learn Gradient Boosting (2.1 s vs 51.2 s per model including tuning, final full run).

### Clustering (Mall Customers; no ground truth)
| Algorithm (2d features) | Selected setting | Clusters | Noise | Silhouette | ARI vs clean at σ = 0.1 / 0.25 / 0.5 |
|---|---|---|---|---|---|
| K-means | k = 5 | 5 | 0% | 0.555 | 0.95 / 0.85 / 0.50 |
| Mean Shift | bandwidth 0.65 | 7 | 0% | 0.529 | 0.92 / 0.75 / 0.34 |
| DBSCAN | eps 0.3, min_samples 5 | 7 | 17.5% | 0.524 | 0.79 / 0.56 / 0.33 |
| Single linkage | k = 2 | 2 | 0% | 0.434 | 0.19 / 0.40 / 0.20 (very erratic) |

K-means with k = 5 matches the five visible groups in income × spending. Single linkage's best silhouette is at k = 2: it **chains** and peels off a few outliers, so its "best" silhouette is not a good segmentation. We therefore do not rank clustering algorithms by silhouette alone. (3d results are in `results/summary/clustering_selected.csv`.)

## 5. Robustness Lab
### 5.1 Is the clean winner the most robust?  (H1)
- **Classification.** XGBoost wins clean data (macro-F1 0.834). A different model is #1 in 13 of 16 perturbation cells, and the new winner beats XGBoost by more than its own seed std in 10 of 16. Random Forest takes over at every level of missing values and of noise (9 of the 16 cells go to it); the Decision Tree wins when the top feature (Sex) is removed; the ANN wins when two features are removed; Nearest Centroid wins at 5% and 10% minority share (train skewed).
- **Regression.** XGBoost wins clean (RMSE 0.449) but is displaced in 9 of 10 cells (Random Forest under missing values, 2-feature loss and σ = 1.0; Linear Regression for noise σ ≤ 0.5).
- **Verdict H1: Supported** for both tasks.

Mean relative change of the primary metric (negative = worse; macro-F1 for classification):
| Model | Noise | Missing | Drop top-k features | Imbalance (train) | Imbalance (train+test) |
|---|---|---|---|---|---|
| XGBoost | -0.047 | -0.072 | -0.190 | -0.173 | +0.028 |
| Random Forest | -0.010 | -0.018 | -0.315 | -0.221 | +0.012 |
| Decision Tree | -0.042 | -0.044 | -0.207 | -0.190 | -0.013 |
| AdaBoost | -0.022 | -0.014 | -0.324 | -0.206 | +0.014 |
| kNN | -0.022 | -0.031 | -0.108 | -0.341 | -0.152 |
| ANN | -0.018 | -0.040 | -0.098 | -0.330 | -0.133 |
| Nearest Centroid | -0.006 | -0.021 | -0.098 | -0.031 | -0.020 |

Nearest Centroid loses the least in relative terms but starts from the lowest clean score (0.741), so "most robust" must be read together with absolute performance (see `tradeoff_clf.csv`).

Regression (relative RMSE change; -1 = error doubled): all models are hurt badly because California's most informative features (Latitude, Longitude, MedInc) are perturbed. Mean change under noise: Linear -0.49, Decision Tree -0.91, Random Forest -1.09, XGBoost -1.71, Gradient Boosting -1.81. In absolute terms at σ = 0.5 the RMSE is Linear 1.08, Random Forest 1.15, Decision Tree 1.23, XGBoost 1.34, Gradient Boosting 1.43, against 1.15 for the dummy: **at σ = 0.5 no tree model beats predicting the mean by much**, and the most accurate boosted models degrade most.

### 5.2 Hypotheses and verdicts
| ID | Hypothesis | Verdict | Evidence (from `results/summary/hypotheses.csv`) |
|---|---|---|---|
| H1 | Best baseline model is not best under every perturbation | **Supported** (clf and reg) | see 5.1 |
| H2 | Tree ensembles lose less than kNN/ANN under noise | **Not supported** | Ensembles (RF/AdaBoost/XGB) better at only 1 of 4 σ levels (e.g. σ=0.25: -0.019 vs +0.002). The ensemble group is dragged down by XGBoost (-0.047) |
| H3 | kNN and Nearest Centroid suffer most under noise and missing values | **Not supported** | Most affected are XGBoost (-0.047 noise, -0.072 missing) and the Decision Tree; Nearest Centroid is among the least affected (-0.006, -0.021) |
| H4 | Under imbalance accuracy hides minority-class failure | **Not supported** if the test set stays balanced; **Supported** if the test set is skewed too | Balanced test (5% minority in train): accuracy falls 13-19 points for most models, so it is not hidden. Train and test both skewed: accuracy *rises* 5.5-9 points for ADA/ANN/DT/kNN/RF/XGB while minority recall falls 30-70 points, and ranking by accuracy differs from ranking by macro-F1 (Kendall τ = 0.62) |
| H5 | Complexity does not predict robustness | **Supported** for classification (ρ = -0.22, p = 0.64, n = 7); **Not supported** for regression (ρ = -0.97, n = 5: more complex = less robust) | Small n; complexity is a coarse ordinal |
| H6 | Single linkage and Mean Shift are less stable than K-means under noise | **Supported** (2d and 3d) | 2d ARI at σ = 0.25: K-means 0.85, Mean Shift 0.75, single linkage 0.40; DBSCAN noise fraction grows with σ (0.18 → 0.27 → 0.39) |

### 5.3 Imbalance in one sentence
Accuracy is a trap exactly when the test distribution is as skewed as the training one: a model can gain accuracy by predicting the majority class while recalling far fewer of the minority class; macro-F1 and minority recall expose this.

## 6. Failure Investigator (Titanic seed 0, 179 test passengers)
- **Agreement.** All models correct on 110 passengers; contested on 56; wrong for *every* model on 13 (the "hard core"): 12 of them are men who survived (1 woman who died), 9 travelled in 3rd class. These cases contradict the strongest pattern in the data (men and 3rd class mostly died), so no model, regardless of family, can get them without extra information.
- **Subgroups (XGBoost).** Error rate: men 18.1% vs women 11.1%; Pclass 1 20.9%, Pclass 3 16.7%, Pclass 2 5.9%; adults 16.0%, children 7.1% (n = 14, small).
- **What models rely on** (permutation importance, drop in macro-F1): the Decision Tree depends almost entirely on `Title` (0.292), a proxy for Sex; XGBoost on Title (0.187), Pclass (0.083), Sex (0.052); Random Forest on Sex (0.148); kNN on Sex/Pclass (0.092 each). Different models use different proxies for the same underlying signal, which also explains why removing "Sex" hurts some models far more than others.
- **Regression.** Permutation importance is dominated by Latitude/Longitude (XGBoost 0.99 / 0.90), then MedInc and AveOccup. For houses at the $500k cap the MAE is 0.57-0.77 for tree models and 1.55 for linear regression, and residuals are positive (true value above prediction by 0.50-1.20): the cap makes the most expensive houses systematically under-predicted.

## 7. Accuracy, stability and complexity
- The most accurate models are not the most stable under feature perturbation: XGBoost wins clean data but has the worst mean change under missing values (-0.072) and noise (-0.047) among classifiers, and Gradient Boosting/XGBoost degrade most under noise on the regression task.
- Complexity (ordinal 1-5) is not a reliable guide on Titanic (ρ = -0.22) but on California the most complex models lose most (ρ = -0.97 with 5 models).
- Cost: XGBoost is both the most accurate and one of the cheapest to train; scikit-learn Gradient Boosting is more than 20× slower and slightly less accurate on California.
- Practical reading: choose by the **worst case you expect**. If inputs may be noisy or incomplete, Random Forest was the most consistently strong classifier in our runs (top-1 under missing values and moderate noise) even though XGBoost leads on clean data.

## 8. Limitations
1. Titanic's test set has 179 rows: seed-to-seed std is 1-5 points, so many differences are ties.
2. Tuning grids are small, so absolute scores are not tuned to the limit; the same budget style is used for every model.
3. Noise is added in standardized space to all numeric features at once, including features (Latitude/Longitude) where this is unrealistically destructive; results show relative, not absolute, deployability.
4. Only 5-7 models per task: Spearman correlations (H5) have little power.
5. Mall Customers is small, likely synthetic, and has no labels; clustering conclusions are about stability and internal metrics only. Its licence could not be verified.
6. ID3/discretized models are excluded from robustness (noise on integer categories is undefined).
7. One noise draw per (seed, level); averaging is over 5 seeds only.

## 9. Reproduce
```
pip install -r requirements.txt
python scripts/run_all.py          # full (~20-30 min), or --quick (about 1.5 min, writes results_quick/)
streamlit run app/Home.py
pytest
```
