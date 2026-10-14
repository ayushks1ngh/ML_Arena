# Experiment Design

Exact specification. Implementation must follow this; deviations get logged in `STATUS.md`.

## 1. Global protocol
- **Seeds:** `SEEDS = [0,1,2,3,4]` (config constant; extend to 10 if runtime allows: cheap on these datasets). Seed controls the train/test split, model random_state, and perturbation RNG.
- **Split:** supervised: 80/20 hold-out per seed. Classification stratified on target. Regression plain random split.
- **Preprocessing (fit on train only), inside an sklearn `Pipeline`/`ColumnTransformer`:** median impute numeric, most-frequent impute categorical, one-hot categorical, standardize numeric. The same transformed feature matrix is used for all models (trees are scale-invariant, so this is fair and simple). Exception: ID3 uses its own discretization (see §3).
- **Tuning:** small fixed grid per model, 3-fold inner CV on the training set only, same scoring metric per task (clf: macro-F1... *see note*; reg: neg RMSE). Same budget philosophy for all. Tuned once per seed on clean train; the **tuned model is reused for test-time perturbations** (no retraining), so the perturbation measures robustness of a deployed model. Imbalance perturbs training data, so models retrain there with the *same hyperparameters*.
  - Note: clf tuning metric = **F1-macro** so tuning does not favour majority-class accuracy; report accuracy too.
- **Aggregation:** mean ± std over seeds. Best model per metric = best mean; flag **tie** if the gap to second place < 1 std of the leader.
- **Timing:** record fit time and predict time per model (supports the accuracy/stability/complexity trade-off).

## 2. Classification (Titanic)
Models and grids (keep small):
| Key | Model | Grid |
|---|---|---|
| `dt` | DecisionTreeClassifier(criterion="entropy") | max_depth {3,5,8,None}, min_samples_leaf {1,5,10} |
| `id3` | custom ID3 (see §3) | max_depth {3,5,None} |
| `knn` | KNeighborsClassifier | n_neighbors {3,5,11,21}, weights {uniform,distance} |
| `nc` | NearestCentroid | metric {euclidean, manhattan} |
| `ann` | MLPClassifier | hidden {(32,),(64,32)}, alpha {1e-4,1e-2}, max_iter 1000, early_stopping |
| `rf` | RandomForestClassifier | n_estimators {200}, max_depth {5,10,None}, min_samples_leaf {1,3} |
| `ada` | AdaBoostClassifier | n_estimators {50,200}, learning_rate {0.1,0.5,1.0} |
| `xgb` | XGBClassifier | n_estimators {200}, max_depth {3,5}, learning_rate {0.05,0.1}, eval_metric logloss |
| Baseline | DummyClassifier(most_frequent) | none: sanity floor |
Optional extra (only if time): `logreg` as a calibration baseline. Everything else is cut.

Metrics: accuracy, precision, recall, F1 (binary, positive = survived) **and macro-F1**, ROC-AUC (if probas), confusion matrix (stored for seed 0).

## 3. ID3 implementation
- Custom `ID3Classifier` in `src/arena/id3.py`: entropy + information gain, multiway splits on categorical features, recursion with stopping rules (pure node, no features, max_depth), majority-class leaves, unseen category → majority class of the node.
- ID3 needs categorical inputs: discretize Age (bins: child/teen/adult/senior or quantiles), Fare (quartiles), keep Pclass/Sex/Embarked/SibSp/Parch (cap at 3+). Discretization bin edges fit on train.
- Compare with sklearn entropy tree on the *same* discretized features (isolates implementation from preprocessing). Print the learned tree and a hand-computable entropy/information-gain example for viva.

## 4. Regression (California Housing)
| Key | Model | Grid |
|---|---|---|
| `lin` | LinearRegression | none |
| `dt` | DecisionTreeRegressor | max_depth {5,10,None}, min_samples_leaf {5,20} |
| `rf` | RandomForestRegressor | n_estimators {200}, max_depth {10,None}, min_samples_leaf {1,5} |
| `gb` | GradientBoostingRegressor | n_estimators {200}, max_depth {3,5}, learning_rate {0.05,0.1} |
| `xgb` | XGBRegressor | n_estimators {300}, max_depth {4,6}, learning_rate {0.05,0.1} |
| Baseline | DummyRegressor(mean) | none |
Metrics: MAE, RMSE, R². Also residual store (seed 0). Full 20,640 rows; if grids are too slow, tune on a 5,000-row subsample of the train set and log it.

## 5. Clustering (Mall Customers)
Features: primary = standardized `Annual Income`, `Spending Score` (2-D, interpretable, visualizable); secondary variant = + `Age`. Gender excluded (document why).
| Key | Algorithm | Settings |
|---|---|---|
| `kmeans` | KMeans | k ∈ 2..10 swept; report silhouette per k; select k by max silhouette (also show elbow) |
| `meanshift` | MeanShift | bandwidth via `estimate_bandwidth` over quantile grid {0.1,0.2,0.3} |
| `agg_single` | AgglomerativeClustering(linkage="single") | n_clusters 2..10 swept; also dendrogram |
| `dbscan` | DBSCAN | eps grid with min_samples {3,5}; k-distance plot to justify eps |
Metrics: silhouette (computed on non-noise points only, with **noise fraction reported alongside**; undefined if <2 clusters → NaN), n_clusters, noise fraction, Davies–Bouldin (optional). No ground truth exists: say so explicitly. Do **not** crown a winner on silhouette alone (it favours convex clusters and is not comparable across very different k): also show cluster sizes and scatter plots. Clustering uses all 200 rows (no split); stability uses multiple seeds for noise draws and k-means init.

## 6. Perturbations (Robustness Lab)
All applied through `src/arena/perturb.py` with explicit RNG, returning a *new* array.
Work in the preprocessed (standardized) space over **numeric columns only**; categorical one-hots are left intact for noise (document it).

| Family | Applied to | Levels | Details |
|---|---|---|---|
| `noise` | test X | σ ∈ {0.1, 0.25, 0.5, 1.0} (in std units) | Add N(0, σ²) to numeric columns |
| `missing` | test X | p ∈ {0.1, 0.2, 0.3, 0.5} | Mask entries independently (MCAR) in numeric columns; re-impute with **train** medians (value from fitted imputer) |
| `drop_feature` | test X | top-1, top-2 features by permutation importance of a reference model (RF, seed's clean fit) | Replace feature with train mean (ablation). Targeted feature loss |
| `imbalance` (clf only) | **train** X,y | minority share ∈ {original (~38%), 20%, 10%, 5%} | Randomly downsample positives (minority of the *new* set); test set untouched; retrain with same hyperparameters |
Regression: noise, missing, drop_feature. Clustering: noise only (see §6.2).

### 6.1 Robustness metrics (per model, per family, per level, per seed)
Record: `baseline_score`, `perturbed_score`, `delta` (perturbed − baseline), `rel_change` (delta / |baseline|, sign-adjusted so that negative = worse for every metric, including error metrics RMSE/MAE where "higher is worse"), perturbation params, seed.
Summaries per model & family: mean `rel_change` across levels; **degradation-curve area** (trapezoid of score vs. level, normalized by baseline); **worst-case drop**; **rank** at each level; **rank flips vs. baseline**; **Kendall τ** between baseline ranking and each level's ranking; top-1 winner at each level.
Primary robustness metric: clf macro-F1 (also report accuracy); reg RMSE.

### 6.2 Clustering stability
For each algorithm at its selected settings: cluster clean data, cluster noised data (σ ∈ {0.1,0.25,0.5} in std units, 5 seeds), compute **ARI** between the two labelings (noise points treated as their own label -1) and silhouette change. Higher ARI = more stable.

## 7. Hypothesis decision rules
Decide using the saved aggregate tables; report the numbers, then the verdict (Supported / Not supported / Inconclusive).
- **H1:** Supported if at some perturbation (family, level) the top-ranked model by primary metric ≠ baseline top model, with the gap > 1 std; Inconclusive if only within-noise flips.
- **H2:** Supported if mean rel_change (noise family) of tree-ensemble group is less negative than kNN and ANN, in the majority of levels.
- **H3:** Supported if kNN/NC have the largest mean relative drop under noise and under missing among all models.
- **H4:** Supported if as minority share falls: accuracy of ≥1 model stays within ~2 points or rises while its macro-F1/minority recall drops ≥ 10 points; and rankings by accuracy vs. macro-F1 differ (Kendall τ < 1).
- **H5:** Complexity proxy = number of effective parameters/leaves or a documented ordinal (1 linear/NC … 5 ensembles/ANN). Compute Spearman ρ between complexity and mean rel_change; Supported if |ρ| < 0.5 or p>0.05 (state small-n caveat: ~8 models).
- **H6:** Supported if mean ARI(single linkage) and ARI(mean shift) < ARI(k-means) at majority of σ levels.

## 8. Failure analysis
- Seed-0 test predictions for every classifier saved: `results/predictions/clf_predictions_seed0.csv` (row id, y_true, each model's pred & proba).
- **Disagreement:** rows where models disagree; "hard core" = rows all models misclassify; "contested" = rows where models split.
- Misclassified inspection table with the original (un-preprocessed) feature values (join via row id).
- Confusion matrices per model; error rates by subgroup (Sex, Pclass, age band).
- Regression: residual vs. prediction plot, worst-error rows, error by target range (the capped $500k houses).
- Importance: native (tree gain/coef) + **permutation importance on the test set** for top models; compare agreement.
- Trade-off view: accuracy vs. robustness vs. fit time (bubble/scatter) + written discussion.

## 9. Result files and schemas (CSV only, long format)
All under `results/`:
- `raw/clf_baseline.csv`, `raw/reg_baseline.csv`: `task,dataset,model,seed,metric,value,fit_time_s,predict_time_s,best_params`
- `raw/robustness_{clf,reg}.csv`: `task,dataset,model,seed,family,level,metric,baseline_score,perturbed_score,delta,rel_change,params_json`
- `raw/clustering.csv`: `dataset,algorithm,params_json,seed,n_clusters,noise_frac,silhouette,davies_bouldin`
- `raw/clustering_stability.csv`: `algorithm,sigma,seed,ari,sil_clean,sil_noisy,n_clusters_noisy`
- `summary/leaderboard_{clf,reg,clustering}.csv`: model × metric mean/std + `rank`, `tie_flag`
- `summary/robustness_summary_{clf,reg}.csv`: model, family, mean_rel_change, auc_norm, worst_drop, rank_flips, kendall_tau
- `predictions/*.csv`, `figures/*.png`
Raw files are produced by scripts; summaries are derived by `src/arena/summarize.py` so figures/tables are always regenerable.
