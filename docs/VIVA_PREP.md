# Viva Prep (grows each phase)

Rules: answers here that cite numbers must be filled in from real results only (marked `[FILL from results]`). Add 4–8 Q&As after each phase.

## A. Project-level
**Q: What is the project's central question?** Which algorithms perform best per task, and does the best one on clean data stay best when data gets noisier, incomplete or imbalanced?
**Q: Why is robustness more interesting than accuracy alone?** Real data is noisy and incomplete; a model that wins a clean benchmark may degrade faster. We test it experimentally via rank stability, not assumption.
**Q: Why these three datasets?** Small, public, cover classification (mixed types + missing), regression (large numeric), clustering (visualizable). Limitations: Titanic small test set; Mall small and likely synthetic with no ground truth; California target capped at $500k.
**Q: Which result was surprising / which hypotheses failed?** `[FILL from results]`

## B. Methodology
**Q: What is data leakage and how did you prevent it?** Using information from test data during training/preprocessing. We split first, fit imputer/scaler/encoder inside a pipeline on train only, tune with inner CV on train only.
**Q: Why multiple seeds?** One split can be lucky; mean±std shows variance. Differences within 1 std are reported as ties.
**Q: Why macro-F1 for tuning and robustness?** Accuracy can reward ignoring the minority class; macro-F1 averages per-class F1.
**Q: Why scale features?** kNN, Nearest Centroid, MLP and clustering depend on distances/gradients; trees don't care.
**Q: How is noise defined?** Gaussian noise with σ in units of each feature's training standard deviation, added to test numeric features only.
**Q: Why perturb only the test set (except imbalance)?** To measure robustness of an already-trained model to deployment shift; imbalance is a training-data property, so train is changed and the model retrained.
**Q: How do you evaluate clustering without labels?** Silhouette (with noise handling), number of clusters, stability under noise (ARI), visual inspection.
**Q: What does the silhouette score measure and what are its limits?** `(b-a)/max(a,b)` per point; favours convex, similar-size clusters; undefined for 1 cluster; not comparable across very different k.

## C. Algorithms (add details when implemented)
- Decision tree / entropy / information gain / **ID3** (worked example: PlayTennis, gain(Outlook)=0.247)
- kNN (distance metrics, curse of dimensionality, scaling)
- Nearest Centroid (relation to k-means, linear-ish boundary)
- ANN (layers, activation functions, backprop, early stopping)
- Random Forest (bagging, feature subsampling, variance reduction)
- AdaBoost (re-weighting misclassified samples, weak learners)
- Gradient Boosting / XGBoost (additive trees on residual gradients, regularization)
- Linear regression (least squares, assumptions), Ridge/Lasso (regularization = MAP link)
- K-means (objective, init, k selection), Mean Shift (bandwidth, mode seeking), Single linkage (chaining), DBSCAN (eps/min_samples, noise points)

## D. Syllabus labs
- **ML vs. MAP:** MLE maximizes P(D|h); MAP maximizes P(D|h)P(h). With a uniform prior they coincide; with little data the prior matters.
- **Perceptron NAND:** weights/bias, step activation, linear separability; XOR impossible for a single perceptron.
- **Activation functions:** sigmoid/tanh saturate (vanishing gradients), ReLU cheap/sparse (dying ReLU), identity makes the net linear.
- **Q-learning:** `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]`; off-policy; exploration vs. exploitation (ε-greedy).
- **Manhattan / Chebyshev:** L1 = Σ|Δ|, L∞ = max|Δ|; different neighbourhood shapes.
- **Single linkage:** cluster distance = min pairwise distance; merges closest pair each step; chaining effect.

## E. Anticipated hard questions
- Why is XGBoost (not) the winner here? `[FILL from results]`
- Is 5 seeds enough? Why no statistical test? (small n; paired comparisons/ties; limitation)
- Why not retrain under test-time noise? (we measure deployed-model robustness; noise-aware training is future work)
- What would you do with more time? (noise-augmented training, more datasets, calibration, formal significance tests)

## F. Added after Day 2 (real numbers from results/summary/leaderboard_*.csv)
**Q: Which classifier won on Titanic and is it a clear win?** XGBoost had the best macro-F1 (0.834 +- 0.017 over 5 seeds), but RF (0.828) and the decision tree (0.825) are within one std, so we call those ties, not wins.
**Q: Which regressor won on California Housing?** XGBoost RMSE 0.449 (x $100k), then gradient boosting 0.465, random forest 0.503, tree 0.598, linear 0.729; the dummy (predict the mean) is 1.148. Gaps here exceed the std (~0.005-0.014), so the ordering is reliable.
**Q: How did you tune without leaking?** `GridSearchCV` wraps the whole `Pipeline(preprocessor, estimator)`, so within each inner fold the imputer/scaler are refit on that fold's training part only.
**Q: Why is the Dummy model in the table?** A sanity floor: any real model must beat predicting the majority class / mean.
**Q: Why does Nearest Centroid do worst among real classifiers?** It models each class by a single mean point (linear-ish boundary), too simple for Titanic's interactions (Sex x Pclass).

## G. Findings and design decisions (numbers from `results/`, see `docs/REPORT.md`)
**Q: Did XGBoost win?** On clean data yes (Titanic macro-F1 0.834 ± 0.017; California RMSE 0.449 ± 0.008), but on Titanic Random Forest and the Decision Tree are within one std (ties). We only call it a winner on California where the gaps exceed the std.
**Q: Is the best clean model the most robust?** No. In 13 of 16 classification and 9 of 10 regression perturbation cells a different model ranks first. Random Forest leads under missing values and noise on Titanic.
**Q: Which hypotheses failed and why?** H2 (ensembles beat kNN/ANN under noise): XGBoost and the Decision Tree, not kNN, lost most. H3 (distance models suffer most): Nearest Centroid and kNN were among the least affected. H4 under a balanced test: accuracy fell too. H5 for regression: ρ = -0.97 (complex = less robust, n = 5). We report these honestly.
**Q: Why did H4 need two protocols?** If the test set stays balanced, a model trained on 5% positives predicts "died" too often and accuracy drops. Accuracy only *hides* the failure when the test set is skewed like the training set: then accuracy rose 5-9 points while minority recall fell 30-70 points.
**Q: Why is Nearest Centroid "most robust" but not the best model?** Relative loss is small because it starts low (0.741 macro-F1) and uses only class means, which noise barely moves. Robustness must be read together with absolute performance.
**Q: What does `rel_change = -1.7` mean for regression?** For error metrics we flip the sign so negative is always worse; -1.7 means RMSE rose by 170% (e.g. from 0.45 to 1.2).
**Q: Why did tree models collapse under noise on California?** Noise was added to all numeric features, including Latitude/Longitude, which trees use for sharp location splits (permutation importance ~1.0). At σ = 0.5 their RMSE (1.1-1.4) is comparable to the dummy's 1.15.
**Q: Why does Linear Regression degrade least under noise on California?** It averages over many features and is smooth; trees make hard threshold decisions on perturbed values. It still has the worst clean RMSE (0.729).
**Q: Why do single-linkage clusters look bad?** Single linkage merges by the *closest* pair, so it chains and isolates outliers; its best silhouette was at k = 2 and its ARI under noise was erratic (0.19-0.40).
**Q: Why not rank clustering algorithms by silhouette?** There are no labels; silhouette favours convex clusters, is undefined for one cluster, ignores DBSCAN's noise points and is not comparable across different k. We also report counts, noise fraction, visual plots and ARI stability.
**Q: How did you pick DBSCAN parameters?** Grid over eps {0.2-0.6} and min_samples {3, 5}; chose the best silhouette among settings with at most 30% noise (eps 0.3, min_samples 5 gave 7 clusters and 17.5% noise in 2d).
**Q: Why exclude ID3 from robustness?** ID3 works on integer-coded categories, so Gaussian noise or median-filling has no meaning; we compare ID3 with an sklearn entropy tree on the same discretized inputs instead (macro-F1 0.796 vs 0.809, a tie within std).
**Q: How does your ID3 work?** At each node pick the feature with maximum information gain `H(y) - Σ |y_v|/|y| H(y_v)`, create one branch per value, recurse until pure / no features / max depth, majority-class leaves. Unit test: PlayTennis entropy 0.940, gain(Outlook) 0.247, Humidity 0.152, Wind 0.048, Temp 0.029; the root is Outlook.
**Q: How are the "missing" perturbations leak-free?** Masked entries are replaced with the **train** median (in standardized space), never statistics from the test set.
**Q: Why perturb the test set instead of retraining on noisy data?** To measure the robustness of an already deployed model. Retraining with noise-aware training is future work.
**Q: Why is imbalance a train-set perturbation?** Imbalance is a property of the training data; the model is retrained with the same hyper-parameters, test set unchanged (plus the matched-skew variant).
**Q: What does grouped permutation importance mean?** Shuffle all one-hot columns of one original feature together, measure the score drop; one-hot columns of `Sex` are shuffled jointly so importance is per feature, not per dummy.
**Q: What do the hard-core failures tell us?** 13 of 179 passengers are misclassified by every model: 12 men who survived, 9 in 3rd class; they contradict the dominant pattern, so more features (not a better algorithm) would be needed.
**Q: Why is the California target special?** It is capped at 5.0 ($500k): houses at the cap are systematically under-predicted (residuals +0.5 to +1.2 for tree models).
**Q: How do you avoid trusting one lucky split?** 5 seeds with different splits and seeds, mean ± std, ties flagged when the gap is below one std of the leader.
**Q: Why scikit-learn MLP and not PyTorch?** Syllabus needs a fully connected network; MLPClassifier is enough, fast and reproducible.
**Q: What did Lab B show about activations?** With early stopping the sigmoid MLP stopped after ~22 epochs and underfit (accuracy 0.655); without early stopping all activations reached ~0.83 accuracy. Activations differ mostly in training speed/reliability on this small problem; identity makes the network linear.
**Q: What does Lab A's polynomial sweep show?** With 30 training points, test RMSE explodes as degree rises (train RMSE keeps falling): overfitting.
**Q: ML vs MAP in the lab-test example?** P(disease)=0.008, P(+|disease)=0.98, P(+|healthy)=0.03: likelihood favours disease (h_ML) but the posterior is 0.21 disease vs 0.79 healthy, so h_MAP = healthy.
**Q: How does Q-learning in Lab E learn?** ε-greedy exploration with decay, update `Q ← Q + α(r + γ max Q' - Q)`; mean return rose from -0.97 in the first 100 episodes to 0.89 in the last 100, and the greedy policy avoids both pits and reaches the goal.
**Q: How would you extend the project?** Noise-aware training, more datasets, formal significance tests across seeds, calibration analysis, finer tuning grids, SHAP explanations.
**Q: What are the biggest weaknesses?** Small Titanic test set, small number of models for correlation tests (H5), noise applied to all features simultaneously, Mall data unlabelled with unverified licence.
