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
