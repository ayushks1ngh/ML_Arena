# ML Arena Plan

## Hypotheses (to be tested, not assumed)
- H1: The best baseline model is not best under every perturbation.
- H2: Ensembles lose less than kNN/ANN under feature noise.
- H3: Distance-based models suffer most from noise and missing features.
- H4: Under imbalance, accuracy hides minority-class failure (compare macro-F1).
- H5: Model complexity does not predict robustness.

## Protocol
5 seeds, mean +/- std; preprocessing fit on train only; perturb test set (imbalance: train set).
Metrics: clf accuracy/precision/recall/F1; reg MAE/RMSE/R2; clustering silhouette, #clusters, noise fraction.

## Days
1 setup + data | 2 leaderboards | 3 clustering + robustness | 4 imbalance, failures, ID3 |
5 dashboard + syllabus notebooks | 6 docs/report/viva | 7 buffer
