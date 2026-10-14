# Syllabus → Experiment map

Rule: benchmark-able algorithms go in the **main platform**; concepts that are not like-for-like comparisons are **separate labs** (small notebooks in `notebooks/syllabus/`, each ≤ 1–2 hours). Do not force unrelated items into leaderboards.

## A. Main platform (comparable, same data, same splits)
| Syllabus item | Where | Notes |
|---|---|---|
| Decision Tree | Classification + Regression arenas | sklearn (entropy for clf) |
| **ID3** | Classification arena + `src/arena/id3.py` | Custom, categorical features after discretization; compared to sklearn entropy tree on identical inputs; PlayTennis unit test (gain(Outlook)=0.247) |
| kNN | Classification arena | scaling required; also tested in Lab F (distance metrics) |
| Nearest Centroid | Classification arena | euclidean vs. manhattan |
| ANN (fully connected) | Classification arena | sklearn `MLPClassifier`; activation study in Lab B |
| Random Forest | Clf + Reg | |
| AdaBoost | Classification | |
| **XGBoost** | Clf + Reg | Required; result not presumed |
| Linear Regression | Regression arena | baseline |
| Gradient Boosting | Regression arena | sklearn |
| Decision Tree Regression | Regression arena | |
| K-means / Mean Shift / Agglomerative (single) / DBSCAN | Clustering arena | internal metrics + stability |

## B. Separate educational labs (not in leaderboards)
| Lab | Syllabus item | Content | Output |
|---|---|---|---|
| A | **Regression types** | On California Housing (subset) or a toy 1-D set: simple, multiple, polynomial (degree sweep → over/underfit), Ridge/Lasso (coefficient shrinkage), plus logistic regression (classification via regression) on Titanic. Short table of "which type when" | notebook + comparison table + plot |
| B | **Activation functions** | Plot sigmoid, tanh, ReLU, Leaky-ReLU, identity + derivatives (vanishing gradient discussion); then train `MLPClassifier(activation=identity/logistic/tanh/relu)` on Titanic (5 seeds) and compare, and show identity ≈ linear model | notebook + plots + table (executed) |
| C | **ML (Maximum Likelihood) vs. MAP hypotheses** | (1) Coin-toss: MLE = k/n, MAP with Beta(α,β) prior, show how prior vs. data size changes the estimate; (2) discrete hypothesis-space example (e.g. disease-test Bayes problem: P(h|D) ∝ P(D|h)P(h), pick h_ML vs. h_MAP); (3) optional bridge: Naive-Bayes-style MAP classification on a Titanic categorical feature, or L2 regularization = Gaussian prior = MAP (link to Ridge in Lab A) | notebook with worked numbers |
| D | **Perceptron: NAND gate** | From-scratch numpy perceptron with step activation + learning rule; truth table, learned weights/bias, convergence epochs, decision boundary plot; hand-set weights (w=-1,-1; b=1.5) for viva; show XOR is not learnable (motivates hidden layers/ANN) | notebook |
| E | **Q-learning / reinforcement learning** | Small gridworld (e.g. 4×4 or FrozenLake-like, implemented in numpy: no gym dependency): Q-table, ε-greedy, α, γ; learning curve (reward per episode); learned policy arrows; one hand-worked Q-update | notebook |
| F | **Manhattan & Chebyshev distances** | Hand-calc on small points with numpy/scipy verification, formulas, unit-ball shapes plot; bridge: kNN on Titanic with euclidean/manhattan/chebyshev (executed, 5 seeds) | notebook |
| G | **Single linkage calculations** | 5–6 points distance matrix; step-by-step merges by hand (min-distance update rule), verify with `scipy.cluster.hierarchy.linkage`, dendrogram; chaining effect demo; link to Mall results | notebook |

## C. Coverage checklist (tick as finished; each must have executed outputs)
- [x] ID3  - [x] kNN  - [x] Nearest Centroid  - [x] ANN  - [x] Random Forest  - [x] AdaBoost  - [x] XGBoost (clf)
- [x] Linear/DT/RF/GB/XGB regression  - [x] K-means  - [x] Mean Shift  - [x] Single linkage  - [x] DBSCAN
- [x] Lab A regression types  - [x] Lab B activations  - [x] Lab C ML/MAP  - [x] Lab D NAND perceptron
- [x] Lab E Q-learning  - [x] Lab F Manhattan/Chebyshev  - [x] Lab G single linkage
