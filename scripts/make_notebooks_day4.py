"""Builds notebook 06 (failure investigator), 07 (hypotheses) and syllabus labs D, F, G."""
import nbformat as nbf
C = lambda t: nbf.v4.new_code_cell(t); M = lambda t: nbf.v4.new_markdown_cell(t)
SETUP = ("import sys, json; sys.path.insert(0, '../src')\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
         "from arena.config import RAW_DIR, SUMMARY_DIR, PRED_DIR, FIG_DIR\npd.set_option('display.width', 180); pd.set_option('display.max_columns', 30)")
SETUP_LAB = "import numpy as np, pandas as pd, matplotlib.pyplot as plt\nnp.set_printoptions(precision=3, suppress=True)"

# ---------------- 06 failure investigator
nb = nbf.v4.new_notebook(); nb.cells = [
M("# 06 - Failure Investigator\nSeed-0 test set of Titanic (179 passengers). Where do models agree, where do they fail, and what do they rely on?"),
C(SETUP + "\ncases = pd.read_csv(SUMMARY_DIR/'failures_clf_cases.csv')\ncm = pd.read_csv(SUMMARY_DIR/'failures_clf_confusion.csv')\nsub = pd.read_csv(SUMMARY_DIR/'failures_clf_subgroups.csv')\nimp = pd.read_csv(SUMMARY_DIR/'failures_importance_clf.csv')\nmodels = [c[:-5] for c in cases.columns if c.endswith('_pred')]\nprint(models)"),
M("## 1. Agreement structure\n`hard_core` = every model is wrong; `contested` = models split; `all_correct`."),
C("cases.category.value_counts().to_frame('passengers').assign(share=lambda d: (d.passengers/d.passengers.sum()).round(3))"),
M("### Hard-core failures (all models wrong), with raw features"),
C("cols = ['row_id','y_true','Pclass','Sex','Age','Fare','Embarked','Title','FamilySize','n_wrong']\ncases[cases.category=='hard_core'][cols]"),
M("### Most contested passengers (models split about evenly)"),
C("c = cases[cases.category=='contested'].assign(split=lambda d: (d.n_wrong - len(models)/2).abs()).sort_values('split').head(10)\nc[cols]"),
M("## 2. Confusion matrices per model (rows = true, columns = predicted; 1 = survived)"),
C("n = len(models); fig, axes = plt.subplots(2, (n+1)//2, figsize=(3*((n+1)//2), 5.4)); axes = axes.ravel()\nfor ax, m in zip(axes, models):\n    t = cm[cm.model==m].pivot(index='true', columns='pred', values='count'); sns.heatmap(t, annot=True, fmt='d', cbar=False, cmap='Blues', ax=ax); ax.set_title(m)\nfor ax in axes[n:]: ax.axis('off')\nplt.tight_layout(); plt.savefig(FIG_DIR/'confusion_matrices.png', dpi=130); plt.show()"),
M("## 3. Error rate by subgroup"),
C("sub.pivot_table(index=['group','value'], columns='model', values='error_rate').round(2).join(sub.drop_duplicates(['group','value']).set_index(['group','value']).n)"),
M("## 4. What do models rely on? Permutation importance (test set, drop in macro-F1) vs. native importance"),
C("p = imp.pivot(index='feature', columns='model', values='perm_importance').round(3); display(p.sort_values('xgb', ascending=False))\nnat = imp.pivot(index='feature', columns='model', values='native_importance').dropna(axis=1, how='all').round(3); display(nat)"),
C("print('Spearman rank agreement of permutation importance between models:'); display(imp.pivot(index='feature', columns='model', values='perm_importance').corr(method='spearman').round(2))\nfrom scipy.stats import spearmanr\nfor m in nat.columns:\n    print(m, 'permutation vs native rank agreement: %.2f' % spearmanr(p[m], nat[m])[0])"),
M("## 5. ID3 learned tree (from-scratch implementation, discretized features)"),
C("from arena.evaluate import split\nfrom arena.preprocess import Discretizer\nfrom arena.id3 import ID3Classifier\nXtr, Xte, ytr, yte = split('clf', 0)\nd = Discretizer().fit(Xtr); m = ID3Classifier(max_depth=3).fit(d.transform(Xtr), ytr)\nprint(m.export_text(list(d.cols_), ['died','survived'])); print('test acc: %.3f' % (m.predict(d.transform(Xte)) == yte).mean())"),
M("## 6. Regression failures (California, seed 0)"),
C("rng = pd.read_csv(SUMMARY_DIR/'failures_reg_by_range.csv')\ndisplay(rng.pivot(index='target_range', columns='model', values='mae').round(3))\ndisplay(rng.pivot(index='target_range', columns='model', values='mean_residual').round(3))"),
M("`mean_residual = actual - predicted`: positive = under-prediction. The `capped(5.0)` row shows the ceiling effect: true prices are cut at $500k so models cannot learn them."),
C("w = pd.read_csv(SUMMARY_DIR/'failures_reg_worst.csv'); w.head(10).round(2)"),
C("impr = pd.read_csv(SUMMARY_DIR/'failures_importance_reg.csv'); impr.pivot(index='feature', columns='model', values='perm_importance').round(3).sort_values('xgb', ascending=False)"),
M("## 7. Trade-off: accuracy vs. robustness vs. complexity vs. cost"),
C("tc = pd.read_csv(SUMMARY_DIR/'tradeoff_clf.csv').dropna(subset=['feature_robustness']); tr = pd.read_csv(SUMMARY_DIR/'tradeoff_reg.csv')\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\nfor ax, t, ttl in zip(axes, [tc, tr], ['Titanic (macro-F1)', 'California (RMSE, lower better)']):\n    ax.scatter(t.score, t.feature_robustness, s=60 + 25*t.fit_time_s.clip(upper=20), alpha=.6)\n    for r in t.itertuples(): ax.annotate(f'{r.model} (c={r.complexity})', (r.score, r.feature_robustness), fontsize=8)\n    ax.set_xlabel('clean score'); ax.set_ylabel('mean rel. change under noise/missing/drop (higher = more robust)'); ax.set_title(ttl)\nplt.tight_layout(); plt.savefig(FIG_DIR/'tradeoff.png', dpi=130); plt.show()\ndisplay(tc.round(3)); display(tr.round(3))"),
C("# Data-driven summary (no hand-written numbers)\nbest = tc.sort_values('score').iloc[-1]; most_rob = tc.sort_values('feature_robustness').iloc[-1]\nprint(f\"Titanic: best clean model = {best.model} ({best.score:.3f}); most feature-robust = {most_rob.model} ({most_rob.feature_robustness:+.3f}, clean score {most_rob.score:.3f}).\")\nbr = tr.sort_values('score').iloc[0]; mr = tr.sort_values('feature_robustness').iloc[-1]\nprint(f\"California: best clean model = {br.model} (RMSE {br.score:.3f}); most feature-robust = {mr.model} ({mr.feature_robustness:+.2f}, RMSE {mr.score:.3f}).\")"),
]
nbf.write(nb, "notebooks/06_failure_investigator.ipynb")

# ---------------- 07 hypotheses
nb = nbf.v4.new_notebook(); nb.cells = [
M("# 07 - Hypothesis verdicts\nComputed by `src/arena/hypotheses.py` from the saved result tables using the decision rules in `docs/EXPERIMENT_DESIGN.md` section 7. Verdicts are whatever the data says; nothing is tuned to support a hypothesis."),
C(SETUP + "\nh = pd.read_csv(SUMMARY_DIR/'hypotheses.csv')\nfor r in h.itertuples():\n    print(f'{r.id:38s} -> {r.verdict}\\n    {r.claim}\\n    {r.evidence}\\n')"),
]
nbf.write(nb, "notebooks/07_hypotheses.ipynb")

# ---------------- Lab D: NAND perceptron
nb = nbf.v4.new_notebook(); nb.cells = [
M("# Lab D - Perceptron implementing a NAND gate\nA perceptron outputs `1 if w.x + b > 0 else 0` and learns with `w <- w + lr*(target - output)*x`. NAND is linearly separable, so it is learnable by one perceptron."),
C(SETUP_LAB + "\nX = np.array([[0,0],[0,1],[1,0],[1,1]]); y_nand = np.array([1,1,1,0]); y_xor = np.array([0,1,1,0])\nstep = lambda z: (z > 0).astype(int)\n\ndef train(X, y, lr=0.1, epochs=100, seed=0):\n    rng = np.random.default_rng(seed); w = rng.normal(0, .1, 2); b = 0.0; hist = []\n    for ep in range(epochs):\n        errors = 0\n        for xi, ti in zip(X, y):\n            out = step(np.array([w @ xi + b]))[0]; err = ti - out\n            w += lr * err * xi; b += lr * err; errors += abs(err)\n        hist.append(errors)\n        if errors == 0: break\n    return w, b, hist\n\nw, b, hist = train(X, y_nand)\nprint('learned w =', w, ' b =', round(b, 3), ' converged after', len(hist), 'epochs')\nprint(pd.DataFrame({'x1': X[:,0], 'x2': X[:,1], 'w.x+b': (X @ w + b).round(3), 'output': step(X @ w + b), 'NAND': y_nand}))"),
M("### Hand-set weights (for the viva)\nw = (-1, -1), b = 1.5: (0,0)->1.5>0 ->1; (0,1)->0.5>0 ->1; (1,0)->0.5>0 ->1; (1,1)->-0.5<=0 ->0. This is exactly NAND."),
C("w2, b2 = np.array([-1., -1.]), 1.5; print('hand-set outputs:', step(X @ w2 + b2), '== NAND', y_nand)"),
C("fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))\nax[0].plot(hist, marker='o'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('misclassifications'); ax[0].set_title('NAND training')\nfor c, m in ((0, 'o'), (1, 's')): ax[1].scatter(*X[y_nand==c].T, marker=m, s=120, label=f'NAND={c}')\nxs = np.linspace(-.2, 1.2); ax[1].plot(xs, -(w[0]*xs + b)/w[1], 'k--', label='learned boundary'); ax[1].set_xlim(-.2, 1.2); ax[1].set_ylim(-.2, 1.2); ax[1].legend(); plt.tight_layout(); plt.show()"),
M("### XOR is NOT linearly separable: a single perceptron cannot learn it\nThis is why we need hidden layers (an ANN)."),
C("w, b, hist = train(X, y_xor, epochs=200); print('XOR misclassifications in last epoch:', hist[-1], 'after', len(hist), 'epochs (never reaches 0)'); print('outputs', step(X @ w + b), 'target', y_xor)"),
]
nbf.write(nb, "notebooks/syllabus/lab_d_nand_perceptron.ipynb")

# ---------------- Lab F: distances
nb = nbf.v4.new_notebook(); nb.cells = [
M("# Lab F - Manhattan and Chebyshev distances\nFor points p, q: **Manhattan (L1)** = sum |p_i - q_i|; **Euclidean (L2)** = sqrt(sum (p_i-q_i)^2); **Chebyshev (L-inf)** = max |p_i - q_i|."),
C("import sys; sys.path.insert(0, '../../src')\n" + SETUP_LAB + "\nfrom scipy.spatial import distance\np, q = np.array([1, 2, 3]), np.array([4, 6, 3])\nprint('diff =', p - q)\nprint('Manhattan = |1-4|+|2-6|+|3-3| =', np.abs(p-q).sum(), ' (scipy', distance.cityblock(p, q), ')')\nprint('Euclidean = sqrt(9+16+0) =', np.sqrt(((p-q)**2).sum()), ' (scipy', distance.euclidean(p, q), ')')\nprint('Chebyshev = max(3,4,0) =', np.abs(p-q).max(), ' (scipy', distance.chebyshev(p, q), ')')"),
M("### Unit balls (all points at distance 1 from the origin)"),
C("t = np.linspace(0, 2*np.pi, 400); fig, ax = plt.subplots(figsize=(4, 4))\nax.plot(np.cos(t), np.sin(t), label='Euclidean (circle)')\nax.plot([1,0,-1,0,1], [0,1,0,-1,0], label='Manhattan (diamond)'); ax.plot([1,-1,-1,1,1], [1,1,-1,-1,1], label='Chebyshev (square)')\nax.set_aspect('equal'); ax.legend(fontsize=7, loc='lower right'); plt.show()"),
M("### Bridge to ML: kNN on Titanic with each metric (same split per seed, k=5, 5 seeds)"),
C("from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score, f1_score\nfrom arena.evaluate import split\nfrom arena.preprocess import build_preprocessor\nrows = []\nfor seed in range(5):\n    Xtr, Xte, ytr, yte = split('clf', seed); pre, _ = build_preprocessor(Xtr); Ztr = pre.fit_transform(Xtr); Zte = pre.transform(Xte)\n    for metric in ['manhattan', 'euclidean', 'chebyshev']:\n        pred = KNeighborsClassifier(5, metric=metric).fit(Ztr, ytr).predict(Zte)\n        rows.append(dict(metric=metric, seed=seed, accuracy=accuracy_score(yte, pred), f1_macro=f1_score(yte, pred, average='macro')))\nr = pd.DataFrame(rows).groupby('metric')[['accuracy', 'f1_macro']].agg(['mean', 'std']).round(3); r"),
M("Differences between metrics are small relative to the seed-to-seed std on this small test set; the choice matters more in high dimensions and for data with outliers."),
]
nbf.write(nb, "notebooks/syllabus/lab_f_manhattan_chebyshev.ipynb")

# ---------------- Lab G: single linkage
nb = nbf.v4.new_notebook(); nb.cells = [
M("# Lab G - Single-linkage agglomerative clustering, step by step\nRule: start with every point as its own cluster; repeatedly merge the two clusters with the **smallest distance**, where the distance between clusters is the **minimum** pairwise distance between their members (single linkage). After merging, the new distance to any other cluster is `min(d(A,X), d(B,X))`."),
C("import sys; sys.path.insert(0, '../../src')\n" + SETUP_LAB + "\nfrom scipy.cluster.hierarchy import linkage, dendrogram\nfrom scipy.spatial.distance import pdist, squareform\npts = {'A': (1, 1), 'B': (1.5, 1), 'C': (5, 5), 'D': (3, 4), 'E': (4, 4), 'F': (3, 3.5)}\nnames = list(pts); P = np.array(list(pts.values()))\nD = squareform(pdist(P)); pd.DataFrame(D, index=names, columns=names).round(2)"),
M("### Manual single-linkage algorithm (prints each merge and the updated distance matrix)"),
C("clusters = {n: [n] for n in names}; dist = {(a, b): D[names.index(a), names.index(b)] for a in names for b in names if a < b}\nmanual = []\nwhile len(clusters) > 1:\n    (a, b), d = min(dist.items(), key=lambda kv: kv[1])\n    new = a + b; clusters[new] = clusters.pop(a) + clusters.pop(b); manual.append((a, b, round(d, 3)))\n    print(f'merge {a} + {b} at distance {d:.3f}  -> clusters: {list(clusters)}')\n    newdist = {}\n    for (x, y), v in dist.items():\n        if {x, y} & {a, b}: continue\n        newdist[(x, y)] = v\n    for o in clusters:\n        if o == new: continue\n        da = dist.get((min(a, o), max(a, o)), dist.get((o, a))); db = dist.get((min(b, o), max(b, o)), dist.get((o, b)))\n        newdist[tuple(sorted((new, o)))] = min(da, db)   # single linkage update rule\n    dist = {tuple(sorted(k)): v for k, v in newdist.items()}"),
M("### Verify against SciPy"),
C("Z = linkage(P, method='single'); print('scipy merge heights :', Z[:, 2].round(3)); print('manual merge heights:', [m[2] for m in manual])\nassert np.allclose(sorted(Z[:, 2]), sorted(m[2] for m in manual), atol=1e-3)\nfig, ax = plt.subplots(figsize=(5, 3.5)); dendrogram(Z, labels=names, ax=ax); ax.set_ylabel('merge distance'); plt.show()"),
M("### Chaining effect\nSingle linkage links points through chains of close neighbours (merge distance = the closest pair), so it keeps a whole chain together and splits off only the isolated pair, giving a lopsided partition. Complete linkage (merge distance = the farthest pair) cuts the chain and gives balanced groups. Compare the two-cluster sizes below."),
C("chain = np.array([[i * 1.0, 0] for i in range(8)] + [[0, 5], [0.5, 5.2]])\nfrom scipy.cluster.hierarchy import fcluster\nfor method in ('single', 'complete'):\n    lab = fcluster(linkage(chain, method), 2, 'maxclust'); print(method, 'two-cluster sizes:', np.bincount(lab)[1:])"),
M("Single linkage on the Mall data (notebook 04) shows the same behaviour: it peels off a couple of outliers and its best silhouette is at k=2."),
]
nbf.write(nb, "notebooks/syllabus/lab_g_single_linkage.ipynb")
