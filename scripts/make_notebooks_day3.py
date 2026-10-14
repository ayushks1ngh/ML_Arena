"""Builds notebooks 04 (clustering) and 05 (robustness lab)."""
import nbformat as nbf
C = lambda t: nbf.v4.new_code_cell(t); M = lambda t: nbf.v4.new_markdown_cell(t)
SETUP = ("import sys, json; sys.path.insert(0, '../src')\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from arena.config import RAW_DIR, SUMMARY_DIR, PRED_DIR\npd.set_option('display.width', 180)")

nb = nbf.v4.new_notebook(); nb.cells = [
M("# 04 - Clustering arena (Mall Customers)\nNo ground-truth labels exist, so we judge clusterings by internal metrics (silhouette on non-noise points, Davies-Bouldin), cluster counts, noise fraction, and **stability under noise** (ARI against the clean clustering). Features: standardized Annual Income + Spending Score ('2d'); '3d' adds Age."),
C(SETUP + "\nraw = pd.read_csv(RAW_DIR/'clustering.csv'); sel = pd.read_csv(SUMMARY_DIR/'clustering_selected.csv')\nstab = pd.read_csv(SUMMARY_DIR/'clustering_stability_summary.csv')\nlab = pd.read_csv(PRED_DIR/'clustering_labels.csv'); mall = pd.read_csv('../data/raw/mall/Mall_Customers.csv')"),
M("## Selected configuration per algorithm (max silhouette; DBSCAN limited to <=30% noise)"),
C("sel.drop(columns=['params_json']).assign(params=sel.params_json).round(3)"),
M("## Sweep: silhouette vs. k (K-means, single linkage)"),
C("d = raw[(raw.features=='2d') & raw.algorithm.isin(['kmeans','agg_single'])].copy(); d['k'] = d.params_json.map(lambda s: json.loads(s)['k'])\nfig, ax = plt.subplots(figsize=(6,3.5))\nfor a, g in d.groupby('algorithm'):\n    g = g.groupby('k').silhouette.mean(); ax.plot(g.index, g.values, marker='o', label=a)\nax.set_xlabel('k'); ax.set_ylabel('silhouette'); ax.legend(); plt.show()"),
M("Single linkage chains: it peels off a few outliers first, so its silhouette is best at k=2 (one or a few outlier points vs. everyone else). That is **the chaining effect**, not a good segmentation."),
C("fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))\nfor ax, a in zip(axes, ['kmeans','meanshift','agg_single','dbscan']):\n    ax.scatter(mall['Annual Income (k$)'], mall['Spending Score (1-100)'], c=lab[a+'_2d'], cmap='tab10', s=18)\n    s = sel[(sel.features=='2d') & (sel.algorithm==a)].iloc[0]; ax.set_title(f\"{a}: k={s.n_clusters}, sil={s.silhouette:.2f}, noise={s.noise_frac:.0%}\", fontsize=9)\n    ax.set_xlabel('Income'); \nplt.tight_layout()"),
M("## Stability under noise (ARI vs. clean clustering; 1.0 = identical)"),
C("p = stab[stab.features=='2d'].pivot(index='sigma', columns='algorithm', values='ari_mean'); print(p.round(3)); p.plot(marker='o', figsize=(6,3.5)); plt.ylabel('mean ARI'); plt.title('2d features'); plt.show()"),
M("Single-linkage ARI is erratic (huge std): whether it isolates the same outliers changes with the noise draw. Compare with k-means in the table above. (H6 verdict is computed in the report notebook step on Day 4.)"),
]
nbf.write(nb, "notebooks/04_clustering_arena.ipynb")

nb = nbf.v4.new_notebook(); nb.cells = [
M("# 05 - Robustness Lab\nTest-time perturbations of a **fixed tuned model** (noise, missing values, drop top features) and **retraining** under class imbalance. `rel_change` is oriented so negative = worse for every metric. All numbers come from `results/raw/robustness_*.csv` (5 seeds)."),
C(SETUP + "\nfrom arena.config import FIG_DIR\nimport seaborn as sns\nrc = pd.read_csv(RAW_DIR/'robustness_clf.csv'); rr = pd.read_csv(RAW_DIR/'robustness_reg.csv')"),
M("## Classification: degradation curves (macro-F1 relative change vs. severity)"),
C("def curves(df, metric, title):\n    d = df[df.metric==metric]; fams = d.family.unique()\n    fig, axes = plt.subplots(1, len(fams), figsize=(4.2*len(fams), 3.6), squeeze=False)\n    for ax, f in zip(axes[0], fams):\n        g = d[d.family==f].groupby(['model','level']).rel_change.mean().unstack(0)\n        g.plot(ax=ax, marker='o', legend=False); ax.set_title(f); ax.axhline(0, color='k', lw=.5)\n    axes[0][0].legend(fontsize=7); fig.suptitle(title); plt.tight_layout(); return fig\nfig = curves(rc, 'f1_macro', 'Classification (Titanic): macro-F1 change'); fig.savefig(FIG_DIR/'robustness_clf_curves.png', dpi=130); plt.show()"),
M("## Mean relative change per model and family (negative = worse)"),
C("sc = pd.read_csv(SUMMARY_DIR/'robustness_summary_clf.csv')\nsns.heatmap(sc.pivot(index='model', columns='family', values='mean_rel_change'), annot=True, fmt='.2f', cmap='RdYlGn', center=0); plt.show()\nsc.round(3)"),
M("## Is the clean-data winner the most robust? Rank stability vs. baseline ranking"),
C("st = pd.read_csv(SUMMARY_DIR/'robustness_rank_stability_clf.csv')\nst.assign(clearly_changed=(st.top1_changed) & (st.gap_vs_baseline_winner > st.top1_std)).round(3)"),
M("`clearly_changed` = a different model tops the ranking AND the gap to the old winner exceeds the new winner's seed std. Otherwise a flip may be noise."),
M("## Imbalance: accuracy vs. macro-F1 vs. minority recall (mean over seeds)"),
C("im = rc[rc.family=='imbalance'].groupby(['model','level','metric']).perturbed_score.mean().unstack('metric').round(3)\nbase = rc[rc.family=='noise'].groupby(['model','metric']).baseline_score.mean().unstack('metric').round(3)\nprint('baseline (original balance)'); print(base); im"),
C("fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))\nfor ax, m in zip(axes, ['accuracy','f1_macro','recall']):\n    im[m].unstack(0).plot(ax=ax, marker='o', legend=False); ax.set_title(m); ax.set_xlabel('minority share in train')\naxes[0].legend(fontsize=7); plt.tight_layout(); plt.show()"),
M("## Regression (California): RMSE change"),
C("fig = curves(rr, 'rmse', 'Regression (California): RMSE relative change (negative = worse)'); fig.savefig(FIG_DIR/'robustness_reg_curves.png', dpi=130); plt.show()\npd.read_csv(SUMMARY_DIR/'robustness_summary_reg.csv').round(3)"),
C("pd.read_csv(SUMMARY_DIR/'robustness_rank_stability_reg.csv').round(3)"),
]
nbf.write(nb, "notebooks/05_robustness_lab.ipynb")
