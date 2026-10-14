import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px
import streamlit as st
from common import header, load

header("Clustering (Mall Customers)", "No ground-truth labels: judged by silhouette (non-noise points), cluster counts, noise fraction and stability under noise.")
feat = st.sidebar.radio("Feature set", ["2d", "3d"], format_func=lambda f: "Income + Spending" if f == "2d" else "Income + Spending + Age")
raw, sel = load("raw/clustering.csv"), load("summary/clustering_selected.csv")
stab, lab = load("summary/clustering_stability_summary.csv"), load("predictions/clustering_labels.csv")
st.subheader("Selected configuration per algorithm (max silhouette; DBSCAN limited to <=30% noise)")
s = sel[sel.features == feat].copy(); st.dataframe(s.drop(columns=["features"]).round(3), hide_index=True, width="stretch")
st.markdown("#### Sweep: silhouette vs. k")
d = raw[(raw.features == feat) & raw.algorithm.isin(["kmeans", "agg_single"])].copy(); d["k"] = d.params_json.map(lambda x: json.loads(x)["k"])
st.plotly_chart(px.line(d.groupby(["algorithm", "k"]).silhouette.mean().reset_index(), x="k", y="silhouette", color="algorithm", markers=True), width="stretch")
st.caption("Single linkage peaks at k=2: it chains and peels off outliers (see Lab G). A high silhouette there is not a good segmentation.")
st.markdown("#### Clusters in standardized Income x Spending space")
algo = st.selectbox("Algorithm", [a for a in s.algorithm])
col = f"{algo}_{feat}"
lab["cluster"] = lab[col].astype(str)
st.plotly_chart(px.scatter(lab, x="Income_z", y="Spending_z", color="cluster", hover_data=["CustomerID"], title=f"{algo} ({feat}); cluster -1 = DBSCAN noise"), width="stretch")
st.markdown("#### Stability under noise (ARI vs. clean clustering; 1 = identical)")
q = stab[stab.features == feat]
st.plotly_chart(px.line(q, x="sigma", y="ari_mean", color="algorithm", markers=True, error_y="ari_std"), width="stretch")
