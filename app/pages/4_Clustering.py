import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px
import streamlit as st
from common import CLUSTER_INFO, CLUSTER_NAMES, glossary_expander, header, how_to_read, load, takeaway

header("Clustering 🧩", "No answer key here: can an algorithm discover natural groups of mall customers by itself?")
how_to_read("**Clustering** sorts similar customers into groups *without being told the groups*. Each customer is one dot: right = higher income, up = higher spending score.",
            "Choose which customer facts to use in the **sidebar** (2 facts, or 3 including age).",
            "Compare the algorithms: a good result gives **tidy, separate blobs** and **stays the same when noise is added**.",
            "With no right answer, we judge with the *silhouette* score (higher = tidier) and *stability* (does it survive noise?).")
feat = st.sidebar.radio("Customer facts to use", ["2d", "3d"], format_func=lambda f: "Income + Spending" if f == "2d" else "Income + Spending + Age")
raw, sel = load("raw/clustering.csv"), load("summary/clustering_selected.csv")
stab, lab = load("summary/clustering_stability_summary.csv"), load("predictions/clustering_labels.csv")

with st.expander("🧑‍🏫 What do the four algorithms do?"):
    for k, v in CLUSTER_INFO.items():
        st.markdown(f"**{CLUSTER_NAMES[k]}** - {v}")

st.markdown("#### 1. The best setting found for each algorithm")
s = sel[sel.features == feat].copy()
s["algorithm"] = s.algorithm.map(CLUSTER_NAMES)
s = s.rename(columns={"n_clusters": "groups found", "noise_frac": "share left ungrouped", "silhouette": "silhouette (higher = tidier)", "davies_bouldin": "Davies-Bouldin (lower = tidier)"})
top = s.sort_values("silhouette (higher = tidier)", ascending=False).iloc[0]
takeaway(f"**{top.algorithm}** scored the tidiest groups ({top['silhouette (higher = tidier)']:.2f}) using {int(top['groups found'])} groups. "
         "But 'tidy' is not the same as 'useful' - check the scatter plot below and the stability chart.")
st.dataframe(s.drop(columns=["features", "params_json"], errors="ignore").round(3), hide_index=True, width="stretch")

st.markdown("#### 2. How many groups? (silhouette vs. k)")
d = raw[(raw.features == feat) & raw.algorithm.isin(["kmeans", "agg_single"])].copy(); d["k"] = d.params_json.map(lambda x: json.loads(x)["k"])
d = d.groupby(["algorithm", "k"]).silhouette.mean().reset_index(); d["algorithm"] = d.algorithm.map(CLUSTER_NAMES)
st.plotly_chart(px.line(d, x="k", y="silhouette", color="algorithm", markers=True, labels={"k": "number of groups (k)"}), width="stretch")
st.caption("Single linkage peaks at k=2 because it 'chains': it peels off a few outliers as one group and lumps everyone else together. A high score there is not a good customer segmentation.")

st.markdown("#### 3. See the groups")
algo = st.selectbox("Algorithm", list(sel[sel.features == feat].algorithm), format_func=lambda a: CLUSTER_NAMES.get(a, a))
col = f"{algo}_{feat}"
lab["cluster"] = lab[col].astype(str)
st.plotly_chart(px.scatter(lab, x="Income_z", y="Spending_z", color="cluster", hover_data=["CustomerID"],
                           labels={"Income_z": "income (standardized, 0 = average)", "Spending_z": "spending score (standardized, 0 = average)"},
                           title=f"{CLUSTER_NAMES.get(algo, algo)}: each colour is one discovered group"), width="stretch")
st.caption("Cluster -1 (DBSCAN only) = customers the algorithm judged to be 'noise' that belong to no group. Axes are standardized so income and spending count equally.")

st.markdown("#### 4. Do the groups survive noise?")
st.write("We jiggle every customer's numbers a little (more jiggle to the right) and re-cluster. **ARI = 1** means the groups are identical to the clean ones; near 0 means they were scrambled.")
q = stab[stab.features == feat].copy(); q["algorithm"] = q.algorithm.map(CLUSTER_NAMES)
last = q[q.sigma == q.sigma.max()].sort_values("ari_mean", ascending=False).iloc[0]
takeaway(f"At the strongest noise tested, **{last.algorithm}** kept its groups best (ARI {last.ari_mean:.2f}). A grouping that falls apart under tiny noise should not be trusted.")
st.plotly_chart(px.line(q, x="sigma", y="ari_mean", color="algorithm", markers=True, error_y="ari_std", labels={"sigma": "noise amount →", "ari_mean": "similarity to clean groups (ARI)"}), width="stretch")
glossary_expander(["Clustering", "Silhouette", "ARI (Adjusted Rand Index)", "Mean ± std"])
