import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.express as px
import streamlit as st
from common import header, load, pretty, task_picker

header("Robustness Lab", "Does the model that wins on clean data stay on top when the data gets harder? rel_change < 0 always means worse.")
task, label = task_picker()
raw = load(f"raw/robustness_{task}.csv")
summ = load(f"summary/robustness_summary_{task}.csv")
stab = load(f"summary/robustness_rank_stability_{task}.csv")
ranks = load(f"summary/robustness_ranks_{task}.csv")
FAM_HELP = {"noise": "Gaussian noise added to numeric test features (sigma in train-std units)",
            "missing": "numeric test entries lost (MCAR) and replaced by the train median",
            "drop_feature": "top-k most important features removed (set to their train mean)",
            "imbalance": "TRAIN set minority share reduced, model retrained; test set unchanged",
            "imbalance_matched": "train AND test minority share reduced (deployment prior = training prior)"}
fams = list(raw.family.unique())
fam = st.sidebar.selectbox("Perturbation family", fams)
metric = st.sidebar.selectbox("Metric", list(raw.metric.unique()), index=list(raw.metric.unique()).index("f1_macro" if task == "clf" else "rmse"))
st.subheader(f"{label}: {fam}")
st.caption(FAM_HELP[fam])

d = raw[(raw.family == fam) & (raw.metric == metric)]
g = d.groupby(["model", "level"]).agg(rel=("rel_change", "mean"), sd=("rel_change", "std"), base=("baseline_score", "mean"), pert=("perturbed_score", "mean")).reset_index()
g["model"] = g.model.map(pretty)
fig = px.line(g, x="level", y="rel", color="model", markers=True, error_y="sd", labels={"rel": f"relative change in {metric}", "level": "severity level"})
fig.add_hline(y=0, line_dash="dot")
st.plotly_chart(fig, width="stretch")
lvl = st.select_slider("Level to inspect", sorted(d.level.unique()))
t = g[g.level == lvl].rename(columns={"base": "baseline", "pert": "perturbed", "rel": "rel_change", "sd": "std(rel_change)"})
st.dataframe(t.drop(columns="level").sort_values("rel_change", ascending=False).round(4), hide_index=True, width="stretch")

st.markdown("#### Rank stability (primary metric)")
r = ranks[ranks.family == fam].copy(); r["model"] = r.model.map(pretty)
fig = px.line(r, x="level", y="rank", color="model", markers=True, labels={"rank": "rank (1 = best)"})
fig.update_yaxes(autorange="reversed", dtick=1)
st.plotly_chart(fig, width="stretch")
s = stab[stab.family == fam].copy(); s["clearly_changed"] = s.top1_changed & (s.gap_vs_baseline_winner > s.top1_std)
s["top1"], s["baseline_top1"] = s.top1.map(pretty), s.baseline_top1.map(pretty)
st.dataframe(s[["level", "baseline_top1", "top1", "top1_changed", "clearly_changed", "kendall_tau", "rank_flips"]].round(3), hide_index=True, width="stretch")
st.caption("`clearly_changed`: a different model is #1 AND the gap exceeds the new winner's seed-to-seed std. kendall_tau = 1 means the ranking is unchanged.")

st.markdown("#### Overview: mean relative change per model and family")
p = summ.pivot(index="model", columns="family", values="mean_rel_change"); p.index = [pretty(m) for m in p.index]
st.plotly_chart(px.imshow(p, text_auto=".2f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto"), width="stretch")

if task == "clf":
    st.markdown("#### Imbalance: accuracy can hide minority-class failure")
    im = raw[raw.family.isin(["imbalance", "imbalance_matched"]) & raw.metric.isin(["accuracy", "f1_macro", "recall"])]
    sel = st.radio("Protocol", ["imbalance", "imbalance_matched"], format_func=lambda f: "train skewed, balanced test" if f == "imbalance" else "train AND test skewed", horizontal=True)
    q = im[im.family == sel].groupby(["model", "level", "metric"]).perturbed_score.mean().reset_index(); q["model"] = q.model.map(pretty)
    q["metric"] = q.metric.replace({"recall": "minority recall"})
    st.plotly_chart(px.line(q, x="level", y="perturbed_score", color="model", facet_col="metric", markers=True, labels={"level": "minority share in train", "perturbed_score": "score"}), width="stretch")
