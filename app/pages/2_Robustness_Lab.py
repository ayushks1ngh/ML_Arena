import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.express as px
import streamlit as st
from common import FAMILY_NAMES, FAMILY_STORY, METRIC_INFO, glossary_expander, header, how_to_read, load, pretty, takeaway, task_picker

header("Robustness Lab 🧪", "Does the model that wins on clean data stay on top when the data gets harder?")
how_to_read("Pick a **task** and a **type of damage** in the sidebar. Read the 'real-world version' box to see what the damage means.",
            "The **line chart** shows each model's score change as the damage grows. 0% = unaffected; **below 0 = worse** (this is flipped for error scores, so down is always bad).",
            "Use the **slider** to pick one damage level and see who coped best and who suffered most.",
            "The **rank chart** shows if the *ordering* changes. If lines cross, the clean-data champion was dethroned.")
task, label = task_picker()
raw = load(f"raw/robustness_{task}.csv")
summ = load(f"summary/robustness_summary_{task}.csv")
stab = load(f"summary/robustness_rank_stability_{task}.csv")
ranks = load(f"summary/robustness_ranks_{task}.csv")
fams = list(raw.family.unique())
fam = st.sidebar.selectbox("Type of damage", fams, format_func=lambda f: FAMILY_NAMES.get(f, f))
metric = st.sidebar.selectbox("Score", list(raw.metric.unique()), index=list(raw.metric.unique()).index("f1_macro" if task == "clf" else "rmse"),
                              help="Which score to track. Hover the question mark on the Leaderboards page for meanings.")
st.subheader(f"{label} · {FAMILY_NAMES.get(fam, fam)}")
st.info(FAMILY_STORY.get(fam, ""))
st.caption(METRIC_INFO.get(metric, ("", ""))[1])

d = raw[(raw.family == fam) & (raw.metric == metric)]
g = d.groupby(["model", "level"]).agg(rel=("rel_change", "mean"), sd=("rel_change", "std"), base=("baseline_score", "mean"), pert=("perturbed_score", "mean")).reset_index()
g["model"] = g.model.map(pretty)
g["rel_pct"], g["sd_pct"] = g.rel * 100, g.sd * 100
st.markdown("#### 1. How much does each model's score change?")
fig = px.line(g, x="level", y="rel_pct", color="model", markers=True, error_y="sd_pct",
              labels={"rel_pct": f"% change in {metric} (below 0 = worse)", "level": "damage level →"})
fig.add_hline(y=0, line_dash="dot", annotation_text="no change")
st.plotly_chart(fig, width="stretch")

st.markdown("#### 2. Zoom in on one damage level")
lvl = st.select_slider("Damage level to inspect", sorted(d.level.unique()))
t = g[g.level == lvl].sort_values("rel", ascending=False)
best_m, worst_m = t.iloc[0], t.iloc[-1]
takeaway(f"At damage level {lvl}, **{best_m.model}** coped best ({best_m.rel_pct:+.1f}% change) and **{worst_m.model}** suffered most ({worst_m.rel_pct:+.1f}%).")
t = t.rename(columns={"base": "score before damage", "pert": "score after damage", "rel_pct": "% change", "sd_pct": "wobble (±%)"})
with st.expander("Show the numbers"):
    st.dataframe(t[["model", "score before damage", "score after damage", "% change", "wobble (±%)"]].round(3), hide_index=True, width="stretch")

st.markdown("#### 3. Does the ranking change?")
r = ranks[ranks.family == fam].copy(); r["model"] = r.model.map(pretty)
fig = px.line(r, x="level", y="rank", color="model", markers=True, labels={"rank": "rank (1 = best)", "level": "damage level →"})
fig.update_yaxes(autorange="reversed", dtick=1)
st.plotly_chart(fig, width="stretch")
s = stab[stab.family == fam].copy(); s["clearly_changed"] = s.top1_changed & (s.gap_vs_baseline_winner > s.top1_std)
n_chg, n_clear = int(s.top1_changed.sum()), int(s.clearly_changed.sum())
base_name = pretty(s.baseline_top1.iloc[0])
if n_chg:
    takeaway(f"The clean-data champion (**{base_name}**) was knocked off #1 at {n_chg} of {len(s)} damage levels"
             f"{f', and {n_clear} of those are clear (bigger than the random wobble)' if n_clear else ', but none are clearly bigger than the random wobble'}.")
else:
    takeaway(f"The clean-data champion (**{base_name}**) stayed #1 at every damage level here.")
s["top1"], s["baseline_top1"] = s.top1.map(pretty), s.baseline_top1.map(pretty)
with st.expander("Show the numbers (rank stability table)"):
    st.dataframe(s[["level", "baseline_top1", "top1", "top1_changed", "clearly_changed", "kendall_tau", "rank_flips"]].round(3), hide_index=True, width="stretch")
    st.caption("`clearly_changed`: a different model is #1 AND its lead exceeds its seed-to-seed wobble. `kendall_tau` = 1 means the order is untouched; lower means more shuffling.")

st.markdown("#### 4. The big picture: who copes best across *all* damage types?")
p = summ.pivot(index="model", columns="family", values="mean_rel_change") * 100
p.index = [pretty(m) for m in p.index]
p.columns = [FAMILY_NAMES.get(c, c) for c in p.columns]
st.plotly_chart(px.imshow(p, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto",
                          labels=dict(color="% change")), width="stretch")
st.caption("Green = barely affected, red = hurt badly. Each cell is the average % change over all levels and repeats of that damage type.")

if task == "clf":
    st.markdown("#### 5. Why accuracy can lie when one outcome is rare")
    st.write("If only 5% of passengers survived, a model that *always says 'did not survive'* is 95% accurate - and useless. Watch **accuracy** versus **minority recall** (how many of the rare cases were actually found).")
    im = raw[raw.family.isin(["imbalance", "imbalance_matched"]) & raw.metric.isin(["accuracy", "f1_macro", "recall"])]
    sel = st.radio("Protocol", ["imbalance", "imbalance_matched"], format_func=lambda f: "Train skewed, test balanced" if f == "imbalance" else "Train AND test skewed (the world really is lopsided)", horizontal=True)
    q = im[im.family == sel].groupby(["model", "level", "metric"]).perturbed_score.mean().reset_index(); q["model"] = q.model.map(pretty)
    q["metric"] = q.metric.replace({"recall": "minority recall (rare cases found)"})
    st.plotly_chart(px.line(q, x="level", y="perturbed_score", color="model", facet_col="metric", markers=True,
                            labels={"level": "rare-class share in training →", "perturbed_score": "score"}), width="stretch")
    st.caption("In the 'train AND test skewed' view accuracy goes *up* while minority recall goes *down* - the classic trap.")
glossary_expander(["Robustness", "Perturbation", "Rank stability", "Class imbalance", "Mean ± std"])
