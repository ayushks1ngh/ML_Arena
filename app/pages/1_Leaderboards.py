import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.graph_objects as go
import streamlit as st
from common import METRIC_INFO, MODEL_INFO, glossary_expander, header, how_to_read, load, pretty, takeaway, task_picker

header("Leaderboards 🏆", "Who wins on clean, undamaged data? Every model sat the same exam, five times.")
how_to_read("Use the **sidebar** to pick a task (predict survival, or predict house price) and a score.",
            "Each **bar** is a model's average score; the thin **whisker** shows how much it wobbled between the 5 repeats.",
            "🟢 green = best, 🟡 yellow = *too close to call* (within the wobble), ⚪ grey = clearly behind.",
            "The **dummy baseline** is a model that just guesses the most common answer. Anything near it has learned nothing.")
task, label = task_picker()
lb = load(f"summary/leaderboard_{task}.csv")
raw = load(f"raw/{task}_baseline.csv")
metric = st.sidebar.selectbox("Metric", sorted(lb.metric.unique()), index=sorted(lb.metric.unique()).index("f1_macro" if task == "clf" else "rmse"))
show_dummy = st.sidebar.checkbox("Show dummy baseline", True)
d = lb[lb.metric == metric].sort_values("rank")
if not show_dummy:
    d = d[d.model != "dummy"]
lower_better = metric in ("rmse", "mae")
best = d[d.is_best].iloc[0]
st.subheader(f"{label}: {metric} ({'lower' if lower_better else 'higher'} is better)")
st.caption(METRIC_INFO.get(metric, ("", ""))[1])
st.success(f"Best: **{pretty(best.model)}** = {best['mean']:.4f} ± {best['std']:.4f}  ({int(best.n_seeds)} repeats)")
ties = d[d.tie_with_best]
if len(ties):
    st.warning("Within 1 std of the best (not a clear win): " + ", ".join(pretty(m) for m in ties.model))
colors = ["#2a9d8f" if b else ("#e9c46a" if t else "#8d99ae") for b, t in zip(d.is_best, d.tie_with_best)]
fig = go.Figure(go.Bar(y=[pretty(m) for m in d.model], x=d["mean"], error_x=dict(type="data", array=d["std"].fillna(0)),
                       orientation="h", marker_color=colors))
fig.update_layout(yaxis=dict(autorange="reversed"), height=380, xaxis_title=metric, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Green = best, yellow = tie with best (gap < 1 std of the leader), grey = clearly behind.")
dm = lb[(lb.metric == metric) & (lb.model == "dummy")]
if len(dm):
    dv = float(dm.iloc[0]["mean"])
    gain = (dv - best["mean"]) if lower_better else (best["mean"] - dv)
    takeaway(f"{pretty(best.model)} {'leads' if not len(ties) else 'is on top, though ' + str(len(ties)) + ' other model(s) are too close to separate'}. "
             f"A know-nothing guesser scores {dv:.3f}; the best model improves on that by {gain:.3f}.")
tbl = d[["rank", "model", "mean", "std", "tie_with_best", "fit_time_s"]].rename(columns={"fit_time_s": "fit time (s, incl. tuning)"})
tbl["model"] = tbl.model.map(pretty)
st.dataframe(tbl.round(4), hide_index=True, width="stretch")
with st.expander("🧑‍🏫 What do these models actually do?"):
    for m in d.model:
        st.markdown(f"**{pretty(m)}** - {MODEL_INFO.get(m, '')}")
glossary_expander(["Mean ± std", "Tie", "Hyperparameter tuning", "Seed"])
with st.expander("All metrics (mean over seeds)"):
    st.dataframe(lb.pivot(index="model", columns="metric", values="mean").round(4), width="stretch")
with st.expander("Per-seed values and tuned hyperparameters"):
    st.dataframe(raw[raw.metric == metric][["model", "seed", "value", "best_params"]], hide_index=True, width="stretch")
