import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.graph_objects as go
import streamlit as st
from common import header, load, pretty, task_picker

header("Leaderboards", "Mean ± std over seeds. Same train/test split for every model within a seed; tuning on train only.")
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
st.success(f"Best: **{pretty(best.model)}** = {best['mean']:.4f} ± {best['std']:.4f}  ({int(best.n_seeds)} seeds)")
ties = d[d.tie_with_best]
if len(ties):
    st.warning("Within 1 std of the best (not a clear win): " + ", ".join(pretty(m) for m in ties.model))
colors = ["#2a9d8f" if b else ("#e9c46a" if t else "#8d99ae") for b, t in zip(d.is_best, d.tie_with_best)]
fig = go.Figure(go.Bar(y=[pretty(m) for m in d.model], x=d["mean"], error_x=dict(type="data", array=d["std"].fillna(0)),
                       orientation="h", marker_color=colors))
fig.update_layout(yaxis=dict(autorange="reversed"), height=380, xaxis_title=metric, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Green = best, yellow = tie with best (gap < 1 std of the leader), grey = clearly behind.")
tbl = d[["rank", "model", "mean", "std", "tie_with_best", "fit_time_s"]].rename(columns={"fit_time_s": "fit time (s, incl. tuning)"})
tbl["model"] = tbl.model.map(pretty)
st.dataframe(tbl.round(4), hide_index=True, width="stretch")
with st.expander("All metrics (mean over seeds)"):
    st.dataframe(lb.pivot(index="model", columns="metric", values="mean").round(4), width="stretch")
with st.expander("Per-seed values and tuned hyperparameters"):
    st.dataframe(raw[raw.metric == metric][["model", "seed", "value", "best_params"]], hide_index=True, width="stretch")
