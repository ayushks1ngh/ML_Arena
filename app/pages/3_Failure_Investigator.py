import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px
import streamlit as st
from common import header, load, pretty

header("Failure Investigator", "Seed-0 test sets: where do models disagree, who gets it wrong, and what do they rely on?")
tabs = st.tabs(["Agreement", "Confusion matrices", "Subgroup errors", "Importance", "Regression", "Trade-offs"])

cases = load("summary/failures_clf_cases.csv")
models = [c[:-5] for c in cases.columns if c.endswith("_pred")]
with tabs[0]:
    st.markdown("`hard_core` = every model wrong, `contested` = models split, `all_correct`.")
    counts = cases.category.value_counts()
    st.bar_chart(counts)
    cat = st.selectbox("Show category", ["hard_core", "contested"])
    cols = ["row_id", "y_true", "Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "n_wrong"] + [f"{m}_pred" for m in models]
    sub = cases[cases.category == cat]
    sex = st.multiselect("Filter Sex", sorted(sub.Sex.unique()), default=list(sub.Sex.unique()))
    st.dataframe(sub[sub.Sex.isin(sex)][cols], hide_index=True, width="stretch")
with tabs[1]:
    cm = load("summary/failures_clf_confusion.csv")
    m = st.selectbox("Model", models, format_func=pretty)
    t = cm[cm.model == m].pivot(index="true", columns="pred", values="count")
    st.plotly_chart(px.imshow(t, text_auto=True, color_continuous_scale="Blues", labels=dict(x="predicted", y="true (1 = survived)")), width="content")
with tabs[2]:
    sg = load("summary/failures_clf_subgroups.csv")
    grp = st.selectbox("Group by", sg.group.unique())
    q = sg[sg.group == grp].copy(); q["model"] = q.model.map(pretty)
    st.plotly_chart(px.bar(q, x="value", y="error_rate", color="model", barmode="group", hover_data=["n"]), width="stretch")
with tabs[3]:
    imp = load("summary/failures_importance_clf.csv")
    ms = st.multiselect("Models", imp.model.unique(), default=list(imp.model.unique())[:3], format_func=pretty)
    q = imp[imp.model.isin(ms)].copy(); q["model"] = q.model.map(pretty)
    st.markdown("Permutation importance (drop in macro-F1 when a feature is shuffled; computed on the test set)")
    st.plotly_chart(px.bar(q, x="feature", y="perm_importance", color="model", barmode="group"), width="stretch")
    nat = q.dropna(subset=["native_importance"])
    if len(nat):
        st.markdown("Native importance (tree gain) for the same models")
        st.plotly_chart(px.bar(nat, x="feature", y="native_importance", color="model", barmode="group"), width="stretch")
with tabs[4]:
    rg = load("summary/failures_reg_by_range.csv")
    st.markdown("MAE by true price range. The last bin is the capped `$500k` group (target ceiling 5.0).")
    rg["model"] = rg.model.map(pretty)
    st.plotly_chart(px.bar(rg, x="target_range", y="mae", color="model", barmode="group"), width="stretch")
    st.plotly_chart(px.bar(rg, x="target_range", y="mean_residual", color="model", barmode="group", title="mean residual (actual - predicted; >0 = under-prediction)"), width="stretch")
    st.dataframe(load("summary/failures_reg_worst.csv").head(15).round(3), hide_index=True, width="stretch")
    ir = load("summary/failures_importance_reg.csv"); ir["model"] = ir.model.map(pretty)
    st.plotly_chart(px.bar(ir, x="feature", y="perm_importance", color="model", barmode="group", title="permutation importance (RMSE increase)"), width="stretch")
with tabs[5]:
    for task, ttl in (("clf", "Titanic: clean macro-F1 vs. feature robustness"), ("reg", "California: clean RMSE vs. feature robustness")):
        t = load(f"summary/tradeoff_{task}.csv").dropna(subset=["feature_robustness"]); t["name"] = t.model.map(pretty)
        st.plotly_chart(px.scatter(t, x="score", y="feature_robustness", size="fit_time_s", color="complexity", text="name", title=ttl,
                                   labels={"score": "clean score", "feature_robustness": "mean rel. change under noise/missing/drop (higher = more robust)"}), width="stretch")
        st.dataframe(t.drop(columns="name").round(3), hide_index=True, width="stretch")
