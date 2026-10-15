import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px
import streamlit as st
from common import header, how_to_read, load, pretty, takeaway

header("Failure Investigator 🔍", "Scores hide the story. Here we look at *which* cases the models get wrong, and what they rely on.")
how_to_read("Each **tab** asks one question about mistakes. Start at the left and move right.",
            "Everything here uses one held-out test set (seed 0), so you can inspect individual passengers and houses.",
            "Tip: failures are not just 'bad luck' - they often show *who* the model is unfair to, or what it secretly depends on.")
tabs = st.tabs(["Who is hard to predict?", "Where do errors go?", "Which groups suffer?", "What do models rely on?", "House prices", "Accuracy vs. reliability"])

cases = load("summary/failures_clf_cases.csv")
models = [c[:-5] for c in cases.columns if c.endswith("_pred")]
with tabs[0]:
    st.markdown("Every test passenger falls in one bucket: **all_correct** (every model got it right), **contested** (models disagreed) or **hard_core** (every model was wrong).")
    counts = cases.category.value_counts()
    tot = int(counts.sum())
    takeaway(f"Of {tot} test passengers, {counts.get('all_correct', 0)} were easy for everyone, {counts.get('contested', 0)} split the models, and "
             f"**{counts.get('hard_core', 0)} ({counts.get('hard_core', 0) / tot:.0%}) fooled every single model** - likely unpredictable from the data we have.")
    st.bar_chart(counts)
    cat = st.selectbox("Show passengers in category", ["hard_core", "contested"], help="hard_core = all models wrong; contested = models split.")
    cols = ["row_id", "y_true", "Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "n_wrong"] + [f"{m}_pred" for m in models]
    sub = cases[cases.category == cat]
    sex = st.multiselect("Filter by Sex", sorted(sub.Sex.unique()), default=list(sub.Sex.unique()))
    st.caption("y_true: 1 = survived, 0 = did not. `*_pred` columns: each model's guess. n_wrong = how many models erred.")
    st.dataframe(sub[sub.Sex.isin(sex)][cols], hide_index=True, width="stretch")
with tabs[1]:
    st.markdown("A **confusion matrix** counts every kind of right and wrong answer. Rows = what really happened, columns = what the model predicted.")
    cm = load("summary/failures_clf_confusion.csv")
    m = st.selectbox("Model", models, format_func=pretty)
    t = cm[cm.model == m].pivot(index="true", columns="pred", values="count")
    try:
        tn, fp, fn, tp = int(t.loc[0, 0]), int(t.loc[0, 1]), int(t.loc[1, 0]), int(t.loc[1, 1])
        takeaway(f"{pretty(m)} correctly spotted {tp} of {tp + fn} survivors and {tn} of {tn + fp} non-survivors. "
                 f"It wrongly predicted death for {fn} survivors and wrongly predicted survival for {fp} who died.")
    except KeyError:
        pass
    st.plotly_chart(px.imshow(t, text_auto=True, color_continuous_scale="Blues", labels=dict(x="predicted (1 = survived)", y="true (1 = survived)")), width="content")
with tabs[2]:
    st.markdown("Overall accuracy can hide that a model is much worse for some groups. Pick a way to slice the passengers and compare **error rates** (lower = better).")
    sg = load("summary/failures_clf_subgroups.csv")
    grp = st.selectbox("Group passengers by", sg.group.unique())
    q = sg[sg.group == grp].copy(); q["model"] = q.model.map(pretty)
    worst = q.sort_values("error_rate", ascending=False).iloc[0]
    takeaway(f"The hardest group under '{grp}' is **{worst.value}** (n={int(worst.n)}), where {worst.model} gets {worst.error_rate:.0%} of cases wrong. "
             "Small groups mean noisy percentages, so hover to see n.")
    st.plotly_chart(px.bar(q, x="value", y="error_rate", color="model", barmode="group", hover_data=["n"], labels={"value": grp, "error_rate": "error rate"}), width="stretch")
with tabs[3]:
    st.markdown("**Permutation importance** shuffles one column at a time. If the score collapses, the model was leaning on that column heavily.")
    imp = load("summary/failures_importance_clf.csv")
    ms = st.multiselect("Models to compare", imp.model.unique(), default=list(imp.model.unique())[:3], format_func=pretty)
    q = imp[imp.model.isin(ms)].copy(); q["model"] = q.model.map(pretty)
    if len(q):
        top = q.groupby("feature").perm_importance.mean().sort_values(ascending=False)
        takeaway(f"Averaged over the chosen models, **{top.index[0]}** matters most (shuffling it costs the most score), followed by {top.index[1] if len(top) > 1 else '-'}.")
    st.plotly_chart(px.bar(q, x="feature", y="perm_importance", color="model", barmode="group", labels={"perm_importance": "drop in macro-F1 when shuffled"}), width="stretch")
    nat = q.dropna(subset=["native_importance"])
    if len(nat):
        st.markdown("Tree models also report their own 'native' importance (how much each column helped the splits):")
        st.plotly_chart(px.bar(nat, x="feature", y="native_importance", color="model", barmode="group"), width="stretch")
with tabs[4]:
    st.markdown("Where do the **house-price** predictions go wrong? Errors are grouped by the true price. The last group is prices *capped at $500k* in the dataset, which models can never reach.")
    rg = load("summary/failures_reg_by_range.csv")
    rg["model"] = rg.model.map(pretty)
    st.plotly_chart(px.bar(rg, x="target_range", y="mae", color="model", barmode="group", labels={"mae": "average error (×$100k)", "target_range": "true price range (×$100k)"}), width="stretch")
    st.plotly_chart(px.bar(rg, x="target_range", y="mean_residual", color="model", barmode="group",
                           title="Bias: above 0 = model guessed too LOW, below 0 = too HIGH"), width="stretch")
    st.markdown("**The 15 worst individual predictions:**")
    st.dataframe(load("summary/failures_reg_worst.csv").head(15).round(3), hide_index=True, width="stretch")
    ir = load("summary/failures_importance_reg.csv"); ir["model"] = ir.model.map(pretty)
    st.plotly_chart(px.bar(ir, x="feature", y="perm_importance", color="model", barmode="group", title="Which columns the price models rely on (error increase when shuffled)"), width="stretch")
with tabs[5]:
    st.markdown("Is the most accurate model also the most reliable? **Right = more accurate on clean data. Higher = less hurt by noise / missing / lost columns.** The ideal model sits in the top-right corner.")
    for task, ttl, flip in (("clf", "Titanic: clean macro-F1 vs. robustness", False), ("reg", "California: clean RMSE vs. robustness (lower RMSE is better, so the ideal is top-LEFT)", True)):
        t = load(f"summary/tradeoff_{task}.csv").dropna(subset=["feature_robustness"]); t["name"] = t.model.map(pretty)
        fig = px.scatter(t, x="score", y="feature_robustness", size="fit_time_s", color="complexity", text="name", title=ttl,
                         labels={"score": "clean score", "feature_robustness": "robustness (higher = better)", "complexity": "complexity (1 simple – 7 complex)"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width="stretch")
        st.caption("Bubble size = training time. Colour = how complex the model is.")
        with st.expander("Show the numbers"):
            st.dataframe(t.drop(columns="name").round(3), hide_index=True, width="stretch")
