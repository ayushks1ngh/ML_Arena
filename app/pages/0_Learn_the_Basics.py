import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from common import GLOSSARY, METRIC_INFO, MODEL_INFO, header, load, pretty

header("Learn the Basics 🎓", "No machine-learning background needed. Read a few lines, then play with a live model.")
t1, t2, t3, t4 = st.tabs(["1 · ML in 2 minutes", "2 · 🎮 Playground", "3 · Meet the models", "4 · Glossary"])

with t1:
    st.markdown("""
### The idea
You show a computer many **examples with the right answer** (the *training data*). It finds patterns. Then you give it **new examples** and it predicts the answer.

### The two golden rules
1. **Grade it on data it has never seen.** We split the data: most goes to *training*, the rest is a hidden *test set*. Scoring on the training data would be like giving a student the exam questions in advance.
2. **A model can learn too much.** If it memorises quirks of the training data (**overfitting**) it does great in practice and badly on the real exam. The Playground tab lets you cause this yourself.

### Three kinds of task in this project
| Task | You predict... | Example here | Typical score |
|---|---|---|---|
| **Classification** | a category | Did this Titanic passenger survive? | accuracy, F1 |
| **Regression** | a number | What is this neighbourhood's median house price? | error (RMSE / MAE) |
| **Clustering** | nothing - find groups | Which mall customers behave alike? | silhouette |

### Why "robustness" matters
A model that scores 85% in a clean lab can drop sharply when real data has typos, gaps or a rare outcome. **ML Arena's whole point is to measure that drop.**
""")
    st.markdown("#### What do the scores mean?")
    pick = st.selectbox("Pick a score to see what it means", list(METRIC_INFO), key="metric_help")
    direction, text = METRIC_INFO[pick]
    st.info(f"**{pick}** ({direction} is better): {text}")

with t2:
    st.markdown("### Play with a real model (live, nothing pre-computed)")
    st.write("Two curved clouds of dots (blue = class 0, orange = class 1) must be told apart. The model is trained on the *circles* and graded on the *diamonds* it never saw. "
             "**Try:** push the complexity slider to the maximum and watch the train score reach ~100% while the test score stalls - that is overfitting.")
    left, right = st.columns([1, 2])
    with left:
        model_name = st.radio("Model", ["Decision Tree", "kNN", "Random Forest"], help="Different algorithms draw different boundaries.")
        if model_name == "kNN":
            cx = st.slider("k = how many neighbours vote (small k = twitchy, large k = smooth)", 1, 50, 5)
        else:
            cx = st.slider("Max depth = how many questions the tree may ask (more = more complex)", 1, 20, 3)
        data_noise = st.slider("Data messiness (how much the two clouds overlap)", 0.05, 0.6, 0.4, 0.05)
        test_noise = st.slider("Extra noise added to the TEST inputs only (like the Robustness Lab)", 0.0, 1.0, 0.0, 0.1,
                               help="Simulates faulty sensors at deployment time. The model is NOT retrained.")
        seed = st.number_input("Random seed", 0, 999, 0, help="Same seed = same dots every time. Change it to see another draw.")
    X, y = make_moons(n_samples=300, noise=data_noise, random_state=int(seed))
    Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]
    rng = np.random.default_rng(int(seed))
    Xte_noisy = Xte + rng.normal(0, test_noise, Xte.shape) if test_noise else Xte
    if model_name == "kNN":
        mdl = KNeighborsClassifier(n_neighbors=cx)
    elif model_name == "Decision Tree":
        mdl = DecisionTreeClassifier(max_depth=cx, random_state=int(seed))
    else:
        mdl = RandomForestClassifier(n_estimators=50, max_depth=cx, random_state=int(seed))
    mdl.fit(Xtr, ytr)
    tr_acc, te_acc, te_noisy_acc = mdl.score(Xtr, ytr), mdl.score(Xte, yte), mdl.score(Xte_noisy, yte)
    xs, ys = np.linspace(-2, 3, 120), np.linspace(-1.5, 2, 120)
    zz = mdl.predict(np.c_[np.repeat(xs[None, :], len(ys), 0).ravel(), np.repeat(ys[:, None], len(xs), 1).ravel()]).reshape(len(ys), len(xs))
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=zz, showscale=False, opacity=0.25, colorscale=[[0, "#4361ee"], [1, "#f77f00"]], hoverinfo="skip"))
    pal = {0: "#4361ee", 1: "#f77f00"}
    for cls in (0, 1):
        fig.add_trace(go.Scatter(x=Xtr[ytr == cls, 0], y=Xtr[ytr == cls, 1], mode="markers", name=f"train, class {cls}",
                                 marker=dict(color=pal[cls], size=8, line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=Xte_noisy[yte == cls, 0], y=Xte_noisy[yte == cls, 1], mode="markers", name=f"test, class {cls}",
                                 marker=dict(color=pal[cls], size=9, symbol="diamond", line=dict(width=1, color="black"))))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.1))
    with right:
        st.plotly_chart(fig, width="stretch")
    a, b, c = st.columns(3)
    a.metric("Score on training dots (circles)", f"{tr_acc:.0%}")
    b.metric("Score on clean test dots", f"{te_acc:.0%}", f"{te_acc - tr_acc:+.0%} vs training", delta_color="off")
    c.metric("Score on noisy test dots (diamonds shown)", f"{te_noisy_acc:.0%}", f"{te_noisy_acc - te_acc:+.0%} vs clean test", delta_color="off")
    gap = tr_acc - te_acc
    if gap > 0.10:
        st.warning(f"⚠️ **Overfitting:** training score is {gap:.0%} higher than test score. The model memorised the circles instead of learning the shape. Lower the complexity.")
    elif tr_acc < 0.80:
        st.warning("📉 **Underfitting:** the model is too simple to capture the shape, so even the training score is low. Raise the complexity.")
    else:
        st.success("✅ **Good balance:** train and test scores are close and high.")
    if test_noise > 0 and te_noisy_acc < te_acc - 0.03:
        st.info("🌧️ Noise at test time hurts even though the model itself did not change. Measuring this for real models is what the *Robustness Lab* page does.")

with t3:
    st.markdown("### The contestants")
    lb = load("summary/leaderboard_clf.csv")
    lbr = load("summary/leaderboard_reg.csv")
    names = list(MODEL_INFO)
    sel = st.selectbox("Choose a model", names, format_func=pretty)
    st.markdown(f"#### {pretty(sel)}")
    st.write(MODEL_INFO[sel])
    cols = st.columns(2)
    for col, (title, df, metric) in zip(cols, (("Titanic (macro-F1, higher is better)", lb, "f1_macro"), ("Housing (RMSE, lower is better)", lbr, "rmse"))):
        row = df[(df.model == sel) & (df.metric == metric)]
        if len(row):
            r = row.iloc[0]
            col.metric(title, f"{r['mean']:.3f} ± {r['std']:.3f}", f"rank {int(r['rank'])} of {df[df.metric == metric].model.nunique()}", delta_color="off")
        else:
            col.metric(title, "-", "not used for this task", delta_color="off")
    with st.expander("See all models side by side"):
        for m in names:
            st.markdown(f"**{pretty(m)}** - {MODEL_INFO[m]}")

with t4:
    q = st.text_input("🔎 Search the glossary", placeholder="e.g. overfitting")
    for k, v in GLOSSARY.items():
        if q.lower() in (k + v).lower():
            st.markdown(f"**{k}**  \n{v}")
