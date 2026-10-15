"""ML Arena dashboard - home: explains the project to a newcomer, then shows the headline findings (all computed from results/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st
from common import glossary_expander, header, load, pretty

header("ML Arena 🏟️", "A small lab that makes machine-learning algorithms compete, then checks whether the winner still wins when the data gets messy.")

lb_c, lb_r = load("summary/leaderboard_clf.csv"), load("summary/leaderboard_reg.csv")
stab_c = load("summary/robustness_rank_stability_clf.csv")
stab_r = load("summary/robustness_rank_stability_reg.csv")
hyp = load("summary/hypotheses.csv")
best_c = lb_c[(lb_c.metric == "f1_macro") & lb_c.is_best].iloc[0]
best_r = lb_r[(lb_r.metric == "rmse") & lb_r.is_best].iloc[0]
chg_c, chg_r = int(stab_c.top1_changed.sum()), int(stab_r.top1_changed.sum())

st.markdown("## 1. What is this project?")
st.markdown(
    "Machine learning means a computer **learns patterns from examples** instead of being given rules. There are dozens of algorithms for "
    "that, and people usually just pick whichever scored best in a test. **ML Arena asks a harder question:**")
st.info("🏁 **Which algorithm is best for each task - and is it still the best when the data gets worse?**\n\n"
        "Think of choosing a car by its time on a dry track. The winner may not be the best car in rain. "
        "Real-world data is the rain: sensors glitch, forms are left blank, one outcome is rare.")

st.markdown("## 2. The three challenges")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🚢 Titanic\n**Classification** - predict a *category*.\n\nGiven a passenger's age, sex, ticket class... did they **survive**?\n\n891 passengers.")
with c2:
    st.markdown("### 🏠 California Housing\n**Regression** - predict a *number*.\n\nGiven a neighbourhood's income, location, house age... what is the **median house price**?\n\n20,640 neighbourhoods.")
with c3:
    st.markdown("### 🛍️ Mall Customers\n**Clustering** - find *groups* with no answer key.\n\nGiven income and spending score, which customers **behave alike**?\n\n200 customers.")

st.markdown("## 3. How the contest works")
s1, s2, s3, s4 = st.columns(4)
s1.markdown("**① Fair start**\n\nEvery model gets the *same* train/test split. Settings are tuned on training data only, so nobody peeks at the exam.")
s2.markdown("**② Leaderboard**\n\nEach model is scored on unseen test data, repeated with 5 random seeds so luck averages out.")
s3.markdown("**③ Make it harder**\n\nThe test data is damaged on purpose: noise, missing values, lost columns, a rare class.")
s4.markdown("**④ Investigate**\n\nWe look at *which* cases models get wrong and *why*.")

st.markdown("## 4. Headline results")
m1, m2, m3 = st.columns(3)
m1.metric("🚢 Best on Titanic (macro-F1, higher is better)", pretty(best_c.model), f"score {best_c['mean']:.3f} ± {best_c['std']:.3f}", delta_color="off",
          help="macro-F1 is a fair accuracy-like score between 0 and 1.")
m2.metric("🏠 Best on housing (RMSE, lower is better)", pretty(best_r.model), f"typical error ≈ ${best_r['mean'] * 100:.0f}k", delta_color="off",
          help=f"RMSE = {best_r['mean']:.3f} in units of $100,000.")
m3.metric("⚠️ Winner dethroned under stress", f"{chg_c}/{len(stab_c)} Titanic · {chg_r}/{len(stab_r)} housing", "test scenarios", delta_color="off",
          help="In how many (damage type, damage level) scenarios a different model ended up #1.")
st.success(
    f"💡 **The answer, in plain English:** on clean data **{pretty(best_c.model)}** and **{pretty(best_r.model)}** win. "
    f"But once the data is damaged, a *different* model takes first place in {chg_c} of {len(stab_c)} Titanic scenarios and "
    f"{chg_r} of {len(stab_r)} housing scenarios. **Being best on clean data does not make a model the most reliable.**")

st.markdown("## 5. Take the guided tour")
st.caption("Click a page to jump there. Each page starts with a 'How to read this page' box.")
tour = [("pages/0_Learn_the_Basics.py", "🎓 Learn the Basics", "Zero ML knowledge needed. Play with a live model, meet every algorithm, read the glossary."),
        ("pages/1_Leaderboards.py", "🏆 Leaderboards", "Who wins on clean data? Pick a task and a score."),
        ("pages/2_Robustness_Lab.py", "🧪 Robustness Lab", "Damage the data with sliders and watch the ranking change."),
        ("pages/3_Failure_Investigator.py", "🔍 Failure Investigator", "Which passengers and houses fool the models?"),
        ("pages/4_Clustering.py", "🧩 Clustering", "Grouping mall customers without an answer key."),
        ("pages/5_Methodology.py", "📋 Methodology", "How we kept the test fair, plus a map of this repository.")]
for path, label, desc in tour:
    a, b = st.columns([1, 3])
    a.page_link(path, label=label)
    b.write(desc)

st.markdown("## 6. Did our predictions come true?")
st.markdown("Before running anything we wrote down six **hypotheses** (educated guesses). They were then *tested* - and several turned out wrong. "
            "That is normal in research and is reported honestly here.")
PLAIN = {
    "H1 (clf)": "The best model on clean Titanic data is not best in every damaged scenario.",
    "H1 (reg)": "The best model on clean housing data is not best in every damaged scenario.",
    "H2": "Tree-team models (Random Forest, AdaBoost, XGBoost) cope with noise better than kNN / neural nets.",
    "H3": "Distance-based models (kNN, Nearest Centroid) suffer the most from noise and missing values.",
    "H4 (train skewed, balanced test)": "With a rare class, plain accuracy hides that the model ignores the rare class (training skewed only).",
    "H4 (train AND test skewed)": "With a rare class, plain accuracy hides that the model ignores the rare class (training and test skewed).",
    "H5 (clf)": "A more complex model is not automatically more robust (Titanic).",
    "H5 (reg)": "A more complex model is not automatically more robust (housing).",
    "H6": "Single linkage and Mean Shift clusters are less stable than K-means when noise is added.",
}
ICON = {"Supported": "✅ Supported", "Not supported": "❌ Not supported", "Inconclusive": "❔ Inconclusive"}
for _, r in hyp.iterrows():
    with st.expander(f"{ICON.get(r.verdict, r.verdict)} - {PLAIN.get(r['id'], r['claim'])}"):
        st.markdown(f"**Original claim:** {r.claim}")
        st.markdown(f"**Verdict:** {r.verdict}  \n*Supported = the data agreed with the guess. Not supported = the data disagreed.*")
        st.markdown(f"**Evidence (computed from results):** {r.evidence}")

st.markdown("## 7. Visiting the GitHub repo?")
st.markdown(
    "- **Just want the story?** Read this page and *Learn the Basics*; no installation needed.\n"
    "- **Want the full write-up?** `docs/REPORT.md` has the findings; `docs/EXPERIMENT_DESIGN.md` has the exact protocol.\n"
    "- **Want to reproduce everything?** `python scripts/run_all.py` regenerates every number you see here (see README).\n"
    "- **Want to read the code?** `src/arena/` holds the algorithms and experiments; the *Methodology* page has a file-by-file map.")
glossary_expander(label="📚 Glossary - words you will meet")
