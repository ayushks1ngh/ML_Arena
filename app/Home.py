"""ML Arena dashboard - home: headline findings and hypothesis scoreboard (all computed from results/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import streamlit as st
from common import RES, header, load, pretty

header("ML Arena", "Machine learning benchmarking & reliability lab: which algorithm is best, and does it stay best when data gets harder?")

lb_c, lb_r = load("summary/leaderboard_clf.csv"), load("summary/leaderboard_reg.csv")
stab_c = load("summary/robustness_rank_stability_clf.csv")
stab_r = load("summary/robustness_rank_stability_reg.csv")
hyp = load("summary/hypotheses.csv")

best_c = lb_c[(lb_c.metric == "f1_macro") & lb_c.is_best].iloc[0]
best_r = lb_r[(lb_r.metric == "rmse") & lb_r.is_best].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Best classifier (macro-F1)", pretty(best_c.model), f"{best_c['mean']:.3f} ± {best_c['std']:.3f}")
c2.metric("Best regressor (RMSE, x$100k)", pretty(best_r.model), f"{best_r['mean']:.3f} ± {best_r['std']:.3f}", delta_color="off")
chg_c, chg_r = int(stab_c.top1_changed.sum()), int(stab_r.top1_changed.sum())
c3.metric("Top model displaced under perturbation", f"{chg_c}/{len(stab_c)} clf, {chg_r}/{len(stab_r)} reg", "(family, level) cells", delta_color="off")

st.markdown("### Key finding")
st.info(f"On clean data **{pretty(best_c.model)}** (classification) and **{pretty(best_r.model)}** (regression) lead the leaderboards. "
        f"Under perturbation a different model becomes #1 in {chg_c} of {len(stab_c)} classification and {chg_r} of {len(stab_r)} "
        f"regression test cells, so the clean-data winner is **not** automatically the most reliable model. "
        "Open the *Robustness Lab* page for the exact numbers.")

st.markdown("### Hypothesis scoreboard (verdicts computed from the data)")
color = {"Supported": "#2a9d8f", "Not supported": "#e76f51", "Inconclusive": "#e9c46a"}
st.dataframe(hyp[["id", "claim", "verdict", "evidence"]].style.map(lambda v: f"background-color: {color.get(v, '')}; color: black" if v in color else "", subset=["verdict"]),
             hide_index=True, width="stretch")

st.markdown("### Pages\n- **Leaderboards**: per-task rankings, metrics, ties\n- **Robustness Lab**: noise, missing values, drop-feature, imbalance\n- **Failure Investigator**: disagreements, confusion matrices, importance, trade-offs\n- **Clustering**: K-means, Mean Shift, single linkage, DBSCAN and their stability\n- **Methodology**: protocol, datasets, limitations")
