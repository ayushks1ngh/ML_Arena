import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from common import ROOT, glossary_expander, header

header("Methodology 📋", "How we kept the contest fair, what the limits are, and where to find things in the repo.")
st.markdown("## How we kept it fair")
a, b = st.columns(2)
with a:
    st.markdown("""
**✅ Same exam for everyone**
Each model gets the identical train/test split in every repeat.

**✅ No peeking (no data leakage)**
Averages, scalers and encoders are learned from *training* data only. Settings are tuned with cross-validation *inside* the training data.

**✅ Repeated 5 times**
Five different random seeds. We report the average and the wobble, and call it a **tie** when the gap is smaller than the wobble.
""")
with b:
    st.markdown("""
**✅ Damage the test set, not the training set**
Noise, missing values and lost columns are applied to the *test* data of an already-trained model (like a deployed model meeting messy real data). The exception is *class imbalance*, which changes the training data and retrains.

**✅ Everything reproducible**
Every random step has a seed. `python scripts/run_all.py` regenerates every number on this dashboard.

**✅ No invented numbers**
The dashboard only *reads* result files; it computes nothing by hand.
""")

st.markdown("## Datasets")
st.table({"Dataset": ["Titanic", "California Housing", "Mall Customers"], "Task": ["Classification", "Regression", "Clustering"],
          "Size": ["891 passengers", "20,640 neighbourhoods", "200 customers"],
          "Target": ["survived (0/1)", "median house value (capped at $500k)", "none"]})

st.markdown("## Honest limitations")
st.warning("""
- Titanic's test set is only ~179 passengers, so many model gaps are within random wobble (flagged as ties).
- Mall Customers is small and has no ground truth, so cluster quality is judged indirectly.
- Only small tuning grids were searched; a bigger search might change rankings.
- 'Complexity vs. robustness' compares only 5-7 models per task, so that result is weak evidence.
""")

st.markdown("## 🗺️ Map of the repository")
st.code("""app/        this dashboard (Home + pages). Only reads CSVs from results/
src/arena/  the actual science:
              data.py, preprocess.py   load + clean data (leak-free)
              models.py, id3.py        the algorithms (ID3 written from scratch)
              evaluate.py              scoring
              perturb.py, robustness.py  the 'make data harder' experiments
              failures.py, hypotheses.py error analysis + hypothesis verdicts
              clustering.py            K-means, Mean Shift, single linkage, DBSCAN
scripts/    run_all.py regenerates every result; one runner per experiment
results/    raw/ per-seed CSVs, summary/ tables the dashboard reads, figures/
notebooks/  step-by-step notebooks 01-07, plus syllabus/ labs A-G
docs/       REPORT.md (findings), EXPERIMENT_DESIGN.md (protocol), VIVA_PREP.md (Q&A)
tests/      pytest suite (leakage, perturbations, dashboard pages)""", language="text")
st.markdown("**Suggested reading order for newcomers:** this dashboard → `docs/REPORT.md` → `notebooks/01_*.ipynb` → `src/arena/`.")
glossary_expander()
for name in ("REPORT.md", "EXPERIMENT_DESIGN.md"):
    f = ROOT / "docs" / name
    if f.exists():
        with st.expander(f"📄 Full document: docs/{name}"):
            st.markdown(f.read_text())
