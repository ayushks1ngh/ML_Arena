import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from common import ROOT, header

header("Methodology", "How the experiments were run, and what the limits are.")
st.markdown("""
**Protocol.** 5 seeds; per seed an 80/20 split (stratified for classification) shared by all models; inner 3-fold CV tuning on the
training set only with small comparable grids; preprocessing (impute, scale, one-hot) fitted on train only; mean ± std reported,
differences below one std are flagged as ties.

**Robustness.** Noise, missing values and top-feature removal perturb the **test set** of a fixed trained model; class imbalance
perturbs the **train set** (model retrained with the same hyperparameters), with an extra variant where the test set is skewed too.
`rel_change` is oriented so negative = worse for every metric.

**Datasets.** Titanic (891 rows), California Housing (20,640 rows, target capped at 5.0), Mall Customers (200 rows, no labels).

**Limitations.** Small Titanic test set (~179 rows): many gaps are within noise. Mall data is small and has no ground truth.
Complexity vs. robustness uses a coarse ordinal over 5-7 models. Only default-style small grids were searched.
""")
for name in ("REPORT.md", "EXPERIMENT_DESIGN.md"):
    f = ROOT / "docs" / name
    if f.exists():
        with st.expander(f"docs/{name}", expanded=(name == "REPORT.md")):
            st.markdown(f.read_text())
