"""Shared helpers for the dashboard. The app only READS CSVs produced by the scripts in scripts/."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TASKS = {"Classification (Titanic)": "clf", "Regression (California Housing)": "reg"}
MODEL_NAMES = {"dt": "Decision Tree", "id3": "ID3", "dt_disc": "Tree (sklearn, discretized)", "knn": "kNN", "nc": "Nearest Centroid",
               "ann": "ANN (MLP)", "rf": "Random Forest", "ada": "AdaBoost", "xgb": "XGBoost", "lin": "Linear Regression",
               "gb": "Gradient Boosting", "dummy": "Dummy baseline"}


def pretty(m):
    return MODEL_NAMES.get(m, m)


def load(rel):
    """Read a results CSV; show a friendly message (not a stack trace) if the experiments were not run yet."""
    path = RES / rel
    if not path.exists():
        st.error(f"Missing results file `results/{rel}`. Run `python scripts/run_all.py` first (see README).")
        st.stop()
    return _read(str(path))


@st.cache_data(show_spinner=False)
def _read(path):
    return pd.read_csv(path)


def header(title, blurb):
    st.set_page_config(page_title=f"ML Arena - {title}", layout="wide")
    st.title(title)
    st.caption(blurb)


def task_picker(key="task"):
    label = st.sidebar.radio("Task", list(TASKS), key=key)
    return TASKS[label], label
