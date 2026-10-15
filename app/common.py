"""Shared helpers for the dashboard. The app only READS CSVs produced by the scripts in scripts/.

This file also holds all the plain-language text (model descriptions, metric meanings, glossary) so that
every page explains things in the same words. Numbers are never written here: they always come from results/.
"""
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

# One-sentence, jargon-free description of every model (shown on the Leaderboards and Learn pages).
MODEL_INFO = {
    "dt": "Plays 20 Questions with the data: a chain of yes/no questions ('Is the passenger female?') ending in an answer.",
    "id3": "A classic textbook decision tree, written from scratch in this project. It only understands categories, so numbers are put into bins first.",
    "dt_disc": "A normal scikit-learn tree fed the same binned data as ID3, so the two can be compared fairly.",
    "knn": "Finds the k most similar examples it has seen and copies the majority answer (or average) - 'ask your neighbours'.",
    "nc": "Squashes each class into one average 'centre point' and picks whichever centre is closest. Very simple.",
    "ann": "A small neural network: layers of simple units whose connection strengths are learned.",
    "rf": "Many different decision trees vote. Their individual mistakes tend to cancel out ('wisdom of the crowd').",
    "ada": "Builds small trees one after another; each new tree focuses on the examples the previous ones got wrong.",
    "xgb": "A highly-engineered 'boosting' method: many small trees, each correcting the leftover errors of the team so far.",
    "lin": "Draws the best straight line (or flat plane) through the data and reads predictions off it.",
    "gb": "Boosting like AdaBoost, but each new tree is trained to fix the remaining prediction error.",
    "dummy": "Not really a model: always guesses the most common answer (or the average). It is the floor every real model must beat.",
}

# What each score means in everyday words.
METRIC_INFO = {
    "accuracy": ("higher", "Share of predictions that were right. Easy to read, but misleading when one answer is much more common than the other."),
    "f1_macro": ("higher", "A fairer score: it measures how well the model does on *each* class separately and averages them, so ignoring the rarer class is punished."),
    "f1": ("higher", "Balance between precision and recall for the 'positive' class (survived)."),
    "precision": ("higher", "When the model says 'survived', how often is it right?"),
    "recall": ("higher", "Of the people who really survived, how many did the model find?"),
    "roc_auc": ("higher", "How well the model *ranks* people: 1.0 = perfect ordering, 0.5 = coin flip."),
    "rmse": ("lower", "Typical size of the price error, in units of $100,000 (0.45 = about $45k off). Big misses are punished extra."),
    "mae": ("lower", "Average size of the price error, in units of $100,000. Every miss counts equally."),
    "r2": ("higher", "Share of the variation in house prices the model explains. 1.0 = perfect, 0 = no better than always guessing the average."),
}

# Friendly names + one-line real-world analogy for each "harder data" scenario.
FAMILY_NAMES = {"noise": "Noisy measurements", "missing": "Missing values", "drop_feature": "Lost columns",
                "imbalance": "Rare class (training only)", "imbalance_matched": "Rare class (training + test)"}
FAMILY_STORY = {
    "noise": "**Real-world version:** a faulty sensor or a typo-prone form. Random jitter is added to the numeric inputs of the *test* data. "
             "Level = how big the jitter is (in units of the column's normal spread).",
    "missing": "**Real-world version:** people skip questions on a form. A share of numeric answers is blanked out and filled with the usual (median) value. "
               "Level = fraction of values lost.",
    "drop_feature": "**Real-world version:** a data source gets switched off. The k *most important* columns are wiped. Level = how many columns are lost.",
    "imbalance": "**Real-world version:** the model is trained on data where one outcome is rare, then judged on a balanced test. "
                 "Level = share of the rare class in the training data (0.05 = only 5%).",
    "imbalance_matched": "**Real-world version:** the world really is lopsided, so training *and* test data have the same rare class. "
                         "Level = share of the rare class (0.05 = only 5%).",
}

CLUSTER_NAMES = {"kmeans": "K-means", "meanshift": "Mean Shift", "agg_single": "Single linkage", "dbscan": "DBSCAN"}
CLUSTER_INFO = {
    "kmeans": "Picks k centre points and assigns every customer to the nearest one, then moves the centres until stable. You choose k.",
    "meanshift": "Slides a window towards the densest spots; each dense spot becomes a group. It decides the number of groups itself.",
    "agg_single": "Starts with every customer alone and keeps merging the two closest groups. 'Single linkage' can chain far-apart points together.",
    "dbscan": "Grows groups from crowded areas and labels isolated customers as 'noise' (cluster -1).",
}

GLOSSARY = {
    "Training set / test set": "The data is split. The model *learns* from the training part and is *graded* on the test part it has never seen - like studying from practice exams, then sitting a new one.",
    "Overfitting": "Memorising the practice exam instead of learning the subject: great training score, poor test score.",
    "Data leakage": "Accidentally letting test information into training (e.g. computing averages on all the data). It inflates scores, so this project prevents it on purpose.",
    "Seed": "A number that fixes the 'random' choices so anyone can reproduce the exact same results.",
    "Mean ± std": "Each experiment is repeated with 5 seeds. 'Mean' is the average score, 'std' shows how much it wobbles between repeats.",
    "Tie": "If two models differ by less than the wobble (1 std), we cannot honestly say one is better, so we call it a tie.",
    "Hyperparameter tuning": "Trying several settings (like tree depth) and keeping the best one - using only the training data.",
    "Robustness": "How little a model's score drops when the data gets worse than it was in training.",
    "Perturbation": "A deliberate worsening of the data (noise, missing values, ...) used to test robustness.",
    "Class imbalance": "When one outcome is much rarer than the other (e.g. few survivors). Models can look accurate by always predicting the common one.",
    "Rank stability": "Whether the model ordering (1st, 2nd, 3rd...) stays the same when conditions change.",
    "Clustering": "Grouping similar items without being told the right answer (no labels).",
    "Silhouette": "Clustering score from -1 to 1: how snugly points sit inside their own cluster versus the next one. Higher is tidier.",
    "ARI (Adjusted Rand Index)": "How similar two groupings are. 1 = identical, around 0 = no better than random. Used to see if clusters survive noise.",
}


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
    st.set_page_config(page_title=f"ML Arena - {title}", page_icon="🏟️", layout="wide")
    st.title(title)
    st.markdown(f"*{blurb}*")
    with st.sidebar:
        st.markdown("**New here?** Start on the *Home* page, then *Learn the Basics*.")


def how_to_read(*steps):
    """A collapsed-by-default-off box that tells a beginner how to read the page in 2-4 steps."""
    with st.expander("📖 How to read this page", expanded=True):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}.** {s}")


def takeaway(text):
    """The one-sentence 'so what' for a chart, computed from the data by the caller."""
    st.success(f"💡 **In plain English:** {text}")


def glossary_expander(keys=None, label="📚 Words used on this page"):
    with st.expander(label):
        for k in (keys or GLOSSARY):
            st.markdown(f"- **{k}**: {GLOSSARY[k]}")


def task_picker(key="task"):
    label = st.sidebar.radio("Task", list(TASKS), key=key,
                             help="Classification = predict a category (survived or not). Regression = predict a number (a house price).")
    return TASKS[label], label
