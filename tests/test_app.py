"""Smoke test: every dashboard page runs without exceptions on the real results."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py"] + sorted((APP / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def _run(page):
    return AppTest.from_file(str(APP / "pages" / page), default_timeout=60).run()


@pytest.mark.parametrize("page", ["1_Leaderboards.py", "2_Robustness_Lab.py"])
def test_task_switch_and_selectors(page):
    for label in ("Classification (Titanic)", "Regression (California Housing)"):
        at = _run(page)
        at.sidebar.radio(key="task").set_value(label).run()
        assert not at.exception, [e.value for e in at.exception]
        for i in range(len(at.sidebar.selectbox)):     # walk every option of every sidebar selectbox
            for opt in list(at.sidebar.selectbox[i].options):
                at = at.sidebar.selectbox[i].set_value(opt).run()    # re-fetch widgets after each rerun
                assert not at.exception, (label, i, opt, [e.value for e in at.exception])


def test_clustering_feature_sets():
    at = _run("4_Clustering.py")
    at.sidebar.radio[0].set_value("3d").run()
    assert not at.exception, [e.value for e in at.exception]


def test_learn_playground_overfitting_demo():
    at = _run("0_Learn_the_Basics.py")
    at.radio[0].set_value("Decision Tree").run()
    at.slider[0].set_value(20).run()          # max depth 20 -> memorises training dots
    assert not at.exception, [e.value for e in at.exception]
    assert any("Overfitting" in w.value for w in at.warning)
