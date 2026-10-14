"""Builds notebooks 02 (classification arena) and 03 (regression arena)."""
import nbformat as nbf
C = lambda t: nbf.v4.new_code_cell(t); M = lambda t: nbf.v4.new_markdown_cell(t)
SETUP = "import sys; sys.path.insert(0, '../src')\nimport pandas as pd, matplotlib.pyplot as plt\nfrom arena.config import RAW_DIR, SUMMARY_DIR, PRED_DIR\nfrom arena.viz import leaderboard_bar\npd.set_option('display.width', 160)"

def build(task, title, metrics, main, extra):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        M(f"# {title}\nAll numbers are loaded from CSVs produced by `scripts/run_{'classification' if task=='clf' else 'regression'}.py` "
          f"(5 seeds, same split for every model within a seed, inner 3-fold CV tuning on train only)."),
        C(SETUP),
        C(f"raw = pd.read_csv(RAW_DIR / '{task}_baseline.csv')\nlb = pd.read_csv(SUMMARY_DIR / 'leaderboard_{task}.csv')\nprint(raw.seed.nunique(), 'seeds,', raw.model.nunique(), 'models')"),
        M("## Leaderboard (mean over seeds, with std)"),
        C(f"wide = lb.pivot(index='model', columns='metric', values='mean')[{metrics}].sort_values('{main}', ascending={task=='reg'})\nstd = lb.pivot(index='model', columns='metric', values='std')[{metrics}]\nwide.round(4)"),
        C(f"fig, axes = plt.subplots(1, {len(metrics)}, figsize=({4*len(metrics)}, 3.5))\nfor ax, m in zip(axes, {metrics}):\n    leaderboard_bar(lb, m, ax)\nplt.tight_layout()"),
        M("## Best model per metric (with tie flag: a gap smaller than the leader's std is not a real win)"),
        C("lb[lb.is_best | lb.tie_with_best][['metric','model','mean','std','is_best','tie_with_best']].round(4)"),
        M("## Cost: fit time (tuning included)"),
        C("lb.drop_duplicates('model')[['model','fit_time_s']].sort_values('fit_time_s').round(2)"),
        M("## Tuned hyperparameters chosen (last seed shown)"),
        C(f"raw[(raw.metric=='{main}') & (raw.seed==raw.seed.max())][['model','best_params']]"),
        M(extra),
    ]
    nbf.write(nb, f"notebooks/{'02_classification_arena' if task=='clf' else '03_regression_arena'}.ipynb")

build("clf", "02 - Classification arena (Titanic)", ["accuracy", "f1", "f1_macro"], "f1_macro",
      "**Reading the table:** models whose bars are yellow are statistically indistinguishable from the best given seed-to-seed variation. The Titanic test set has only ~179 rows, so small gaps are noise. ID3 is added to this table on Day 4.")
build("reg", "03 - Regression arena (California Housing)", ["rmse", "mae", "r2"], "rmse",
      "**Reading the table:** RMSE/MAE are errors (lower is better). Targets are capped at 5.0, so every model under-predicts the most expensive houses; this is examined in the Failure Investigator.")
