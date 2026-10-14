"""Builds notebooks/01_data_and_preprocessing.ipynb (executed afterwards with nbconvert)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = lambda t: nbf.v4.new_code_cell(t); M = lambda t: nbf.v4.new_markdown_cell(t)
nb.cells = [
M("# 01 - Data & preprocessing\nWhat the three datasets look like, and exactly how they are prepared. Key rule: **split first, fit preprocessing on train only**."),
C("import sys; sys.path.insert(0, '../src')\nimport pandas as pd\nfrom arena.data import load_titanic, load_california, load_mall\nfrom arena.evaluate import split\nfrom arena.preprocess import build_preprocessor"),
M("## Titanic (classification)"),
C("X, y = load_titanic()\nprint(X.shape, 'survival rate = %.3f' % y.mean())\nX.head()"),
C("X.isna().sum()[lambda s: s > 0]"),
C("X.describe().T"),
M("Missing Age (~20%) is imputed with the **train median** inside the pipeline. Cabin is dropped (77% missing). `Title`, `FamilySize`, `IsAlone` are engineered from Name/SibSp/Parch."),
M("## California Housing (regression)"),
C("Xc, yc = load_california()\nprint(Xc.shape, 'target range', yc.min(), yc.max())\nprint('share of houses at the 5.0 cap: %.3f' % (yc >= 5.0).mean())\nXc.describe().T"),
M("The target is capped at 5.0 ($500k): a ceiling effect that matters for error analysis later."),
M("## Mall Customers (clustering)"),
C("m = load_mall(); print(m.shape); m.describe().T"),
M("200 rows, no labels. Clustering uses standardized Annual Income and Spending Score (Gender excluded: a binary one-hot distorts Euclidean distance)."),
M("## Leakage check: preprocessing is fitted on the training split only"),
C("Xtr, Xte, ytr, yte = split('clf', 0)\npre, n_num = build_preprocessor(Xtr); pre.fit(Xtr)\nprint('train/test sizes', len(Xtr), len(Xte), '| numeric cols first:', n_num)\nprint('scaler means come from TRAIN:', pre.named_transformers_['num'].named_steps['sc'].mean_.round(3))\nprint('test class share %.3f vs train %.3f (stratified)' % (yte.mean(), ytr.mean()))"),
]
nbf.write(nb, "notebooks/01_data_and_preprocessing.ipynb")
