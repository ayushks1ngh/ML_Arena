"""Verify imports and dataset access. Run: python scripts/check_env.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy, pandas, sklearn, xgboost, scipy, matplotlib, plotly, streamlit  # noqa: F401
from arena.data import load_california, load_mall, load_titanic

print("versions:", {m.__name__: m.__version__ for m in (numpy, pandas, sklearn, xgboost, scipy, streamlit)})
X, y = load_titanic(); print("titanic", X.shape, "pos rate %.3f" % y.mean())
X, y = load_california(); print("california", X.shape, "target max", y.max())
print("mall", load_mall().shape)
print("OK")
