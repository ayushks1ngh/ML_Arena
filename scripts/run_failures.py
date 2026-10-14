"""Failure Investigator tables + trade-off tables + hypothesis verdicts. Usage: python scripts/run_failures.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from arena import failures, hypotheses

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 400)
cases, imp = failures.classification_failures()
print(cases.category.value_counts().to_dict())
worst, imp_r = failures.regression_failures()
for t in ("clf", "reg"):
    print(failures.tradeoff(t).round(3))
h = hypotheses.run()
for r in h.itertuples():
    print(f"\n{r.id} [{r.verdict}] {r.claim}\n  {r.evidence}")
