"""Central settings. Everything random in the project is driven by SEEDS."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RAW_DIR, SUMMARY_DIR = RESULTS / "raw", RESULTS / "summary"
PRED_DIR, FIG_DIR = RESULTS / "predictions", RESULTS / "figures"

SEEDS = [0, 1, 2, 3, 4]       # full run
QUICK_SEEDS = [0, 1]          # smoke-test run
TEST_SIZE = 0.2
INNER_CV = 3

# Perturbation levels (see docs/EXPERIMENT_DESIGN.md section 6)
NOISE_SIGMAS = [0.1, 0.25, 0.5, 1.0]        # in std units of standardized features
MISSING_FRACS = [0.1, 0.2, 0.3, 0.5]
DROP_TOP_K = [1, 2]
MINORITY_SHARES = [None, 0.20, 0.10, 0.05]  # None = original class balance

for d in (RAW_DIR, SUMMARY_DIR, PRED_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
