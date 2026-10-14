"""Regenerate EVERYTHING in one command.

    python scripts/run_all.py            full run (5 seeds, tuned grids; ~20-30 min)
    python scripts/run_all.py --quick    smoke test (2 seeds, no tuning); writes to results_quick/, never touches results/
Each step runs as its own process with a timeout, so a hang cannot block the pipeline forever.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY, S = sys.executable, ROOT / "scripts"
# (name, script, extra args, timeout seconds for the full run)
STEPS = [("classification", "run_classification.py", [], 1500),
         ("regression", "run_regression.py", [], 2400),
         ("leaderboards", "make_leaderboards.py", [], 300),
         ("robustness-clf", "run_robustness.py", ["--task", "clf"], 1500),
         ("robustness-reg", "run_robustness.py", ["--task", "reg"], 2400),
         ("clustering", "run_clustering.py", [], 600),
         ("failures+hypotheses", "run_failures.py", [], 900)]

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", nargs="*", help="run only these step names")
a = ap.parse_args()
env = dict(os.environ)
if a.quick:
    env["ARENA_RESULTS_DIR"] = str(ROOT / "results_quick")
t_all = time.time()
for name, script, args, timeout in STEPS:
    if a.only and name not in a.only:
        continue
    cmd = [PY, str(S / script)] + args + (["--quick"] if a.quick and script not in ("make_leaderboards.py", "run_failures.py") else [])
    t0 = time.time()
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, cwd=ROOT, timeout=(timeout // 5 if a.quick else timeout))
    except subprocess.TimeoutExpired:
        sys.exit(f"STEP '{name}' TIMED OUT")
    except subprocess.CalledProcessError as e:
        sys.exit(f"STEP '{name}' FAILED (exit {e.returncode})")
    print(f"--- {name} done in {time.time() - t0:.0f}s", flush=True)
print(f"\nALL DONE in {time.time() - t_all:.0f}s -> {env.get('ARENA_RESULTS_DIR', ROOT / 'results')}")
