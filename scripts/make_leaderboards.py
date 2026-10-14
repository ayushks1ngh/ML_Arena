"""Raw baseline CSVs -> leaderboard summaries. Usage: python scripts/make_leaderboards.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena.summarize import write_leaderboard

for t in ("clf", "reg"):
    lb = write_leaderboard(t)
    print(t, "leaderboard:", len(lb), "rows")
