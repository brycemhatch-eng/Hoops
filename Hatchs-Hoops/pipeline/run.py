"""One command to refresh Hatch's Hoops:
schedules + box scores (sportsdataverse) -> lines for newly finished games -> today's lines (ESPN)
-> sportsbook odds (The Odds API, optional) -> features -> walk-forward model -> export -> locked picks ledger.

    python pipeline/run.py          # daily refresh
    python pipeline/run.py --all    # re-download every season's schedules and box scores
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(script, *args):
    print(f"\n== {script}")
    subprocess.run([sys.executable, str(ROOT / "pipeline" / script), *args], check=True, cwd=ROOT)


if __name__ == "__main__":
    for d in ("raw", "data", "output", "docs/data"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    step("download.py", *(["--all"] if "--all" in sys.argv else []))
    for s in ("lines.py", "espn.py", "odds.py", "features.py", "model.py", "export.py", "picks.py"):
        step(s)
    print("\nDone -> docs/data")
