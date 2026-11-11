"""Download men's college basketball schedules and team box scores (sportsdataverse / hoopR).

Season labels follow the NCAA convention: 2027 = the 2026-27 season.
History is downloaded once; the current season is re-downloaded every run.

    python pipeline/download.py          # missing seasons + current season
    python pipeline/download.py --all    # everything again
"""
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
FIRST = 2012          # ratings warm up on 2012-2014; lines (and the backtest) start in 2014-15
FILES = {"sched": "schedules/parquet/mbb_schedule_{y}.parquet", "box": "team_box/parquet/team_box_{y}.parquet"}


def current_season(today=None):
    t = today or date.today()
    return t.year + 1 if t.month >= 8 else t.year


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def main(refresh_all=False):
    RAW.mkdir(exist_ok=True)
    cur = current_season()
    for y in range(FIRST, cur + 1):
        for key, tpl in FILES.items():
            dest = RAW / f"{key}_{y}.parquet"
            if refresh_all or y == cur or not dest.exists():
                try:
                    fetch(f"{BASE}/{tpl.format(y=y)}", dest)
                except Exception as e:
                    print(f"  skip {dest.name}: {e}")


if __name__ == "__main__":
    main("--all" in sys.argv)
