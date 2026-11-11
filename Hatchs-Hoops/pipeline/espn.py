"""Lines for upcoming games from ESPN's public scoreboard -> raw/upcoming_lines.csv

Covers today and the next few days. Uses the same odds parser as pipeline/lines.py, keyed by ESPN
game id (the same id the schedule uses). If ESPN can't be reached the last saved file is kept.
"""
import json
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lines import COLS, best_entry, parse_odds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "raw" / "upcoming_lines.csv"
URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
       "?dates={d}&groups=50&limit=500")
DAYS_AHEAD = 3


def parse_scoreboard(sb):
    rows = []
    for ev in sb.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        e = best_entry(comp.get("odds"))
        if not e:
            continue
        ab = {c.get("homeAway"): (c.get("team") or {}).get("abbreviation") for c in comp.get("competitors", [])}
        r = parse_odds(e, ab.get("home"), ab.get("away"))
        if r["home_spread"] is None and r["total"] is None:
            continue
        r.update(game_id=int(ev["id"]), has_line=1)
        rows.append(r)
    return rows


def main():
    rows, ok = [], 0
    for i in range(DAYS_AHEAD + 1):
        d = (date.today() + timedelta(days=i)).strftime("%Y%m%d")
        try:
            with urllib.request.urlopen(URL.format(d=d), timeout=30) as resp:
                rows += parse_scoreboard(json.load(resp))
                ok += 1
        except Exception as e:
            print(f"  espn {d}: {e.__class__.__name__}")
    if not ok:
        print("espn: unreachable; keeping last saved upcoming lines")
        return
    df = pd.DataFrame(rows, columns=[c for c in COLS if c != "season"])
    df.to_csv(OUT, index=False)
    print(f"espn: lines for {len(df)} upcoming games")


if __name__ == "__main__":
    main()
