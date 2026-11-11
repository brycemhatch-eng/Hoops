"""Betting lines for every completed game -> raw/lines.csv (cached; only new games are fetched).

ESPN's game summary for each finished game includes a "pickcenter" block with the pregame
spread, total and moneylines (and, for recent seasons, the opening numbers too). sportsdataverse
mirrors those summaries on GitHub, so each game is one small, gzip-compressed request.

Columns (home perspective): home_spread is the home team's handicap (-6.5 = home favored by 6.5).
Games with no posted line are cached with has_line=0 so they aren't fetched again.
"""
import gzip
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = RAW / "lines.csv"
URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{gid}.json"
FIRST_LINES = 2014
COLS = ["game_id", "season", "has_line", "provider", "home_spread", "total", "home_ml", "away_ml",
        "home_sp_odds", "away_sp_odds", "over_odds", "under_odds", "open_home_spread", "open_total",
        "open_home_ml", "open_away_ml"]


def num(x):
    """'+4.5' / 'o134.5' / '-192' / 4.5 / None -> float or None"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lstrip("ou").replace("+", "")
    if s.upper() in ("EVEN", "EV"):
        return 100.0
    try:
        return float(s)
    except ValueError:
        return None


def deep(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def fav_from_details(details, home_abbr, away_abbr):
    """'DUKE -17.0' -> ('home'|'away'|'even', 17.0); None if it can't be resolved"""
    if not isinstance(details, str) or not details.strip():
        return None
    d = details.strip()
    if d.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        return ("even", 0.0)
    parts = d.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    who, n = parts[0].strip().upper(), num(parts[1])
    if n is None:
        return None
    if home_abbr and who == str(home_abbr).upper():
        return ("home", abs(n))
    if away_abbr and who == str(away_abbr).upper():
        return ("away", abs(n))
    return None


def parse_odds(e, home_abbr=None, away_abbr=None):
    """One ESPN odds entry (pickcenter item or scoreboard odds item) -> dict, home perspective.

    The home handicap's sign is taken, in order, from: the explicitly signed closing line; the text
    ('DUKE -17.0') matched to the team abbreviations; the favorite flags. If none of these settle which
    team is favored the spread is dropped rather than guessed."""
    h, a = e.get("homeTeamOdds") or {}, e.get("awayTeamOdds") or {}
    sp = num(deep(e, "pointSpread", "home", "close", "line"))
    if sp is None and e.get("spread") is not None:
        s = abs(float(e["spread"]))
        fav = fav_from_details(e.get("details"), home_abbr, away_abbr)
        if fav:
            sp = 0.0 if fav[0] == "even" else (-fav[1] if fav[0] == "home" else fav[1])
        elif h.get("favorite") and not a.get("favorite"):
            sp = -s
        elif a.get("favorite") and not h.get("favorite"):
            sp = s
        elif s == 0:
            sp = 0.0
    return {
        "provider": deep(e, "provider", "name"),
        "home_spread": sp,
        "total": num(e.get("overUnder")) if e.get("overUnder") is not None else num(deep(e, "total", "over", "close", "line")),
        "home_ml": num(h.get("moneyLine")) if h.get("moneyLine") is not None else num(deep(e, "moneyline", "home", "close", "odds")),
        "away_ml": num(a.get("moneyLine")) if a.get("moneyLine") is not None else num(deep(e, "moneyline", "away", "close", "odds")),
        "home_sp_odds": num(h.get("spreadOdds")) or num(deep(e, "pointSpread", "home", "close", "odds")),
        "away_sp_odds": num(a.get("spreadOdds")) or num(deep(e, "pointSpread", "away", "close", "odds")),
        "over_odds": num(e.get("overOdds")) or num(deep(e, "total", "over", "close", "odds")),
        "under_odds": num(e.get("underOdds")) or num(deep(e, "total", "under", "close", "odds")),
        "open_home_spread": num(deep(e, "pointSpread", "home", "open", "line")),
        "open_total": num(deep(e, "total", "over", "open", "line")),
        "open_home_ml": num(deep(e, "moneyline", "home", "open", "odds")),
        "open_away_ml": num(deep(e, "moneyline", "away", "open", "odds")),
    }


def best_entry(entries):
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    with_line = [e for e in entries if e.get("spread") is not None or deep(e, "pointSpread", "home", "close", "line")]
    return (with_line or entries or [None])[0]


def fetch(gid):
    req = urllib.request.Request(URL.format(gid=gid), headers={"Accept-Encoding": "gzip"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            d = json.loads(body)
            e = best_entry(d.get("pickcenter"))
            row = {"game_id": gid, "has_line": 0}
            if e:
                comp = (deep(d, "header", "competitions") or [{}])[0]
                ab = {c.get("homeAway"): deep(c, "team", "abbreviation") for c in comp.get("competitors", [])}
                row.update(parse_odds(e, ab.get("home"), ab.get("away")))
                row["has_line"] = int(row["home_spread"] is not None or row["total"] is not None)
            return row
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return {"game_id": gid, "has_line": 0}
        except Exception:
            pass
    return None   # network trouble: try again next run


def main(workers=32, refetch=None):
    sched = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "status_type_completed", "game_json"])
                       for f in sorted(RAW.glob("sched_*.parquet"))], ignore_index=True)
    sched = sched[(sched.season >= FIRST_LINES) & sched.status_type_completed.fillna(False).astype(bool)
                  & sched.game_json.fillna(False).astype(bool)]
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    todo = sorted(set(sched.game_id.astype(int)) - set(old.game_id.astype(int)))
    if refetch is not None:
        todo = sorted(set(todo) | set(refetch))
    print(f"lines: {len(old):,} cached, {len(todo):,} to fetch")
    rows, done = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fetch, todo):
            done += 1
            if r:
                rows.append(r)
            if done % 2000 == 0:
                print(f"  {done:,}/{len(todo):,}")
    new = pd.DataFrame(rows, columns=[c for c in COLS if c != "season"])
    new["season"] = new.game_id.map(dict(zip(sched.game_id.astype(int), sched.season)))
    out = pd.concat([old, new[COLS]], ignore_index=True).drop_duplicates("game_id", keep="last").sort_values("game_id")
    out.to_csv(OUT, index=False)
    got = out[out.has_line == 1].groupby("season").size().to_dict()
    print(f"lines: +{len(new):,} games; with lines by season {got}")


if __name__ == "__main__":
    import sys
    if "--fix-signs" in sys.argv:     # one-off: re-pull lines saved before favorites were resolved from the text
        cache = pd.read_csv(OUT)
        main(refetch=cache[(cache.has_line == 1) & cache.home_ml.isna()].game_id.astype(int).tolist())
    else:
        main()
