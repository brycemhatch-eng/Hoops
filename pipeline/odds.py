"""Sportsbook odds from The Odds API -> raw/book_odds.json (one row per game x book), matched to ESPN game ids.

Key: ODDS_API_KEY env var, or ~/.config/hatchs-hoops/odds.env (never committed or published).
Without a key, or if the API can't be reached, the last saved file is kept.
"""
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = RAW / "book_odds.json"
BOOKS = "draftkings,fanduel,betmgm,williamhill_us,espnbet,betrivers,fanatics,hardrockbet"
LABEL = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "espnbet": "ESPN BET", "betrivers": "BetRivers", "fanatics": "Fanatics", "hardrockbet": "Hard Rock"}


def key():
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    f = Path.home() / ".config" / "hatchs-hoops" / "odds.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


ALIAS = {"(fl)": "", "(chi)": "chicago", "(oh)": "ohio", "(mn)": "", "(pa)": "", "(md)": "maryland",
         "int'l": "international", "intl": "international", "univ": "", "university": "", "u.": ""}


def norm(s):
    s = str(s).lower()
    for a, b in ALIAS.items():
        s = s.replace(a, b)
    s = s.replace("&", "and").replace("state", "st").replace("saint", "st")
    return re.sub(r"[^a-z0-9]", "", s)


def variants(r, side):
    vals = [getattr(r, f"{side}_{c}", "") for c in ("display_name", "location", "short_display_name", "name")]
    loc, nick = vals[1], vals[3]
    return {norm(v) for v in (vals[0], vals[1], vals[2], f"{loc} {nick}", f"{vals[2]} {nick}") if v}


def sim(name, cands):
    n = norm(name)
    best = 0.0
    for c in cands:
        if n == c:
            return 1.0
        best = max(best, SequenceMatcher(None, n, c).ratio())
    return best


def match(ev, day_games):
    """Pick the scheduled game whose two teams best match the event's team names (either orientation)."""
    best, arg = 0.0, None
    for r in day_games:
        hv, av = variants(r, "home"), variants(r, "away")
        straight = sim(ev["home_team"], hv) + sim(ev["away_team"], av)
        flipped = sim(ev["home_team"], av) + sim(ev["away_team"], hv)
        sc, fl = (straight, False) if straight >= flipped else (flipped, True)
        if sc > best:
            best, arg = sc, (r, fl)
    return arg if best >= 1.55 else None


def normalize(events):
    sched = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("sched_*.parquet"))[-2:]], ignore_index=True)
    sched = sched[~sched.status_type_completed.fillna(False).astype(bool)
                  & (sched.home_display_name != "TBD") & (sched.away_display_name != "TBD")].copy()
    sched["d"] = pd.to_datetime(sched.game_date).dt.date
    by_day = {d: list(g.itertuples()) for d, g in sched.groupby("d")}
    rows, miss = [], set()
    for ev in events:
        ko = pd.Timestamp(ev["commence_time"]).tz_convert("America/New_York").date()
        cands = by_day.get(ko, []) + by_day.get(ko - pd.Timedelta(days=1), []) + by_day.get(ko + pd.Timedelta(days=1), [])
        m = match(ev, cands)
        if m is None:
            miss.add(f"{ev['away_team']} @ {ev['home_team']}")
            continue
        g, flip = m
        for bk in ev.get("bookmakers", []):
            r = {"game_id": int(g.game_id), "kickoff": ev["commence_time"], "book": LABEL.get(bk["key"], bk["title"])}
            H, A = (ev["away_team"], ev["home_team"]) if flip else (ev["home_team"], ev["away_team"])
            for mk in bk["markets"]:
                o = {x["name"]: x for x in mk["outcomes"]}
                if mk["key"] == "spreads" and H in o and A in o:
                    r.update(home_spread=o[H]["point"], home_sp_price=o[H]["price"], away_sp_price=o[A]["price"])
                elif mk["key"] == "totals" and "Over" in o and "Under" in o:
                    r.update(total=o["Over"]["point"], over_price=o["Over"]["price"], under_price=o["Under"]["price"])
                elif mk["key"] == "h2h" and H in o and A in o:
                    r.update(home_ml=o[H]["price"], away_ml=o[A]["price"])
            rows.append(r)
    if miss:
        print(f"  odds: {len(miss)} games not matched to the schedule (e.g. {sorted(miss)[:3]})")
    return rows


def main():
    k = key()
    if not k:
        print("odds: no API key; keeping last saved book odds")
        return
    url = ("https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
           f"?apiKey={k}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers={BOOKS}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            events = json.load(resp)
            left = resp.headers.get("x-requests-remaining")
    except Exception as e:
        print(f"odds: API unreachable ({e.__class__.__name__}); keeping last saved book odds")
        return
    rows = normalize(events)
    OUT.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                               "source": "The Odds API", "rows": rows}, indent=1))
    print(f"odds: {len(rows)} game-book rows saved; {left} API requests left this month")


if __name__ == "__main__":
    main()
