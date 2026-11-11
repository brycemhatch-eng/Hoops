"""Locked picks ledger with closing line value -> docs/data/picks.json

Game cards are rebuilt every run, so a pick can flip during the day and the card is graded at the closing
line. This ledger records each play the first time the model flags it, at the best price available at that
moment, and never changes it:

  lock   - an upcoming game becomes a play (edge over the threshold and positive EV at a real book price):
           store line, price, book, edge, win probability and quarter-Kelly stake, with a timestamp
  close  - after tip-off: store the closing line and closing line value (points / % win prob the locked number beat it by)
  grade  - after the final: win / loss / push at the LOCKED line and price, flat and Kelly-sized

docs/data/picks.json is committed by the daily workflow, so the history persists.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "output" / "ml.json"
LEDGER = ROOT / "docs" / "data" / "picks.json"
ET = ZoneInfo("America/New_York")
CONSENSUS_WINDOW_H = 24     # without book prices, lock ESPN's line only within a day of tip-off


def payout(o):
    return o / 100 if o > 0 else 100 / -o


def implied(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def tip(g):
    return datetime.fromisoformat(g["tip"]).replace(tzinfo=ET)


def best_offer(shop, gid, market, side, thr, cap=None):
    if thr is None:
        return None
    rows = [x for x in shop if x["id"] == gid and x["market"] == market and x["side"] == side
            and x.get("edge") is not None and x["edge"] >= thr and (cap is None or x["edge"] <= cap) and (x.get("ev") or 0) > 0]
    if not rows:
        return None
    if market == "Spread":
        key = lambda x: (x["line"], payout(x["price"]))
    elif market == "Total":
        key = lambda x: ((-x["line"] if side == "over" else x["line"]), payout(x["price"]))
    else:
        key = lambda x: payout(x["price"])
    return max(rows, key=key)


def lock_new(ledger, ml, now):
    have = {p["id"] for p in ledger}
    thr = ml["thresholds"]
    shop = (ml.get("shop") or {}).get("rows", [])
    shop_games = {x["id"] for x in shop}
    names = ml["teams"]
    n = 0
    for g in ml["games"]:
        if g["final"] or tip(g) <= now:
            continue
        for b in g["bets"]:
            pid = f"{g['id']}:{b['market']}"
            if pid in have or not b.get("play"):
                continue
            o = best_offer(shop, g["id"], b["market"], b["side"], thr[b["market"]], (ml.get("max_edge") or {}).get(b["market"]))
            if o is None:
                if g["id"] in shop_games:       # books post it, but none clears the bar at its own number
                    continue
                if (tip(g) - now).total_seconds() > CONSENSUS_WINDOW_H * 3600:
                    continue
            src = o or b
            price = src.get("price", src.get("odds"))
            if price is None:
                continue
            ledger.append({
                "id": pid, "game_id": g["id"], "season": ml["season"], "date": g["date"], "tip": g["tip"],
                "home": g["home"], "away": g["away"], "matchup": f"{names[str(g['away'])]['name']} @ {names[str(g['home'])]['name']}",
                "market": b["market"], "side": b["side"], "pick": src["pick"], "line": src.get("line"), "price": price,
                "book": o["book"] if o else "ESPN line", "edge": src["edge"], "prob": src["prob"], "ev": src["ev"], "kelly": src.get("kelly", 0.0),
                "locked_at": now.strftime("%Y-%m-%d %H:%M UTC"),
                "close_line": None, "close_prob": None, "clv": None, "result": None, "units": None, "units_kelly": None,
            })
            n += 1
    return n


def close_and_grade(ledger, ml, now, cards=None):
    games = {g["id"]: g for g in (cards or ml["games"])}
    n = 0
    for p in ledger:
        g = games.get(p["game_id"])
        if g is None or p["result"] is not None or tip(g) > now:
            continue
        home = p["side"] == "home"
        if p["market"] == "Spread" and g.get("home_spread") is not None:
            p["close_line"] = g["home_spread"] if home else -g["home_spread"]
            p["clv"] = round(p["line"] - p["close_line"], 1)
        elif p["market"] == "Total" and g.get("total") is not None:
            p["close_line"] = g["total"]
            p["clv"] = round((g["total"] - p["line"]) if p["side"] == "over" else (p["line"] - g["total"]), 1)
        elif p["market"] == "Moneyline" and g.get("home_ml") is not None and g.get("away_ml") is not None:
            ph, pa = implied(g["home_ml"]), implied(g["away_ml"])
            fair = (ph if home else pa) / (ph + pa)
            p["close_prob"] = round(fair, 3)
            p["clv"] = round(100 * (fair - implied(p["price"])), 1)
        if not g["final"]:
            continue
        hs, as_ = g["home_score"], g["away_score"]
        if p["market"] == "Spread":
            m = (hs - as_ if home else as_ - hs) + p["line"]
            res = "P" if m == 0 else ("W" if m > 0 else "L")
        elif p["market"] == "Total":
            d = hs + as_ - p["line"]
            res = "P" if d == 0 else ("W" if (d > 0) == (p["side"] == "over") else "L")
        else:
            res = "W" if (hs > as_) == home else "L"
        flat = 0.0 if res == "P" else (payout(p["price"]) if res == "W" else -1.0)
        p.update(result=res, units=round(flat, 3), units_kelly=round(flat * (p["kelly"] or 0), 3))
        n += 1
    return n


def summary(ledger):
    out = {}
    for mk in ("Spread", "Total", "Moneyline", "All"):
        ps = [p for p in ledger if mk == "All" or p["market"] == mk]
        g = [p for p in ps if p["result"]]
        c = [p["clv"] for p in ps if p["clv"] is not None and p["market"] != "Moneyline"]
        out[mk] = {"locked": len(ps), "w": sum(p["result"] == "W" for p in g), "l": sum(p["result"] == "L" for p in g),
                   "p": sum(p["result"] == "P" for p in g), "units": round(sum(p["units"] for p in g), 2),
                   "units_kelly": round(sum(p["units_kelly"] for p in g), 2), "staked_kelly": round(sum(p["kelly"] or 0 for p in g), 2),
                   "clv_pts": round(float(np.mean(c)), 2) if c else None, "beat_close": round(float(np.mean([x > 0 for x in c])), 3) if c else None}
    m = [p["clv"] for p in ledger if p["clv"] is not None and p["market"] == "Moneyline"]
    out["Moneyline"]["clv_pct"] = round(float(np.mean(m)), 2) if m else None
    return out


def main(now=None):
    now = now or datetime.now(timezone.utc)
    ml = json.loads(ML.read_text())
    data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"picks": []}
    led = data["picks"]
    added = lock_new(led, ml, now)
    graded = close_and_grade(led, ml, now)
    led.sort(key=lambda p: (p["tip"], p["id"]))
    LEDGER.write_text(json.dumps({"updated": now.strftime("%Y-%m-%d %H:%M UTC"), "thresholds": ml["thresholds"],
                                  "summary": summary(led), "picks": led}, separators=(",", ":")))
    print(f"picks: {added} locked, {graded} graded, {len(led)} in ledger")


if __name__ == "__main__":
    main()
