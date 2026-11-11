"""Model output -> website JSON in docs/data.

  ml.json       report, backtest, calibration, upcoming + recent game cards, season record, line shop
  rankings.json adjusted ratings, records and Elo for every Division I team
  history.json  every backtest game (compact) for the strategy simulator

Edge probabilities are calibrated on the out-of-sample backtest, not taken from the raw model:
spread and total = logistic fit of "pick won" on the size of the model-vs-line gap;
moneyline = logistic fit of home win on logit(market) + logit(model).
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT, SITE, RAW = ROOT / "output", ROOT / "docs" / "data", ROOT / "raw"
SITE.mkdir(parents=True, exist_ok=True)
ET = ZoneInfo("America/New_York")
# A "play": edge at least this big and positive EV at the posted price, from the 2015-2026 walk-forward backtest:
#   totals 4+ pts: 54.0% over 6,852 bets, above the 52.4% break-even in all 10 test seasons -> the model's plays.
#   spreads (clean sportsbook lines, 2019+): 7-12 pt edges hit 52.5% over 434 bets, about break-even -> leans only.
#     Above 12 pts a spread gap was usually bad line data, so those are flagged "check line".
#   moneylines: marginal at best -> fair prices only.
THRESHOLD = {"Spread": None, "Total": 4.0, "Moneyline": None}
MAX_EDGE = {"Spread": 12.0, "Total": None, "Moneyline": None}
KELLY_FRACTION, KELLY_CAP = 0.25, 3.0
CARD_DAYS_BACK, CARD_DAYS_AHEAD = 7, 3


def american_prob(o):
    o = np.asarray(o, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def clean(v):
    """NaN / inf -> None, recursively, so the JSON is valid"""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.floating,)):
        return clean(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def payout(o):
    o = np.asarray(o, float)
    return np.where(o > 0, o / 100, 100 / -o)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def r(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def kelly(p, odds):
    if p is None or odds is None or not np.isfinite(odds):
        return 0.0
    b = float(payout(odds))
    return round(float(np.clip(100 * KELLY_FRACTION * (p * b - (1 - p)) / b, 0, KELLY_CAP)), 2)


def ev100(p, odds):
    return r(100 * (p * float(payout(odds)) - (1 - p)), 1)


def sp_line(x):
    return "PK" if x == 0 else f"{x:+g}"


def main():
    pr = pd.read_parquet(OUT / "predictions.parquet")
    rep = json.loads((OUT / "report.json").read_text())
    current = int(pr.season.max())
    for c in ("home_sp_odds", "away_sp_odds", "over_odds", "under_odds"):
        pr[c] = pr[c].fillna(-110)
    hist = pr[pr.result.notna() & (pr.season < current)].copy()

    # ---- calibrators
    sp = hist[hist.home_spread.notna() & (hist.result + hist.home_spread != 0)
              & ((hist.pred_margin + hist.home_spread).abs() <= MAX_EDGE["Spread"])]   # bigger gaps were mostly bad line data
    gap = (sp.pred_margin + sp.home_spread).abs().to_numpy()
    won = np.where(sp.pred_margin + sp.home_spread > 0, sp.result + sp.home_spread > 0, sp.result + sp.home_spread < 0)
    cal_sp = LogisticRegression().fit(gap.reshape(-1, 1), won.astype(int))
    to = hist[hist.total.notna() & (hist.total_pts != hist.total)]
    gap_t = (to.pred_total - to.total).abs().to_numpy()
    won_t = np.where(to.pred_total > to.total, to.total_pts > to.total, to.total_pts < to.total)
    cal_to = LogisticRegression().fit(gap_t.reshape(-1, 1), won_t.astype(int))
    ml = hist[hist.home_ml.notna() & hist.away_ml.notna() & (hist.result != 0)].copy()
    ph, pa = american_prob(ml.home_ml), american_prob(ml.away_ml)
    ml["mkt"] = ph / (ph + pa)
    cal_ml = LogisticRegression(C=100).fit(np.c_[logit(ml.mkt), logit(ml.p_home)], (ml.result > 0).astype(int))
    # beyond ~12 points there are too few backtest bets to trust the curve, so it's held flat there
    EDGE_CAP = 12.0
    p_sp = lambda e: float(cal_sp.predict_proba([[min(abs(e), EDGE_CAP)]])[0, 1])
    p_to = lambda e: float(cal_to.predict_proba([[min(abs(e), EDGE_CAP)]])[0, 1])

    ml["bin"] = pd.cut(ml.p_home, np.linspace(0, 1, 11))
    calib = (ml.groupby("bin", observed=True).agg(pred=("p_home", "mean"), mkt=("mkt", "mean"),
                                                  actual=("result", lambda s: (s > 0).mean()), n=("p_home", "size"))
             .reset_index(drop=True))
    calib = [{k: (int(v) if k == "n" else r(v, 3)) for k, v in row.items()} for row in calib.to_dict("records")]

    # ---- cumulative units by season at the play thresholds
    def season_units(df):
        out = {}
        s = df[df.home_spread.notna()]
        e = s.pred_margin + s.home_spread
        cov = np.sign(s.result + s.home_spread)
        w = np.where(e > 0, cov > 0, cov < 0)
        o = np.where(e > 0, s.home_sp_odds, s.away_sp_odds)
        pnl = np.where(cov == 0, 0, np.where(w, payout(o), -1))
        out["spread"] = pd.Series(np.where((e.abs() >= 7) & (e.abs() <= 12), pnl, 0), index=s.index).groupby(s.season).sum()
        t = df[df.total.notna()]
        e = t.pred_total - t.total
        sd = np.sign(t.total_pts - t.total)
        w = np.where(e > 0, sd > 0, sd < 0)
        o = np.where(e > 0, t.over_odds, t.under_odds)
        pnl = np.where(sd == 0, 0, np.where(w, payout(o), -1))
        out["total"] = pd.Series(np.where(e.abs() >= THRESHOLD["Total"], pnl, 0), index=t.index).groupby(t.season).sum()
        return pd.DataFrame(out).fillna(0)
    by = season_units(hist)
    curve = [{"season": int(s), "spread": r(a), "total": r(b), "spread_cum": r(ca), "total_cum": r(cb)}
             for s, a, b, ca, cb in zip(by.index, by.spread, by.total, by.spread.cumsum(), by.total.cumsum())]

    # ---- game cards
    def bets(g):
        out = []
        if pd.notna(g.home_spread):
            e = g.pred_margin + g.home_spread
            home = e > 0
            p, odds = p_sp(e), (g.home_sp_odds if home else g.away_sp_odds)
            line = g.home_spread if home else -g.home_spread
            b = {"market": "Spread", "side": "home" if home else "away", "pick": f"{g.home_abbreviation if home else g.away_abbreviation} {sp_line(line)}",
                 "line": r(line), "edge": r(abs(e)), "unit": "pts", "prob": r(p, 3), "odds": r(odds, 0), "ev": ev100(p, odds), "kelly": kelly(p, odds)}
            if pd.notna(g.result):
                c = g.result + g.home_spread
                b["result"] = "P" if c == 0 else ("W" if (c > 0) == home else "L")
            out.append(b)
        if pd.notna(g.total):
            e = g.pred_total - g.total
            over = e > 0
            p, odds = p_to(e), (g.over_odds if over else g.under_odds)
            b = {"market": "Total", "side": "over" if over else "under", "pick": f"{'Over' if over else 'Under'} {g.total:g}",
                 "line": r(g.total), "edge": r(abs(e)), "unit": "pts", "prob": r(p, 3), "odds": r(odds, 0), "ev": ev100(p, odds), "kelly": kelly(p, odds)}
            if pd.notna(g.result):
                d = g.total_pts - g.total
                b["result"] = "P" if d == 0 else ("W" if (d > 0) == over else "L")
            out.append(b)
        if pd.notna(g.home_ml) and pd.notna(g.away_ml):
            hp, ap = float(american_prob(g.home_ml)), float(american_prob(g.away_ml))
            mk = hp / (hp + ap)
            fair = float(cal_ml.predict_proba([[logit(mk), logit(g.p_home)]])[0, 1])
            home = fair > mk
            p, odds = (fair if home else 1 - fair), (g.home_ml if home else g.away_ml)
            b = {"market": "Moneyline", "side": "home" if home else "away", "pick": f"{g.home_abbreviation if home else g.away_abbreviation} ML",
                 "line": None, "edge": r(100 * abs(fair - mk)), "unit": "%", "prob": r(p, 3), "odds": r(odds, 0), "ev": ev100(p, odds),
                 "kelly": kelly(p, odds), "mkt_home": r(mk, 3), "fair_home": r(fair, 3)}
            if pd.notna(g.result) and g.result != 0:
                b["result"] = "W" if (g.result > 0) == home else "L"
            out.append(b)
        for b in out:
            thr, cap = THRESHOLD[b["market"]], MAX_EDGE[b["market"]]
            b["check"] = bool(cap is not None and b["edge"] is not None and b["edge"] > cap)
            b["play"] = bool(thr is not None and b["edge"] is not None and b["edge"] >= thr and (b["ev"] or 0) > 0 and not b["check"])
        return out

    def card(g):
        tip = pd.Timestamp(g.game_date_time)
        tip = (tip.tz_localize("UTC") if tip.tzinfo is None else tip).tz_convert(ET)
        return {"id": str(g.game_id), "date": str(g.date)[:10], "tip": tip.strftime("%Y-%m-%dT%H:%M"),
                "home": int(g.home_id), "away": int(g.away_id), "neutral": int(g.neutral), "conf": int(g.conf_game), "post": int(g.post),
                "note": g.notes_headline if isinstance(g.notes_headline, str) else None,
                "venue": g.venue_full_name if isinstance(g.venue_full_name, str) else None, "tv": g.broadcast_name if isinstance(g.broadcast_name, str) else None,
                "home_rank": r(g.home_current_rank, 0) if r(g.home_current_rank, 0) and g.home_current_rank < 26 else None,
                "away_rank": r(g.away_current_rank, 0) if r(g.away_current_rank, 0) and g.away_current_rank < 26 else None,
                "pred_home": r(g.pred_home), "pred_away": r(g.pred_away), "margin": r(g.pred_margin), "total_pred": r(g.pred_total),
                "p_home": r(g.p_home, 3), "home_spread": r(g.home_spread), "total": r(g.total), "home_ml": r(g.home_ml, 0), "away_ml": r(g.away_ml, 0),
                "open_spread": r(g.open_home_spread), "open_total": r(g.open_total),
                "home_score": r(g.home_score, 0), "away_score": r(g.away_score, 0), "final": bool(pd.notna(g.result)), "bets": bets(g)}

    season = pr[pr.season == current].sort_values(["date", "game_date_time", "game_id"])
    now = datetime.now(ET)
    today = pd.Timestamp(now.date())
    window = season[(season.date >= today - timedelta(days=CARD_DAYS_BACK)) & (season.date <= today + timedelta(days=CARD_DAYS_AHEAD))]
    cards = [card(g) for g in window.itertuples(index=False)]

    # season record: every flagged play this season (re-graded at the closing line)
    record = []
    for g in season[season.result.notna()].itertuples(index=False):
        for b in bets(g):
            if b["play"]:
                record.append({"id": str(g.game_id), "date": str(g.date)[:10], "home": int(g.home_id), "away": int(g.away_id),
                               "score": f"{int(g.away_score)}–{int(g.home_score)}", **{k: b[k] for k in ("market", "pick", "edge", "odds", "result", "kelly")}})
    ytd = {}
    for mk in [m for m, t in THRESHOLD.items() if t is not None]:
        rs = [x for x in record if x["market"] == mk]
        ytd[mk] = {"w": sum(x["result"] == "W" for x in rs), "l": sum(x["result"] == "L" for x in rs), "p": sum(x["result"] == "P" for x in rs),
                   "units": round(sum(0 if x["result"] == "P" else (float(payout(x["odds"])) if x["result"] == "W" else -1) for x in rs), 2)}

    # ---- line shop (The Odds API), model vs every book's own number
    shop = None
    bo_path = RAW / "book_odds.json"
    if bo_path.exists():
        bo = json.loads(bo_path.read_text())
        up = season[season.result.isna()].set_index("game_id")
        rows = []
        for x in bo["rows"]:
            if x["game_id"] not in up.index:
                continue
            g = up.loc[x["game_id"]]
            base = {"id": str(x["game_id"]), "date": str(g.date)[:10], "home": int(g.home_id), "away": int(g.away_id), "book": x["book"]}
            if x.get("home_spread") is not None:
                e = g.pred_margin + x["home_spread"]
                home = e > 0
                price = x["home_sp_price"] if home else x["away_sp_price"]
                line = x["home_spread"] if home else -x["home_spread"]
                p = p_sp(e)
                rows.append({**base, "market": "Spread", "side": "home" if home else "away",
                             "pick": f"{g.home_abbreviation if home else g.away_abbreviation} {sp_line(line)}", "line": line, "price": price,
                             "edge": r(abs(e)), "prob": r(p, 3), "ev": ev100(p, price), "kelly": kelly(p, price)})
            if x.get("total") is not None:
                e = g.pred_total - x["total"]
                over = e > 0
                price = x["over_price"] if over else x["under_price"]
                p = p_to(e)
                rows.append({**base, "market": "Total", "side": "over" if over else "under", "pick": f"{'Over' if over else 'Under'} {x['total']:g}",
                             "line": x["total"], "price": price, "edge": r(abs(e)), "prob": r(p, 3), "ev": ev100(p, price), "kelly": kelly(p, price)})
            if x.get("home_ml") is not None and x.get("away_ml") is not None:
                hp, ap = float(american_prob(x["home_ml"])), float(american_prob(x["away_ml"]))
                mk = hp / (hp + ap)
                fair = float(cal_ml.predict_proba([[logit(mk), logit(g.p_home)]])[0, 1])
                home = fair > mk
                price = x["home_ml"] if home else x["away_ml"]
                p = fair if home else 1 - fair
                rows.append({**base, "market": "Moneyline", "side": "home" if home else "away",
                             "pick": f"{g.home_abbreviation if home else g.away_abbreviation} ML", "line": None, "price": price,
                             "edge": r(100 * abs(fair - mk)), "prob": r(p, 3), "ev": ev100(p, price), "kelly": kelly(p, price)})
        shop = {"fetched_at": bo.get("fetched_at"), "books": sorted({x["book"] for x in bo["rows"]}), "rows": rows}

    # ---- teams + rankings
    rat = pd.read_parquet(ROOT / "data" / "ratings.parquet")
    latest = rat[(rat.season == current) & rat.date.isna()].copy()
    latest["net"] = latest.adj_o + latest.adj_d
    gm = pd.read_parquet(ROOT / "data" / "games.parquet")
    gs = gm[gm.season == current]
    info = {}
    for s in ("home", "away"):
        for t in gs.itertuples():
            tid = int(getattr(t, f"{s}_id"))
            info.setdefault(tid, {"name": getattr(t, f"{s}_location"), "full": getattr(t, f"{s}_display_name"),
                                  "abbr": getattr(t, f"{s}_abbreviation"), "color": "#" + str(getattr(t, f"{s}_color") or "555555").lstrip("#"),
                                  "logo": getattr(t, f"{s}_logo")})
    fin = gs[gs.result.notna()]
    wl = {}
    for t in fin.itertuples():
        hw = t.result > 0
        wl.setdefault(t.home_id, [0, 0])[0 if hw else 1] += 1
        wl.setdefault(t.away_id, [0, 0])[1 if hw else 0] += 1
    tg = pd.read_parquet(ROOT / "data" / "team_games.parquet", columns=["season", "team", "elo", "date", "is_final"])
    elo_now = tg[tg.season == current].sort_values("date").groupby("team").elo.last()
    latest["rank"] = latest.net.rank(ascending=False, method="first").astype(int)
    prev = rat[(rat.season == current - 1) & rat.date.isna()].assign(net=lambda d: d.adj_o + d.adj_d)
    prev["rank_ly"] = prev.net.rank(ascending=False, method="first").astype(int)
    latest = latest.merge(prev[["team", "rank_ly"]], on="team", how="left")
    rankings = []
    for x in latest.sort_values("rank").itertuples():
        i = info.get(int(x.team))
        if not i:
            continue
        w, l = wl.get(int(x.team), [0, 0])
        rankings.append({"team": int(x.team), "rank": int(x.rank), "rank_ly": r(x.rank_ly, 0), "w": w, "l": l,
                         "adj_o": r(x.mu + x.adj_o), "adj_d": r(x.mu - x.adj_d), "net": r(x.net), "tempo": r(x.mu_t + x.adj_t),
                         "elo": r(elo_now.get(x.team), 0), "games": int(x.games)})
    # four factors (latest EWMA) for game pages
    tgf = pd.read_parquet(ROOT / "data" / "team_games.parquet")
    ff_cols = [c for c in tgf.columns if c[:2] in ("o_", "d_")]
    ffl = tgf[tgf.season == current].sort_values("date").groupby("team")[ff_cols].last()
    ff = {int(t): {c: r(v, 3) for c, v in row.items()} for t, row in ffl.iterrows()}

    # ---- history for the simulator (all backtest games with lines)
    hcols = ["season", "pred_margin", "pred_total", "p_home", "result", "total_pts", "home_spread", "total",
             "home_sp_odds", "away_sp_odds", "over_odds", "under_odds", "home_ml", "away_ml"]
    hh = hist[hist.home_spread.notna() | hist.total.notna()][hcols]
    history = {"cols": hcols, "rows": [[r(v, 3) if isinstance(v, float) else int(v) for v in row] for row in hh.itertuples(index=False)]}

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "season": current, "today": str(today.date()),
        "data_through": str(season[season.result.notna()].date.max())[:10] if season.result.notna().any() else None,
        "thresholds": THRESHOLD, "max_edge": MAX_EDGE, "metrics": rep["metrics"], "backtest": rep["backtest"], "ats_by_season": rep["ats_by_season"],
        "importance": rep["importance"], "calibration": calib, "curve": curve,
        "edge_curve": {"spread": [{"edge": e, "prob": r(p_sp(e), 3)} for e in range(0, 13)],
                       "total": [{"edge": e, "prob": r(p_to(e), 3)} for e in range(0, 13)]},
        "games": cards, "record": record, "ytd": ytd, "shop": shop, "teams": info, "four_factors": ff,
    }
    out = clean(out)
    (OUT / "ml.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    (SITE / "ml.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    (SITE / "rankings.json").write_text(json.dumps(clean({"season": current, "teams": rankings}), separators=(",", ":")))
    (SITE / "history.json").write_text(json.dumps(clean(history), separators=(",", ":")))
    print(f"export: {len(cards)} game cards, {len(record)} graded plays, {len(rankings)} teams; ytd {ytd}")


if __name__ == "__main__":
    main()
