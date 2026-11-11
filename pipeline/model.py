"""Train, walk-forward test and compare to Vegas.

Each team's points are predicted from its adjusted offense vs the opponent's adjusted defense, both
tempos, four factors for and against, Elo, venue, rest and point in the season. Two learners are blended:
ridge regression (stable, linear) and LightGBM (interactions). Game outputs: projected score for each
team -> margin and total. Win probability comes from the projected margin through a logistic curve fit
on earlier seasons.

Testing is walk-forward: season S is predicted by a model trained only on seasons before S, so every
backtest number is out of sample. Betting lines are never model inputs, only used for evaluation.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

FIRST_TRAIN = 2013          # 2012 only warms up the ratings
FIRST_TEST = 2015           # lines start in 2014; first test season needs two training seasons
EW = ["efg", "tov", "orb", "ftr", "tpr", "ftp", "tpp"]
W_RIDGE = float(os.environ.get("W_RIDGE", "0.6"))
TOTAL_BIAS_HALFLIFE = 600   # games (~10% of a season)
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=600, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def feature_frame(t: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=t.index)
    X["exp_pts"], X["exp_eff"], X["exp_poss"] = t.exp_pts, t.exp_eff, t.exp_poss
    for c in ("adj_o", "adj_d", "adj_t", "opp_adj_o", "opp_adj_d", "opp_adj_t", "mu", "mu_t"):
        X[c] = t[c]
    for k in EW:
        X[f"o_{k}"], X[f"oppd_{k}"] = t[f"o_{k}"], t[f"opp_d_{k}"]
        X[f"d_{k}"], X[f"oppo_{k}"] = t[f"d_{k}"], t[f"opp_o_{k}"]
    X["elo_diff"] = t.elo - t.opp_elo
    X["loc"], X["neutral"], X["conf"], X["post"] = t["loc"], t.neutral, t.conf_game, t.post
    X["rest"], X["opp_rest"] = t.rest.clip(0, 10), t.opp_rest.clip(0, 10)
    X["b2b"] = (t.rest <= 1).astype(int)
    X["games"], X["opp_games"] = t.games, t.opp_games
    X["day"] = t.day
    return X


def fit_predict(train, test):
    Xtr, Xte = feature_frame(train), feature_frame(test)
    ytr = train.points.values
    w = 0.9 ** (train.season.max() - train.season)
    med = Xtr.median().fillna(0)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    ridge.fit(Xtr.fillna(med), ytr, ridge__sample_weight=w)
    gbm = lgb.LGBMRegressor(**LGB_PARAMS)
    gbm.fit(Xtr, ytr, sample_weight=w)
    p = W_RIDGE * ridge.predict(Xte.fillna(med)) + (1 - W_RIDGE) * gbm.predict(Xte)
    ptr = W_RIDGE * ridge.predict(Xtr.fillna(med)) + (1 - W_RIDGE) * gbm.predict(Xtr)
    return p, ptr, gbm


def to_games(t, pred):
    t = t.assign(pred=pred)
    h = t[t.side == "home"].set_index("game_id")
    a = t[t.side == "away"].set_index("game_id")
    g = pd.DataFrame({"pred_home": h.pred, "pred_away": a.pred.reindex(h.index)})
    g["pred_margin"] = g.pred_home - g.pred_away
    g["pred_total"] = g.pred_home + g.pred_away
    return g


def american_prob(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def payout(o):
    o = np.asarray(o, float)
    return np.where(o > 0, o / 100, 100 / -o)


def debias_totals(pr):
    """Subtract the model's running lean against the closing total (games from earlier days only)."""
    pr = pr.sort_values(["date"], kind="stable").copy()
    done = pr.result.notna() & pr.total.notna()
    resid = (pr.pred_total - pr.total).where(done)
    bias = resid.ewm(halflife=TOTAL_BIAS_HALFLIFE, ignore_na=True).mean()
    day_bias = bias.groupby(pr.date).last().shift(1).ffill().fillna(0)
    pr["total_bias"] = pr.date.map(day_bias).values
    pr["pred_total"] -= pr.total_bias
    pr["pred_home"] -= pr.total_bias / 2
    pr["pred_away"] -= pr.total_bias / 2
    return pr


def run():
    t = pd.read_parquet(DATA / "team_games.parquet")
    games = pd.read_parquet(DATA / "games.parquet").set_index("game_id")
    t = t[(t.season >= FIRST_TRAIN) & t.adj_o.notna() & t.opp_adj_o.notna()]
    done = t[t.is_final]
    current = int(t.season.max())
    res, imp = [], None
    for S in range(FIRST_TEST, current + 1):
        train, test = done[done.season < S], t[t.season == S]
        pred, ptr, gbm = fit_predict(train, test)
        tg = to_games(train, ptr).join(games[["result"]])
        tg = tg[tg.result.notna() & (tg.result != 0)]
        cal = LogisticRegression(C=10).fit(tg[["pred_margin"]], (tg.result > 0).astype(int))
        g = to_games(test, pred)
        g = g[g.pred_away.notna()]
        g["p_home"] = cal.predict_proba(g[["pred_margin"]])[:, 1]
        g["sd_margin"] = float(np.std(tg.result - tg.pred_margin))
        res.append(g)
        imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feature_frame(train).columns)
        print(f"  {S}: trained on {train.game_id.nunique():,} games, predicted {len(g):,}")
    pr = pd.concat(res).join(games).reset_index()
    pr = debias_totals(pr).set_index("game_id")

    ev = pr[pr.result.notna() & (pr.season < current)].copy()
    ev["home_win"] = (ev.result > 0).astype(int)
    sp = ev[ev.home_spread.notna()]
    to = ev[ev.total.notna()]
    ml_ok = ev.home_ml.notna() & ev.away_ml.notna()
    ph, pa = american_prob(ev.home_ml.fillna(-110)), american_prob(ev.away_ml.fillna(-110))
    ev["mkt_home"] = np.where(ml_ok, ph / (ph + pa), np.nan)
    m = ev[ml_ok]
    metrics = {
        "test_seasons": f"{FIRST_TEST}-{current - 1}", "games": int(len(ev)), "games_with_lines": int(len(sp)),
        "margin_mae": {"model": round(float((sp.result - sp.pred_margin).abs().mean()), 2),
                       "vegas": round(float((sp.result + sp.home_spread).abs().mean()), 2)},
        "total_mae": {"model": round(float((to.total_pts - to.pred_total).abs().mean()), 2),
                      "vegas": round(float((to.total_pts - to.total).abs().mean()), 2)},
        "winner_acc": {"model": round(float(((m.p_home > .5) == m.home_win).mean()), 4),
                       "vegas": round(float(((m.mkt_home > .5) == m.home_win).mean()), 4)},
        "brier": {"model": round(float(((m.p_home - m.home_win) ** 2).mean()), 4),
                  "vegas": round(float(((m.mkt_home - m.home_win) ** 2).mean()), 4)},
        "corr_with_vegas_spread": round(float(np.corrcoef(sp.pred_margin, -sp.home_spread)[0, 1]), 3),
    }

    bt = ev.copy()
    for c in ("home_sp_odds", "away_sp_odds", "over_odds", "under_odds"):
        bt[c] = bt[c].fillna(-110)
    rows = []
    sb = bt[bt.home_spread.notna()]
    tb = bt[bt.total.notna()]
    for thr in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10):
        e = sb.pred_margin + sb.home_spread              # >0: home covers by e in the model
        home = e > 0
        cov = np.sign(sb.result + sb.home_spread)
        won = np.where(home, cov > 0, cov < 0)
        push = cov == 0
        odds = np.where(home, sb.home_sp_odds, sb.away_sp_odds)
        pnl = np.where(push, 0, np.where(won, payout(odds), -1))
        s = (e.abs() >= max(thr, .5)).values
        rows.append({"market": "Spread", "edge": f"≥{thr} pts", "bets": int(s.sum()), "win_pct": round(float(won[s & ~push].mean()), 4),
                     "units": round(float(pnl[s].sum()), 1), "roi": round(float(pnl[s].mean()), 4)})
        e = tb.pred_total - tb.total
        over = e > 0
        sd = np.sign(tb.total_pts - tb.total)
        won = np.where(over, sd > 0, sd < 0)
        push = sd == 0
        odds = np.where(over, tb.over_odds, tb.under_odds)
        pnl = np.where(push, 0, np.where(won, payout(odds), -1))
        s = (e.abs() >= max(thr, .5)).values
        rows.append({"market": "Total", "edge": f"≥{thr} pts", "bets": int(s.sum()), "win_pct": round(float(won[s & ~push].mean()), 4),
                     "units": round(float(pnl[s].sum()), 1), "roi": round(float(pnl[s].mean()), 4)})
    mb = bt[bt.mkt_home.notna()]
    for thr in (0.0, 0.03, 0.05, 0.08, 0.10):
        eh = mb.p_home - mb.mkt_home
        home = eh > 0
        sel = (np.abs(eh) >= max(thr, .005)).values
        won = np.where(home, mb.result > 0, mb.result < 0)
        odds = np.where(home, mb.home_ml, mb.away_ml)
        pnl = np.where(won, payout(odds), -1)
        rows.append({"market": "Moneyline", "edge": f"≥{int(thr * 100)}%", "bets": int(sel.sum()), "win_pct": round(float(won[sel].mean()), 4),
                     "units": round(float(pnl[sel].sum()), 1), "roi": round(float(pnl[sel].mean()), 4)})
    backtest = pd.DataFrame(rows)

    e = sb.pred_margin + sb.home_spread
    cov = np.sign(sb.result + sb.home_spread)
    w = np.where(e > 0, cov > 0, cov < 0)
    k = (e.abs() >= 7) & (cov != 0)
    by_season = (pd.DataFrame({"season": sb.season[k], "w": w[k.values]}).groupby("season").w.agg(["mean", "size"])
                 .rename(columns={"mean": "win_pct", "size": "bets"}).reset_index())
    by_season["win_pct"] = by_season.win_pct.round(3)

    pr.reset_index().to_parquet(OUT / "predictions.parquet", index=False)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    out = {"metrics": metrics, "backtest": backtest.to_dict("records"), "ats_by_season": by_season.to_dict("records"),
           "importance": imp.round(4).head(20).to_dict()}
    (OUT / "report.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["metrics"], indent=2))
    print(pd.DataFrame(r["backtest"]).to_string(index=False))
    print(pd.Series(r["importance"]).head(10).to_string())
