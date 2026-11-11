"""Pre-game features for every Division I game since 2011-12.

Everything for a game is computed only from games played on EARLIER days (no leakage):

  * adjusted efficiency: points per 100 possessions, opponent- and venue-adjusted, solved each game day
    as a weighted ridge regression (offense, defense, home court). Recent games count more, and each team
    starts the season shrunk toward its rating from last year.
  * adjusted tempo: possessions per game, same approach.
  * four factors (eFG%, turnover rate, offensive rebound rate, free-throw rate) plus 3-point rate and FT%,
    for and against, exponentially weighted and carried across seasons with regression to the mean.
  * Elo (margin-of-victory adjusted), rest days, venue, conference / postseason game.

Output: data/team_games.parquet (one row per team per game) and data/games.parquet (one row per game).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "raw", ROOT / "data"
OUT.mkdir(exist_ok=True)

D1_MIN_GAMES = 10       # a team is Division I in a season if it plays at least this many games vs the schedule
CARRY = 0.6             # share of last season's rating kept as this season's starting point
PRIOR_GAMES = 5.0       # strength of that starting point, in games
HALF_LIFE_DAYS = 50     # recency weighting inside a season
NEW_TEAM_PRIOR = -6.0   # new D1 programs start well below average (pts / 100 poss, each side)

# EWMA box-score rates
HALF_LIFE_G = 10
DECAY = 0.5 ** (1 / HALF_LIFE_G)
SEASON_CARRY = 0.35
EW_PRIOR = 4.0


# ------------------------------------------------------------------ 1. load
def load():
    sc = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("sched_*.parquet"))], ignore_index=True)
    box = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("box_*.parquet"))], ignore_index=True)
    sc = sc.drop_duplicates("game_id")
    sc["date"] = pd.to_datetime(sc.game_date).astype("datetime64[ns]")
    sc["is_final"] = sc.status_type_completed.fillna(False).astype(bool) & sc.home_score.notna() & (sc.status_type_name != "STATUS_CANCELED")
    sc = sc[~sc.status_type_name.isin(["STATUS_CANCELED", "STATUS_POSTPONED"])]
    for c in ("home_id", "away_id"):
        sc[c] = sc[c].astype(int)
    for c in ("home_score", "away_score"):
        sc[c] = pd.to_numeric(sc[c], errors="coerce")
    sc.loc[~sc.is_final, ["home_score", "away_score"]] = np.nan
    # Division I membership by season
    apps = pd.concat([sc[["season", "home_id"]].rename(columns={"home_id": "t"}),
                      sc[["season", "away_id"]].rename(columns={"away_id": "t"})])
    cnt = apps.groupby(["season", "t"]).size()
    d1 = set(cnt[cnt >= D1_MIN_GAMES].index)
    sc["home_d1"] = [(s, t) in d1 for s, t in zip(sc.season, sc.home_id)]
    sc["away_d1"] = [(s, t) in d1 for s, t in zip(sc.season, sc.away_id)]
    sc["neutral"] = sc.neutral_site.fillna(False).astype(int)
    sc["conf_game"] = sc.conference_competition.fillna(False).astype(int)
    sc["post"] = (sc.season_type == 3).astype(int)
    sc = sc.sort_values(["date", "game_id"]).reset_index(drop=True)

    b = box.copy()
    b["team_id"] = b.team_id.astype(int)
    num = ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
           "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "team_score"]
    for c in num + ["total_turnovers", "turnovers"]:
        if c in b:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["tov"] = b.total_turnovers.fillna(b.turnovers) if "total_turnovers" in b else b.turnovers
    b = b[["game_id", "team_id"] + num + ["tov"]].dropna(subset=["field_goals_attempted"])
    b = b.drop_duplicates(["game_id", "team_id"])
    o = b.merge(b.rename(columns={"team_id": "opp_id", "defensive_rebounds": "opp_drb", "offensive_rebounds": "opp_orb",
                                  "field_goals_attempted": "opp_fga", "free_throws_attempted": "opp_fta", "tov": "opp_tov"})
                [["game_id", "opp_id", "opp_drb", "opp_orb", "opp_fga", "opp_fta", "opp_tov"]], on="game_id")
    o = o[o.team_id != o.opp_id]
    o["poss_t"] = o.field_goals_attempted - o.offensive_rebounds + o.tov + 0.475 * o.free_throws_attempted
    o["poss_o"] = o.opp_fga - o.opp_orb + o.opp_tov + 0.475 * o.opp_fta
    o["poss"] = (o.poss_t + o.poss_o) / 2
    o = o[o.poss.between(40, 110)]
    return sc, o


# ------------------------------------------------------------------ 2. adjusted efficiency + tempo
def solve(X, y, w, prior, lam):
    W = sparse.diags(w)
    A = (X.T @ W @ X).toarray() + np.diag(lam)
    rhs = X.T @ (w * y) + lam * prior
    return np.linalg.solve(A, rhs)


def adjusted_ratings(sc: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """For every (season, game day, team): ratings from games strictly before that day."""
    g = sc[sc.home_d1 | sc.away_d1].copy()
    pts = box.set_index(["game_id", "team_id"])
    out, prev = [], {}            # prev[(team)] = (O, D, T) final from last season
    league = {}
    for season, gs in g.groupby("season", sort=True):
        teams = sorted(set(gs.home_id[gs.home_d1]) | set(gs.away_id[gs.away_d1]))
        idx = {t: i for i, t in enumerate(teams)}
        NON = len(teams)                      # all non-D1 opponents share one rating
        T = len(teams) + 1
        tix = lambda t, d1: idx[t] if d1 else NON
        played = gs[gs.is_final]
        # one row per team-side: eff of team vs opp
        parts = []
        for s_, o_, sign in (("home", "away", 1), ("away", "home", -1)):
            p = pd.DataFrame({"gid": played.game_id.values, "date": played.date.values,
                              "team_id": played[f"{s_}_id"].values,
                              "t": [tix(t, d) for t, d in zip(played[f"{s_}_id"], played[f"{s_}_d1"])],
                              "o": [tix(t, d) for t, d in zip(played[f"{o_}_id"], played[f"{o_}_d1"])],
                              "loc": np.where(played.neutral.values == 1, 0, sign)})
            parts.append(p)
        R = pd.concat(parts, ignore_index=True).merge(
            box[["game_id", "team_id", "team_score", "poss"]].rename(columns={"game_id": "gid"}), on=["gid", "team_id"])
        if R.empty:
            continue
        R["eff"] = 100 * R.team_score / R.poss
        R = R.sort_values(["date", "gid"]).reset_index(drop=True)
        n = len(R)
        # efficiency design: [mu, hca, O_0..O_T-1, D_0..D_T-1]
        P = 2 + 2 * T
        ri = np.repeat(np.arange(n), 4)
        ci = np.c_[np.zeros(n, int), np.ones(n, int), 2 + R.t.values, 2 + T + R.o.values].ravel()
        vv = np.c_[np.ones(n), R["loc"].values, np.ones(n), -np.ones(n)].ravel()
        X = sparse.csr_matrix((vv, (ri, ci)), shape=(n, P))
        # tempo design: one row per game
        G = R.drop_duplicates("gid")
        m = len(G)
        Xt = sparse.csr_matrix((np.c_[np.ones(m), np.ones(m), np.ones(m)].ravel(),
                                (np.repeat(np.arange(m), 3), np.c_[np.zeros(m, int), 1 + G.t.values, 1 + G.o.values].ravel())),
                               shape=(m, 1 + T))
        pO = np.array([CARRY * prev[t][0] if t in prev else NEW_TEAM_PRIOR for t in teams] + [-14.0])
        pD = np.array([CARRY * prev[t][1] if t in prev else NEW_TEAM_PRIOR for t in teams] + [-14.0])
        pT = np.array([CARRY * prev[t][2] if t in prev else 0.0 for t in teams] + [0.0])
        lam = np.r_[1e-6, 3000.0, np.full(2 * T, PRIOR_GAMES)]   # home court anchored to last season
        lam_t = np.r_[1e-6, np.full(T, PRIOR_GAMES)]
        prior = np.r_[league.get("mu", 103.0), league.get("hca", 3.0), pO, pD]
        prior_t = np.r_[league.get("tempo", 69.0), pT]
        days = sorted(set(gs.date))
        today = gs.date.max() + pd.Timedelta(days=1)
        res = None
        for d in days + [today]:
            mask = (R.date < d).values
            if mask.sum() < 50:
                beta, bt = prior, prior_t
            else:
                age = (d - R.date[mask]).dt.days.values
                w = 0.5 ** (age / HALF_LIFE_DAYS)
                beta = solve(X[mask], R.eff.values[mask], w, prior, lam)
                gm = (G.date < d).values
                wt = 0.5 ** ((d - G.date[gm]).dt.days.values / HALF_LIFE_DAYS)
                bt = solve(Xt[gm], G.poss.values[gm], wt, prior_t, lam_t)
            nplay = pd.Series(np.r_[R.t.values[mask]]).value_counts()
            mu, hca = beta[0], beta[1]
            O, D, Tm = beta[2:2 + T], beta[2 + T:], bt[1:]
            out.append(pd.DataFrame({"season": season, "date": d if d != today else pd.NaT, "team": teams,
                                     "adj_o": O[:-1], "adj_d": D[:-1], "adj_t": Tm[:-1], "mu": mu, "hca": hca,
                                     "mu_t": bt[0], "games": [int(nplay.get(i, 0)) for i in range(len(teams))]}))
            res = (O, D, Tm, mu, hca, bt[0])
        O, D, Tm, mu, hca, mt = res
        prev = {t: (O[i], D[i], Tm[i]) for t, i in idx.items()}
        league = {"mu": mu, "hca": hca, "tempo": mt}
        print(f"  {season}: {len(teams)} D1 teams, {len(played):,} games, mu {mu:.1f} hca {hca:.2f} tempo {mt:.1f}")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ 3. EWMA four factors
EW = {  # name: (numerator, denominator) per team-game, offense; defense uses the opponent's line
    "efg": ("efg_num", "field_goals_attempted"), "tov": ("tov", "poss_t"), "orb": ("offensive_rebounds", "orb_den"),
    "ftr": ("free_throws_attempted", "field_goals_attempted"), "tpr": ("three_point_field_goals_attempted", "field_goals_attempted"),
    "ftp": ("free_throws_made", "free_throws_attempted"), "tpp": ("three_point_field_goals_made", "three_point_field_goals_attempted"),
}


def ewma_rates(sc, box):
    b = box.copy()
    b["efg_num"] = b.field_goals_made + 0.5 * b.three_point_field_goals_made
    b["orb_den"] = b.offensive_rebounds + b.opp_drb
    gm = sc[["game_id", "season", "date"]]
    b = b.merge(gm, on="game_id")
    league = {k: b[n].sum() / b[d].sum() for k, (n, d) in EW.items()}
    pw = {k: EW_PRIOR * b[d].mean() for k, (n, d) in EW.items()}
    out = []
    # offense: a team's own lines; defense: its opponents' lines in games against it
    for side, frame, key in (("o", b, "team_id"), ("d", b, "opp_id")):
        frame = frame.sort_values(["date", "game_id"])
        for team, d in frame.groupby(key, sort=False):
            st = {k: [0.0, 0.0] for k in EW}
            last = None
            recs = []
            dates = d.date.values
            for i, r in enumerate(d.itertuples(index=False)):
                if last is not None and r.season != last:
                    for k in st:
                        st[k][0] *= SEASON_CARRY; st[k][1] *= SEASON_CARRY
                rec = {"team": team, "game_id": r.game_id}
                for k in EW:
                    rec[f"{side}_{k}"] = (st[k][0] + league[k] * pw[k]) / (st[k][1] + pw[k])
                recs.append(rec)
                for k, (num, den) in EW.items():
                    nv, dv = getattr(r, num), getattr(r, den)
                    if pd.notna(nv) and pd.notna(dv):
                        st[k][0] = st[k][0] * DECAY + nv
                        st[k][1] = st[k][1] * DECAY + dv
                last = r.season
            nxt = {"team": team, "game_id": -1}
            for k in EW:
                nxt[f"{side}_{k}"] = (st[k][0] + league[k] * pw[k]) / (st[k][1] + pw[k])
            recs.append(nxt)
            out.append(pd.DataFrame(recs))
    o = pd.concat([x for x in out if "o_efg" in x], ignore_index=True)
    d = pd.concat([x for x in out if "d_efg" in x], ignore_index=True)
    return o.merge(d, on=["team", "game_id"], how="outer")


# ------------------------------------------------------------------ 4. Elo
def elo(sc, k=26, hca=75, revert=0.35):
    r, seen, rows = {}, {}, []
    for g in sc.itertuples():
        for t, d1 in ((g.home_id, g.home_d1), (g.away_id, g.away_d1)):
            if t not in r:
                r[t], seen[t] = (1500.0 if d1 else 1150.0), g.season
            elif seen[t] != g.season:
                r[t] = r[t] * (1 - revert) + 1500 * revert; seen[t] = g.season
        rows.append((g.game_id, r[g.home_id], r[g.away_id]))
        if not g.is_final:
            continue
        diff = r[g.home_id] - r[g.away_id] + (0 if g.neutral else hca)
        exp = 1 / (1 + 10 ** (-diff / 400))
        m = g.home_score - g.away_score
        act = 1.0 if m > 0 else 0.0
        wd = diff if m > 0 else -diff
        mult = math.log(abs(m) + 1) * 2.2 / (wd * 0.001 + 2.2)
        sh = k * mult * (act - exp)
        r[g.home_id] += sh; r[g.away_id] -= sh
    return pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away"])


# ------------------------------------------------------------------ 5. assemble
def build():
    print("1/5 loading")
    sc, box = load()
    print("2/5 adjusted efficiency and tempo")
    rat = adjusted_ratings(sc, box)
    print("3/5 four factors, Elo, rest")
    ew = ewma_rates(sc, box)
    el = elo(sc)
    # rest days
    side = pd.concat([sc[["game_id", "season", "date", "home_id"]].rename(columns={"home_id": "team"}),
                      sc[["game_id", "season", "date", "away_id"]].rename(columns={"away_id": "team"})]).sort_values(["date", "game_id"])
    side["rest"] = side.groupby(["team", "season"]).date.diff().dt.days.clip(upper=10).fillna(10)
    rest = side.set_index(["game_id", "team"]).rest

    print("4/5 team-perspective rows")
    g = sc[sc.home_d1 & sc.away_d1].copy()
    lines = pd.read_csv(RAW / "lines.csv") if (RAW / "lines.csv").exists() else pd.DataFrame(columns=["game_id"])
    if (RAW / "upcoming_lines.csv").exists():                     # today's lines for unplayed games (pipeline/espn.py)
        up = pd.read_csv(RAW / "upcoming_lines.csv")
        lines = pd.concat([lines[~lines.game_id.isin(up.game_id)], up], ignore_index=True)
    for c in ("home_ml", "away_ml", "home_sp_odds", "away_sp_odds", "over_odds", "under_odds", "open_home_ml", "open_away_ml"):
        if c in lines:                                  # ESPN sometimes stores a missing price as 0
            lines[c] = pd.to_numeric(lines[c], errors="coerce").where(lambda x: x.abs() >= 100)
    # Spread direction can't be trusted unless it comes from a named sportsbook AND agrees with that book's moneyline:
    # ESPN's older "consensus" records sometimes swap home and away (e.g. a 17-pt road favorite that was really the home team),
    # which creates fake edges that "win" the backtest. Totals have no side, so they're kept from every source.
    if "home_spread" in lines:
        hs, hm, am = (pd.to_numeric(lines[c], errors="coerce") for c in ("home_spread", "home_ml", "away_ml"))
        ok = lines.provider.notna() & (lines.provider != "consensus") & hm.notna() & am.notna() & \
             ((hs == 0) | (hm == am) | ((hs < 0) == (hm < am)))
        for c in ("home_spread", "home_sp_odds", "away_sp_odds", "open_home_spread", "home_ml", "away_ml", "open_home_ml", "open_away_ml"):
            if c in lines:
                lines.loc[~ok, c] = np.nan
    g = g.merge(lines.drop(columns=[c for c in ("season",) if c in lines]), on="game_id", how="left")
    g = g.merge(el, on="game_id", how="left")
    g["result"] = g.home_score - g.away_score
    g["total_pts"] = g.home_score + g.away_score
    rows = []
    for s, o in (("home", "away"), ("away", "home")):
        t = pd.DataFrame({"game_id": g.game_id, "season": g.season, "date": g.date, "side": s,
                          "team": g[f"{s}_id"], "opp": g[f"{o}_id"], "is_final": g.is_final,
                          "loc": np.where(g.neutral == 1, 0, 1 if s == "home" else -1),
                          "points": g[f"{s}_score"], "opp_points": g[f"{o}_score"],
                          "elo": g[f"elo_{s}"], "opp_elo": g[f"elo_{o}"],
                          "conf_game": g.conf_game, "post": g.post, "neutral": g.neutral})
        rows.append(t)
    tg = pd.concat(rows, ignore_index=True)
    # ratings entering the game day (last solve on or before that date) / latest for unplayed
    rat["date"] = pd.to_datetime(rat.date).astype("datetime64[ns]")
    rat_played = rat.dropna(subset=["date"])
    rat_next = rat[rat.date.isna()].drop(columns="date")
    tg = tg.sort_values("date")
    keyed = []
    for season, d in tg.groupby("season"):
        rp = rat_played[rat_played.season == season].sort_values("date")
        fin = d[d.is_final].sort_values("date")
        a = pd.merge_asof(fin, rp.drop(columns="season"), on="date", by="team", direction="backward")
        b = pd.merge_asof(a.sort_values("date"), rp.drop(columns="season").rename(columns=lambda c: c if c in ("date",) else f"opp_{c}")
                          .rename(columns={"opp_team": "opp"}), on="date", by="opp", direction="backward")
        un = d[~d.is_final]
        rn = rat_next[rat_next.season == season].drop(columns="season")
        un = un.merge(rn, on="team", how="left").merge(rn.rename(columns=lambda c: f"opp_{c}" if c != "team" else "opp"), on="opp", how="left")
        keyed += [b, un]
    tg = pd.concat(keyed, ignore_index=True)
    # EWMA rates: for finals the row keyed by game_id, for unplayed the team's latest state
    ewp = ew[ew.game_id != -1].set_index(["game_id", "team"])
    ewn = ew[ew.game_id == -1].drop(columns="game_id").set_index("team")
    fin = tg.is_final
    A = tg[fin].join(ewp, on=["game_id", "team"]).join(ewp.add_prefix("opp_"), on=["game_id", "opp"])
    B = tg[~fin].join(ewn, on="team").join(ewn.add_prefix("opp_"), on="opp")
    tg = pd.concat([A, B], ignore_index=True)
    tg["rest"] = [rest.get((gid, t), 10) for gid, t in zip(tg.game_id, tg.team)]
    tg["opp_rest"] = [rest.get((gid, t), 10) for gid, t in zip(tg.game_id, tg.opp)]
    tg["exp_eff"] = tg.mu + tg.adj_o - tg.opp_adj_d + tg.hca * tg["loc"]
    tg["exp_poss"] = tg.mu_t + tg.adj_t + tg.opp_adj_t
    tg["exp_pts"] = tg.exp_eff * tg.exp_poss / 100
    tg["day"] = (tg.date - tg.groupby("season").date.transform("min")).dt.days
    tg = tg.sort_values(["date", "game_id", "side"]).reset_index(drop=True)

    keep = ["game_id", "season", "season_type", "date", "game_date_time", "is_final", "neutral", "conf_game", "post",
            "home_id", "away_id", "home_location", "away_location", "home_abbreviation", "away_abbreviation",
            "home_display_name", "away_display_name", "home_color", "away_color", "home_logo", "away_logo",
            "home_score", "away_score", "result", "total_pts", "venue_full_name", "venue_address_city", "venue_address_state",
            "notes_headline", "home_current_rank", "away_current_rank", "broadcast_name"] + \
           [c for c in lines.columns if c not in ("game_id", "season")]
    print("5/5 writing")
    tg.to_parquet(OUT / "team_games.parquet", index=False)
    g[[c for c in keep if c in g]].to_parquet(OUT / "games.parquet", index=False)
    rat.to_parquet(OUT / "ratings.parquet", index=False)
    print(f"done: {len(g):,} D1 games, {len(tg):,} team rows, seasons {g.season.min()}-{g.season.max()}")


if __name__ == "__main__":
    build()
