# Hatch's Hoops

**A machine-learning men's college basketball model: opponent-adjusted efficiency ratings for every Division I team, projected scores and win probabilities for every game, and plays priced against the line.**

Sister site to [Hatch's Picks](https://brycemhatch-eng.github.io/Hatch-s-Picks/) (NFL). The site rebuilds itself twice a day with GitHub Actions: it pulls the latest results and lines, retrains the model, and publishes new picks.

| Page | What it shows |
|---|---|
| **Today** | Best bets, then every game on the slate with the model's score, win probability, the line and the model's picks. Tabs for nearby days |
| **Game pages** | Model vs the line for spread, total and moneyline; adjusted ratings and four factors side by side; every sportsbook's price |
| **Rankings** | Sortable power rankings for all D-I teams: net, adjusted offense and defense, tempo, Elo, last season's rank |
| **Line Shop** | Each play at the best price across sportsbooks, with the worst price for comparison |
| **Record** | Locked plays (price when first flagged, closing line value, graded at that price) and every flagged play this season |
| **Performance** | Walk-forward backtest vs the closing line, win rate by edge size, calibration and a strategy simulator |

## Results (walk-forward, out of sample, 2014-15 through 2025-26)

| | Model | Closing line |
|---|---|---|
| Avg margin miss | 8.98 pts | 8.76 pts |
| Avg total miss | 13.50 pts | 13.35 pts |
| Picks the winner | 70.9% | 71.7% |
| **Totals, 4+ pt edges** | **54.0%** (6,852 bets, +3.1% ROI) | break-even 52.4% |
| Spreads, 7-12 pt edges | 52.5% (434 bets, about break-even) | |

The line is a little more accurate game by game, as expected. The model's edge is in **totals**: at 4+ points of disagreement it beat break-even in every one of the 10 test seasons. Spreads didn't show a reliable edge, so the site lists them as leans, not plays.

Caveats, so the numbers are read correctly:
- The 4-point threshold was chosen after looking at the backtest, so the true edge is probably somewhat smaller.
- ESPN's public data has no lines for 2023-24 and 2024-25; those seasons are trained on but skipped in the betting results.
- ESPN's older "consensus" lines (2014-18) sometimes swap home and away. Spread results use only lines from a named sportsbook that agree with that book's moneyline (2019 on). Totals have no side, so every season counts.

## How it works

1. **Data:** schedules and team box scores for every D-I game since 2011-12 from [sportsdataverse / hoopR](https://github.com/sportsdataverse/hoopR-mbb-data), and the pregame lines ESPN publishes with each game. Today's lines come from ESPN's scoreboard; book-by-book prices from [The Odds API](https://the-odds-api.com) (optional).
2. **Adjusted efficiency:** for every game day, each team's offense and defense (points per 100 possessions) and home-court advantage are solved together as a weighted ridge regression over the games played so far. Recent games count more, and each team starts the season anchored to 60% of last year's rating. Tempo is rated the same way.
3. **Other features (all known before tip-off):** four factors for and against (eFG%, turnover rate, offensive rebound rate, FT rate) plus 3-point rate and FT%, exponentially weighted; Elo; rest days; home / away / neutral; conference and postseason games; point in the season.
4. **Model:** ridge regression plus LightGBM, blended 60/40, each predicting a team's points. A logistic layer turns the projected margin into win probability.
5. **Testing:** walk-forward by season. Each year is predicted by a model trained only on earlier years.
6. **Totals de-bias:** before each game day, the model's running lean against the closing total is subtracted, so it can't drift into betting one side.
7. **Edge:** the gap between the model and a line becomes a win probability using how often gaps that size won in the backtest (held flat above 12 points). That gives expected value at the posted price and a quarter-Kelly stake (1 unit = 1% of bankroll, max 3).
8. **Locked plays:** `pipeline/picks.py` records each play once, at the best book price when first flagged, then tracks closing line value and grades it at that price.

```
pipeline/
  run.py          one command: download -> lines -> today's lines -> book odds -> features -> model -> export -> picks
  download.py     schedules + team box scores (hoopR)
  lines.py        pregame lines for every finished game -> raw/lines.csv (cached; only new games are fetched)
  espn.py         lines for today and the next 3 days (ESPN scoreboard)
  odds.py         sportsbook prices (The Odds API), matched to ESPN game ids
  features.py     adjusted ratings, four factors, Elo, rest -> data/
  model.py        walk-forward training + backtest vs the line
  export.py       website JSON (docs/data)
  picks.py        locked picks ledger -> docs/data/picks.json
docs/             the website (GitHub Pages)
.github/workflows/daily.yml   twice-daily refresh
```

## Set up

1. Create a new GitHub repository (e.g. `Hatchs-Hoops`) and upload everything in this folder, including the hidden `.github` folder.
2. **Settings → Pages:** deploy from the `main` branch, `/docs` folder.
3. **Settings → Actions → General → Workflow permissions:** choose "Read and write permissions" so the daily job can commit.
4. Optional, for the Line Shop: **Settings → Secrets and variables → Actions → New repository secret**, name `ODDS_API_KEY`, the same key as Hatch's Picks.
5. **Actions → Daily model refresh → Run workflow** to build the site the first time. It runs on its own at 5:15 AM and 1:45 PM Arizona time after that.

Run it locally:

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key   # optional
python pipeline/run.py         # first run downloads ~35 MB of schedules and box scores
cd docs && python -m http.server   # open http://localhost:8000
```

*For research and entertainment. Not betting advice.*
