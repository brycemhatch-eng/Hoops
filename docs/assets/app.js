/* Hatch's Hoops — single-page site over docs/data/*.json (built daily by the pipeline).
   Routes: #/ (today) · #/day/YYYY-MM-DD · #/game/ID · #/rankings · #/lines · #/record · #/performance · #/about */
(() => {
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = (x, d = 0) => x == null ? '—' : (100 * x).toFixed(d) + '%';
const sgn = (x, d = 1) => x == null ? '—' : (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
const odds = o => o == null ? '—' : (o > 0 ? '+' + o : '−' + Math.abs(o));
const money = x => (x < 0 ? '−$' : '+$') + Math.abs(x).toFixed(Math.abs(x) < 10 ? 2 : 0);
const payout = o => o > 0 ? o / 100 : 100 / -o;
const line = x => x == null ? '—' : x === 0 ? 'PK' : sgn(x, 1).replace('.0', '');
const store = {get(k, d) { try { const v = localStorage.getItem('hh-' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
               set(k, v) { try { localStorage.setItem('hh-' + k, JSON.stringify(v)); } catch (e) {} }};

let ML, RK, PK, HI;
let stake = store.get('stake', 100);
const charts = [];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const T = id => (ML.teams[String(id)] || {name: '?', abbr: '?', color: '#555555'});
const RANK = {};
const SEASON = s => `${s - 1}–${String(s).slice(2)}`;

// ---------------------------------------------------------------- theme
const savedTheme = store.get('theme', null);
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$('#theme').onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  store.set('theme', document.documentElement.dataset.theme);
  route();
};

// ---------------------------------------------------------------- small pieces
function luminance(hex) {
  const n = parseInt(String(hex).replace('#', '').padEnd(6, '0').slice(0, 6), 16);
  const c = [n >> 16, (n >> 8) & 255, n & 255].map(v => { v /= 255; return v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; });
  return .2126 * c[0] + .7152 * c[1] + .0722 * c[2];
}
const badge = (id, s = 30) => { const t = T(id), c = t.color || '#555555';
  const ab = (t.abbr || '?').slice(0, 4), fs = s * (ab.length > 3 ? .3 : .38);
  return `<span class="tb" style="--s:${s}px;font-size:${fs}px;background:${esc(c)};color:${luminance(c) > .45 ? '#111' : '#fff'}">${esc(ab)}</span>`; };
const tname = (id, rank) => `${rank ? `<span class="rk">${rank}</span>` : ''}${esc(T(id).name)}`;
const tipTime = g => { const [d, t] = g.tip.split('T'); let [h, m] = t.split(':').map(Number);
  if (h === 0 && m === 0) return 'TBD'; const ap = h >= 12 ? 'PM' : 'AM'; h = h % 12 || 12; return `${h}:${String(m).padStart(2, '0')} ${ap}`; };
const dayLabel = d => new Date(d + 'T12:00:00').toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric'});
const foot = () => `<footer>Model data through ${esc(ML.data_through || '—')} · rebuilt ${esc(ML.generated_at)} · times are Eastern · for research and entertainment, not betting advice.</footer>`;
const killCharts = () => { while (charts.length) charts.pop().destroy(); };
function chart(el, cfg) {
  const ink = css('--muted'), grid = css('--grid');
  Chart.defaults.font.family = css('--body'); Chart.defaults.color = ink;
  cfg.options = Object.assign({responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {legend: {labels: {boxWidth: 12}}}, scales: {x: {grid: {color: grid}}, y: {grid: {color: grid}}}}, cfg.options || {});
  const c = new Chart(el, cfg); charts.push(c); return c;
}
const betChip = b => `<span class="pk-chip ${b.result ? b.result.toLowerCase() : b.play ? 'play' : ''}" title="${esc(b.market)}: model edge ${b.edge}${b.unit === '%' ? '%' : ' pts'}, ${pct(b.prob, 1)} to win">${esc(b.pick)} <small>${odds(b.odds)}</small></span>`;

// ---------------------------------------------------------------- today / day slate
let onlyPlays = store.get('onlyPlays', false), q = '';
function pageDay(day) {
  const days = [...new Set(ML.games.map(g => g.date))].sort();
  const dd = day && days.includes(day) ? day : (days.includes(ML.today) ? ML.today : days.find(d => d >= ML.today) || days[days.length - 1]);
  const all = ML.games.filter(g => g.date === dd).sort((a, b) => a.tip.localeCompare(b.tip));
  const plays = all.flatMap(g => g.bets.filter(b => b.play).map(b => ({g, b}))).sort((a, b) => b.b.ev - a.b.ev);
  const y = ML.ytd, tot = Object.values(y).reduce((s, x) => ({w: s.w + x.w, l: s.l + x.l, u: s.u + x.units}), {w: 0, l: 0, u: 0});
  const bt = ML.backtest.find(x => x.market === 'Total' && x.edge === `≥${ML.thresholds.Total} pts`) || {};
  $('#hero').innerHTML = `<section class="hero"><div class="wrap">
    <div><div class="eyebrow" style="color:var(--accent)">Men's college basketball · ${SEASON(ML.season)}</div>
      <h1>Hatch's <span>Hoops</span></h1>
      <p>Opponent-adjusted efficiency ratings for all ${RK.teams.length} Division I teams feed a machine-learning model that projects every game's score, then prices it against the line.</p>
      <div class="stamp">Updated ${esc(ML.generated_at)}</div></div>
    <div class="strip">
      <div><b>${tot.w}–${tot.l}</b><small>${SEASON(ML.season)} plays</small></div>
      <div><b>${sgn(tot.u, 1)}<i> u</i></b><small>units this season, flat</small></div>
      <div><b>${pct(bt.win_pct, 1)}</b><small>backtest totals, ${ML.thresholds.Total}+ pt edges</small></div>
      <div><b>${(bt.bets || 0).toLocaleString()}</b><small>backtest totals plays</small></div>
    </div></div></section>`;
  const ix = days.indexOf(dd), shown = days.slice(Math.max(0, ix - 3), ix + 4);
  const rows = all.filter(g => (!onlyPlays || g.bets.some(b => b.play)) &&
    (!q || (T(g.home).full + ' ' + T(g.away).full + ' ' + T(g.home).abbr + ' ' + T(g.away).abbr).toLowerCase().includes(q)));
  $('#app').innerHTML = `
    <div class="head"><div><h2>${dd === ML.today ? 'Today' : dayLabel(dd)}</h2><p>${all.length} Division I games${dd === ML.today ? ' on ' + dayLabel(dd) : ''}. A <b>play</b> is a total where the model and the line differ by ${ML.thresholds.Total}+ points with positive expected value at the posted price; that's where the backtest edge is. Spreads and moneylines are shown as model leans and fair prices.</p></div></div>
    <nav class="days" aria-label="Dates">${shown.map(d => `<a href="#/day/${d}" ${d === dd ? 'aria-current="page"' : ''}><small>${new Date(d + 'T12:00:00').toLocaleDateString('en-US', {weekday: 'short'})}</small><b>${new Date(d + 'T12:00:00').toLocaleDateString('en-US', {month: 'short', day: 'numeric'})}</b></a>`).join('')}</nav>
    ${plays.length ? `<section><div class="head" style="margin-bottom:10px"><h3 style="font-size:22px;text-transform:uppercase">Best bets</h3></div><div class="bb">${plays.slice(0, 9).map(({g, b}) => {
      const t = b.side === 'home' ? g.home : b.side === 'away' ? g.away : null;
      return `<a href="#/game/${g.id}" style="--c:${esc(t ? T(t).color : 'var(--accent)')}">
        <div class="r1"><span>${esc(T(g.away).name)} @ ${esc(T(g.home).name)}</span><span>${tipTime(g)}</span></div>
        <div class="pk">${esc(b.pick)} <span class="num" style="font-size:16px;color:var(--muted)">${odds(b.odds)}</span></div>
        <div class="r3"><span>${b.market}</span><span>edge ${b.edge}${b.unit === '%' ? '%' : ''}</span><span>win ${pct(b.prob, 1)}</span><span>${b.kelly}u Kelly</span></div>
        ${b.result ? `<span class="pill ${b.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[b.result]}</span>` : `<div class="ev ${b.ev >= 0 ? 'pos' : 'neg'}">${money(stake * b.ev / 100)} <small class="muted" style="font-size:11px">EV at $${stake}</small></div>`}</a>`; }).join('')}</div></section>` : ''}
    <div class="card"><div class="body toolbar">
      <input type="search" id="q" placeholder="Find a team" value="${esc(q)}" aria-label="Find a team" style="min-width:200px">
      <label class="tog"><input type="checkbox" id="op" ${onlyPlays ? 'checked' : ''}> Plays only</label>
      <label class="f" style="grid-auto-flow:column;align-items:center">Stake $<input id="stake" type="number" min="1" step="10" value="${stake}"></label></div>
      <div class="tbl"><table class="slate"><thead><tr><th>Time</th><th>Matchup</th><th class="n">Model</th><th class="n">Score</th><th class="n">Win %</th><th class="n">Line</th><th class="n hide-m">Total</th><th>Model picks</th></tr></thead><tbody>
      ${rows.map(g => { const fin = g.final, aw = fin && g.away_score > g.home_score;
        return `<tr onclick="location.hash='#/game/${g.id}'" style="cursor:pointer">
          <td class="num muted">${fin ? 'Final' : tipTime(g)}${g.neutral ? '<br><small class="muted">neutral</small>' : ''}</td>
          <td><div class="mu"><div class="t ${fin ? (aw ? 'win' : 'lose') : ''}">${badge(g.away, 24)}${tname(g.away, g.away_rank)}</div><div class="t ${fin ? (aw ? 'lose' : 'win') : ''}">${badge(g.home, 24)}${g.neutral ? '' : '<span class="muted" style="font-weight:400">@</span>'} ${tname(g.home, g.home_rank)}</div></div></td>
          <td class="n"><div class="stk"><span>${g.pred_away ?? '—'}</span><span>${g.pred_home ?? '—'}</span></div></td>
          <td class="n"><div class="stk">${fin ? `<span><b>${g.away_score}</b></span><span><b>${g.home_score}</b></span>` : '<span class="faint">—</span><span class="faint">—</span>'}</div></td>
          <td class="n"><div class="stk"><span>${pct(1 - g.p_home)}</span><span>${pct(g.p_home)}</span></div></td>
          <td class="n"><div class="stk"><span class="faint">${g.home_spread == null ? '—' : line(-g.home_spread)}</span><span class="faint">${line(g.home_spread)}</span></div></td>
          <td class="n hide-m num faint">${g.total ?? '—'}</td>
          <td><div class="picks">${g.bets.filter(b => b.play || b.result).filter(b => b.play).map(betChip).join('') || (g.bets.length ? '<span class="muted" style="font-size:12px">no edge</span>' : '<span class="muted" style="font-size:12px">no line yet</span>')}</div></td></tr>`; }).join('') || '<tr><td colspan="8" class="empty">No games match.</td></tr>'}
      </tbody></table></div></div>${foot()}`;
  $('#q').oninput = e => { q = e.target.value.trim().toLowerCase(); clearTimeout(window._q); window._q = setTimeout(() => { route.keep = true; pageDay(dd); $('#q').focus(); $('#q').setSelectionRange(99, 99); }, 250); };
  $('#op').onchange = e => { onlyPlays = e.target.checked; store.set('onlyPlays', onlyPlays); pageDay(dd); };
  $('#stake').onchange = e => { stake = Math.max(1, +e.target.value || 100); store.set('stake', stake); pageDay(dd); };
}

// ---------------------------------------------------------------- game page
function pageGame(id) {
  const g = ML.games.find(x => x.id === id);
  if (!g) { $('#hero').innerHTML = ''; $('#app').innerHTML = `<div class="card empty">That game isn't in the current window. <a href="#/">Back to today</a></div>`; return; }
  const H = T(g.home), A = T(g.away), rh = RANK[g.home] || {}, ra = RANK[g.away] || {};
  const FF = ML.four_factors, fh = FF[g.home] || {}, fa = FF[g.away] || {};
  $('#hero').innerHTML = `<section class="gh"><div class="gh-bg"><i style="background:${esc(A.color)}"></i><i style="background:${esc(H.color)}"></i></div><div class="wrap">
    <a class="back" href="#/day/${g.date}">← ${dayLabel(g.date)}</a>
    <div class="gh-main">
      <div class="gh-team">${badge(g.away, 52)}<div class="nm">${esc(A.name)}</div><div class="sub">${ra.rank ? `#${ra.rank} in model · ` : ''}${ra.w ?? 0}–${ra.l ?? 0}</div><div class="wp">${pct(1 - g.p_home)}<small>win probability</small></div></div>
      <div class="gh-mid"><small>${g.final ? 'Final' : 'Model projection'}</small><div class="sc">${g.final ? `${g.away_score}–${g.home_score}` : `${Math.round(g.pred_away)}–${Math.round(g.pred_home)}`}</div>
        ${g.final ? `<div class="vg">model ${g.pred_away}–${g.pred_home}</div>` : ''}
        <div class="vg">${g.home_spread == null ? 'no line yet' : `${esc(H.abbr)} ${line(g.home_spread)} · O/U ${g.total ?? '—'}`}</div>
        <small>${dayLabel(g.date)} · ${tipTime(g)} ET${g.neutral ? ' · neutral' : ''}${g.tv ? ' · ' + esc(g.tv) : ''}</small></div>
      <div class="gh-team r">${badge(g.home, 52)}<div class="nm">${esc(H.name)}</div><div class="sub">${rh.rank ? `#${rh.rank} in model · ` : ''}${rh.w ?? 0}–${rh.l ?? 0}</div><div class="wp">${pct(g.p_home)}<small>win probability</small></div></div>
    </div></div></section>`;
  const sb = g.bets.slice().sort((a, b) => b.ev - a.ev);
  const books = ML.shop ? ML.shop.rows.filter(x => x.id === g.id) : [];
  // bars show where each team sits in the Division I range for that stat (longer = better)
  const range = k => { const v = k in (RK.teams[0] || {}) ? RK.teams.map(t => t[k]) : Object.values(FF).map(x => x[k]);
    const xs = v.filter(x => x != null).sort((a, b) => a - b); return [xs[Math.floor(xs.length * .02)], xs[Math.floor(xs.length * .98)]]; };
  const cmp = (lbl, a, h, fmt, hiGood = true, sub, key) => {
    if (a == null || h == null) return '';
    const [lo, hi] = key ? range(key) : [Math.min(a, h), Math.max(a, h)];
    const w = v => Math.max(4, Math.min(100, 100 * (hiGood ? v - lo : hi - v) / ((hi - lo) || 1)));
    const aw = (hiGood ? a >= h : a <= h);
    return `<div class="cmp-row"><div class="v ${aw ? '' : 'muted'}">${fmt(a)}</div><div class="cbar l"><i style="width:${w(a)}%;background:${esc(A.color)}"></i></div><div class="lbl">${lbl}${sub ? `<br><small class="muted">${sub}</small>` : ''}</div><div class="cbar r"><i style="width:${w(h)}%;background:${esc(H.color)}"></i></div><div class="v r ${aw ? 'muted' : ''}">${fmt(h)}</div></div>`;
  };
  const p1 = x => (100 * x).toFixed(1) + '%', n1 = x => x.toFixed(1);
  $('#app').innerHTML = `
    ${g.note ? `<div class="note">${esc(g.note)}</div>` : ''}
    <div class="cols">
      <div class="card"><h3>The bets</h3><p class="sub">Model vs the ${g.final ? 'closing' : 'current'} line. Win probability is calibrated on how often edges this size actually won in the backtest. Last column: ${g.final ? 'profit' : 'expected value'} at $${stake}.</p>
        <div class="bets" style="padding:8px 0 4px">${sb.map(b => `<div class="bet"><span class="mkt">${b.market}</span><span class="pk">${esc(b.pick)} <span>${odds(b.odds)} · edge ${b.edge}${b.unit === '%' ? '%' : ' pts'} · ${pct(b.prob, 1)} · ${b.kelly}u</span></span>${b.result ? `<span class="pill ${b.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[b.result]}</span>` : b.play ? '<span class="pill play">Play</span>' : b.check ? '<span class="pill warn" title="A gap this big was usually bad line data in the backtest">Check line</span>' : '<span class="pill lean">Lean</span>'}${(() => { const v = b.result ? (b.result === 'W' ? stake * payout(b.odds) : b.result === 'L' ? -stake : 0) : stake * b.ev / 100; return `<span class="ev ${v >= 0 ? 'pos' : 'neg'}" title="${b.result ? 'profit' : 'expected value'} at $${stake}">${money(v)}</span>`; })()}</div>`).join('') || '<div class="empty">No line posted yet.</div>'}</div>
        ${g.open_spread != null ? `<p class="sub" style="padding-bottom:14px">Opened ${esc(H.abbr)} ${line(g.open_spread)}, total ${g.open_total ?? '—'}.</p>` : ''}</div>
      <div class="card"><h3>Matchup</h3><p class="sub">Adjusted ratings are points per 100 possessions against an average D-I opponent on a neutral floor.</p><div class="body cmp">
        <div class="cmp-h"><span>${esc(A.abbr)}</span><span></span><span></span><span></span><span>${esc(H.abbr)}</span></div>
        ${cmp('Model rank', ra.rank, rh.rank, x => '#' + x, false, '', 'rank')}
        ${cmp('Adj. offense', ra.adj_o, rh.adj_o, n1, true, '', 'adj_o')}
        ${cmp('Adj. defense', ra.adj_d, rh.adj_d, n1, false, 'lower is better', 'adj_d')}
        ${cmp('Net rating', ra.net, rh.net, x => sgn(x, 1), true, '', 'net')}
        ${cmp('Tempo', ra.tempo, rh.tempo, n1, true, 'possessions / 40', 'tempo')}
        ${cmp('eFG%', fa.o_efg, fh.o_efg, p1, true, '', 'o_efg')}
        ${cmp('Turnover rate', fa.o_tov, fh.o_tov, p1, false, '', 'o_tov')}
        ${cmp('Off. rebound %', fa.o_orb, fh.o_orb, p1, true, '', 'o_orb')}
        ${cmp('FT rate', fa.o_ftr, fh.o_ftr, p1, true, '', 'o_ftr')}
        ${cmp('3-pt rate', fa.o_tpr, fh.o_tpr, p1, true, 'share of shots', 'o_tpr')}
        ${cmp('Opp. eFG%', fa.d_efg, fh.d_efg, p1, false, '', 'd_efg')}
        ${cmp('Forced TO rate', fa.d_tov, fh.d_tov, p1, true, '', 'd_tov')}
      </div></div>
    </div>
    ${books.length ? `<div class="card"><h3>Every book</h3><p class="sub">The model's side priced at each sportsbook. Updated ${esc(ML.shop.fetched_at)}.</p><div class="tbl"><table><thead><tr><th>Book</th><th>Market</th><th>Play</th><th class="n">Price</th><th class="n">Edge</th><th class="n">Win %</th><th class="n">EV at $${stake}</th></tr></thead><tbody>
      ${books.sort((a, b) => b.ev - a.ev).map(x => `<tr><td>${esc(x.book)}</td><td>${x.market}</td><td><b>${esc(x.pick)}</b></td><td class="n num">${odds(x.price)}</td><td class="n num">${x.edge}${x.market === 'Moneyline' ? '%' : ''}</td><td class="n num">${pct(x.prob, 1)}</td><td class="n num ${x.ev >= 0 ? 'pos' : 'neg'}">${money(stake * x.ev / 100)}</td></tr>`).join('')}</tbody></table></div></div>` : ''}
    ${foot()}`;
}

// ---------------------------------------------------------------- rankings
let rsort = store.get('rsort', ['rank', 1]), rq = '';
function pageRankings() {
  $('#hero').innerHTML = '';
  const cols = [['rank', '#', 1], ['team', 'Team', 1], ['w', 'Record', -1], ['net', 'Net', -1], ['adj_o', 'Adj. O', -1], ['adj_d', 'Adj. D', 1], ['tempo', 'Tempo', -1], ['elo', 'Elo', -1], ['rank_ly', 'Last season', 1]];
  const [k, dir] = rsort;
  const val = (t, c) => c === 'team' ? T(t.team).name : c === 'w' ? t.w - t.l : t[c] ?? 1e9 * (dir > 0 ? 1 : -1);
  const rows = RK.teams.filter(t => !rq || T(t.team).full.toLowerCase().includes(rq) || (T(t.team).abbr || '').toLowerCase().includes(rq))
    .sort((a, b) => { const x = val(a, k), y = val(b, k); return (typeof x === 'string' ? x.localeCompare(y) : x - y) * dir; });
  const mxO = Math.max(...RK.teams.map(t => t.adj_o)), mnO = Math.min(...RK.teams.map(t => t.adj_o));
  const mxD = Math.max(...RK.teams.map(t => t.adj_d)), mnD = Math.min(...RK.teams.map(t => t.adj_d));
  const bar = (v, lo, hi, inv) => `<span class="rkbar"><i style="width:${Math.max(4, 100 * (inv ? hi - v : v - lo) / (hi - lo))}%;background:var(--model)"></i></span>`;
  $('#app').innerHTML = `<div class="head"><div><h2>Power rankings</h2><p>Every Division I team by net adjusted efficiency: points scored minus points allowed per 100 possessions against an average opponent on a neutral floor. Early in the season each team is anchored to last year's rating; by January it's almost all this season.</p></div>
      <input type="search" id="rq" placeholder="Find a team" value="${esc(rq)}" aria-label="Find a team"></div>
    <div class="card"><div class="tbl" style="max-height:75vh"><table><thead><tr>${cols.map(([c, l]) => `<th class="sort ${['rank', 'team'].includes(c) ? '' : 'n'}" data-k="${c}" ${c === k ? `aria-sort="${dir > 0 ? 'ascending' : 'descending'}"` : ''}>${l}</th>`).join('')}</tr></thead><tbody>
    ${rows.map(t => `<tr><td class="rk">${t.rank}</td><td><span class="tlink">${badge(t.team, 26)}<span class="tn">${esc(T(t.team).full)}</span></span></td><td class="n num">${t.w}–${t.l}</td>
      <td class="n num"><b>${sgn(t.net, 1)}</b></td><td class="n num">${t.adj_o.toFixed(1)}${bar(t.adj_o, mnO, mxO)}</td><td class="n num">${t.adj_d.toFixed(1)}${bar(t.adj_d, mnD, mxD, true)}</td>
      <td class="n num">${t.tempo.toFixed(1)}</td><td class="n num">${t.elo ?? '—'}</td><td class="n num muted">${t.rank_ly ?? 'new'}</td></tr>`).join('')}
    </tbody></table></div></div>${foot()}`;
  document.querySelectorAll('th.sort').forEach(th => th.onclick = () => { const c = th.dataset.k;
    rsort = [c, c === k ? -dir : (cols.find(x => x[0] === c)[2])]; store.set('rsort', rsort); route.keep = true; pageRankings(); });
  $('#rq').oninput = e => { rq = e.target.value.trim().toLowerCase(); clearTimeout(window._rq); window._rq = setTimeout(() => { route.keep = true; pageRankings(); $('#rq').focus(); $('#rq').setSelectionRange(99, 99); }, 200); };
}

// ---------------------------------------------------------------- line shop
function pageLines() {
  $('#hero').innerHTML = '';
  const S = ML.shop;
  if (!S || !S.rows.length) { $('#app').innerHTML = `<div class="head"><div><h2>Line shop</h2><p>Book-by-book prices appear here once the daily refresh has an Odds API key (the <code>ODDS_API_KEY</code> repository secret).</p></div></div>${foot()}`; return; }
  const by = {};
  S.rows.forEach(x => { (by[`${x.id}|${x.market}|${x.side}`] ||= []).push(x); });
  const best = Object.values(by).map(rs => { rs.sort((a, b) => b.ev - a.ev); return {b: rs[0], w: rs[rs.length - 1], n: rs.length}; })
    .filter(x => ML.thresholds[x.b.market] != null && x.b.ev > 0 && x.b.edge >= ML.thresholds[x.b.market]).sort((a, b) => b.b.ev - a.b.ev);
  $('#app').innerHTML = `<div class="head"><div><h2>Line shop</h2><p>The model's side of every upcoming game at ${S.books.length} sportsbooks (${S.books.map(esc).join(', ')}). Each book's own line and price is compared to the model, so the best number often lives at a different book. Prices as of ${esc(S.fetched_at)}.</p></div></div>
    <div class="card"><h3>Plays at the best price</h3><div class="tbl"><table><thead><tr><th>Date</th><th>Game</th><th>Play</th><th>Best book</th><th class="n">Price</th><th class="n">Edge</th><th class="n">Win %</th><th class="n">Kelly</th><th class="n">EV at $${stake}</th><th class="n">Worst book</th></tr></thead><tbody>
    ${best.map(({b, w, n}) => `<tr onclick="location.hash='#/game/${b.id}'" style="cursor:pointer"><td class="num muted">${b.date.slice(5)}</td><td>${esc(T(b.away).name)} @ ${esc(T(b.home).name)}</td><td><b>${esc(b.pick)}</b></td><td>${esc(b.book)}</td><td class="n num">${odds(b.price)}</td><td class="n num">${b.edge}${b.market === 'Moneyline' ? '%' : ''}</td><td class="n num">${pct(b.prob, 1)}</td><td class="n num">${b.kelly}u</td><td class="n num pos">${money(stake * b.ev / 100)}</td><td class="n num muted">${n > 1 ? `${esc(w.book)} ${w.line == null ? '' : line(w.line) + ' '}${odds(w.price)}` : '—'}</td></tr>`).join('') || '<tr><td colspan="10" class="empty">No plays at current prices.</td></tr>'}
    </tbody></table></div></div>${foot()}`;
}

// ---------------------------------------------------------------- record
function lockedPlays() {
  if (!PK || !PK.picks) return '';
  const S = PK.summary.All, ps = [...PK.picks].sort((a, b) => b.tip.localeCompare(a.tip));
  const roiK = S.staked_kelly ? S.units_kelly / S.staked_kelly : null;
  const close = p => p.market === 'Moneyline' ? (p.close_prob == null ? '—' : pct(p.close_prob, 1)) : p.close_line == null ? '—' : p.market === 'Total' ? p.close_line : line(p.close_line);
  return `<div class="card"><h3>Locked plays</h3><p class="sub">Recorded the first time the model flags a play, at the best sportsbook price at that moment, and never changed. Closing line value (CLV) is how many points the locked number beat the closing line by; beating the close consistently is the clearest sign of a real edge. Kelly stakes are in units of 1% of bankroll (quarter Kelly, max 3).</p>
    <div class="body"><div class="kpis"><div><b class="num">${S.locked}</b><small>plays locked</small></div><div><b class="num">${S.w}–${S.l}${S.p ? '–' + S.p : ''}</b><small>graded at locked price</small></div>
      <div><b class="num ${S.units >= 0 ? 'pos' : 'neg'}">${money(S.units * stake)}</b><small>flat $${stake}/play</small></div>
      <div><b class="num ${S.units_kelly >= 0 ? 'pos' : 'neg'}">${sgn(S.units_kelly, 2)}u</b><small>Kelly-sized${roiK == null ? '' : ' · ' + pct(roiK, 1) + ' ROI'}</small></div>
      <div><b class="num ${(S.clv_pts ?? 0) >= 0 ? 'pos' : 'neg'}">${S.clv_pts == null ? '—' : sgn(S.clv_pts, 2)}</b><small>avg CLV (pts)</small></div>
      <div><b class="num">${S.beat_close == null ? '—' : pct(S.beat_close)}</b><small>beat the close</small></div></div></div>
    <div class="tbl"><table><thead><tr><th>Date</th><th>Game</th><th>Play</th><th>Book</th><th class="n">Locked</th><th class="n">Price</th><th class="n">Close</th><th class="n">CLV</th><th class="n">Kelly</th><th>Result</th><th class="n">Units</th></tr></thead><tbody>
    ${ps.map(p => `<tr><td class="num muted">${p.date.slice(5)}</td><td>${esc(p.matchup)}</td><td><b>${esc(p.pick)}</b></td><td>${esc(p.book)}</td><td class="n num">${p.market === 'Moneyline' ? 'ML' : p.market === 'Total' ? p.line : line(p.line)}</td><td class="n num">${odds(p.price)}</td><td class="n num">${close(p)}</td><td class="n num ${p.clv > 0 ? 'pos' : p.clv < 0 ? 'neg' : ''}">${p.clv == null ? '—' : sgn(p.clv, 1)}</td><td class="n num">${(p.kelly || 0).toFixed(2)}u</td><td>${p.result ? `<span class="pill ${p.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[p.result]}</span>` : '<span class="pill lean">Open</span>'}</td><td class="n num ${(p.units_kelly ?? 0) >= 0 ? 'pos' : 'neg'}">${p.units_kelly == null ? '—' : sgn(p.units_kelly, 2) + 'u'}</td></tr>`).join('') || '<tr><td colspan="11" class="empty">No locked plays yet. They appear as soon as the model flags one.</td></tr>'}
    </tbody></table></div></div>`;
}
function pageRecord() {
  $('#hero').innerHTML = '';
  const R = ML.record.slice().sort((a, b) => b.date.localeCompare(a.date));
  const mk = Object.entries(ML.ytd), tu = mk.reduce((s, [, x]) => s + x.units, 0);
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">Live test · out of sample</div><h2>${SEASON(ML.season)} record</h2><p>The model is never trained on this season's results before predicting them.</p></div></div>
    ${lockedPlays()}
    <div class="card"><h3>Every flagged play, re-graded</h3><p class="sub">Each game's latest pick graded at the closing line. Picks can change during the day, so this shows how the model's final view did; the locked plays above are the record of bets you could have placed.</p>
      <div class="body"><div class="kpis">${mk.map(([m, x]) => `<div><b class="num">${x.w}–${x.l}${x.p ? '–' + x.p : ''}</b><small>${m}</small><em class="${x.units >= 0 ? 'pos' : 'neg'}">${money(x.units * stake)} at $${stake}</em></div>`).join('')}<div><b class="num ${tu >= 0 ? 'pos' : 'neg'}">${money(tu * stake)}</b><small>net at $${stake}/play</small></div></div></div>
      <div class="tbl" style="max-height:70vh"><table><thead><tr><th>Date</th><th>Game</th><th>Play</th><th class="n">Odds</th><th class="n">Edge</th><th>Final</th><th>Result</th><th class="n">Profit</th></tr></thead><tbody>
      ${R.map(x => { const p = x.result === 'W' ? stake * payout(x.odds) : x.result === 'L' ? -stake : 0;
        return `<tr><td class="num muted">${x.date.slice(5)}</td><td>${esc(T(x.away).name)} @ ${esc(T(x.home).name)}</td><td><b>${esc(x.pick)}</b></td><td class="n num">${odds(x.odds)}</td><td class="n num">${x.edge}${x.market === 'Moneyline' ? '%' : ''}</td><td class="num">${x.score}</td><td><span class="pill ${x.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[x.result]}</span></td><td class="n num ${p >= 0 ? 'pos' : 'neg'}">${money(p)}</td></tr>`; }).join('') || '<tr><td colspan="8" class="empty">No graded plays yet.</td></tr>'}
      </tbody></table></div></div>${foot()}`;
}

// ---------------------------------------------------------------- performance
const sim = store.get('sim', {market: 'Total', edge: 4, from: 2015, to: 2026});
function simulate() {
  const C = Object.fromEntries(HI.cols.map((c, i) => [c, i])), out = [];
  for (const r of HI.rows) {
    const s = r[C.season]; if (s < sim.from || s > sim.to) continue;
    let e, won, push, price;
    if (sim.market === 'Spread') {
      if (r[C.home_spread] == null) continue;
      e = r[C.pred_margin] + r[C.home_spread]; const c = r[C.result] + r[C.home_spread];
      won = e > 0 ? c > 0 : c < 0; push = c === 0; price = e > 0 ? r[C.home_sp_odds] : r[C.away_sp_odds]; e = Math.abs(e);
    } else if (sim.market === 'Total') {
      if (r[C.total] == null) continue;
      e = r[C.pred_total] - r[C.total]; const d = r[C.total_pts] - r[C.total];
      won = e > 0 ? d > 0 : d < 0; push = d === 0; price = e > 0 ? r[C.over_odds] : r[C.under_odds]; e = Math.abs(e);
    } else {
      const hm = r[C.home_ml], am = r[C.away_ml]; if (hm == null || am == null || r[C.result] === 0) continue;
      const ip = o => o > 0 ? 100 / (o + 100) : -o / (-o + 100), mk = ip(hm) / (ip(hm) + ip(am)), d = r[C.p_home] - mk;
      e = Math.abs(d) * 100; won = d > 0 ? r[C.result] > 0 : r[C.result] < 0; push = false; price = d > 0 ? hm : am;
    }
    if (e < Math.max(sim.edge, .5) || price == null) continue;
    out.push({s, won, push, pnl: push ? 0 : won ? payout(price) : -1});
  }
  return out;
}
function pagePerformance() {
  $('#hero').innerHTML = '';
  const M = ML.metrics, seasons = ML.curve.map(x => x.season);
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">Walk-forward backtest · ${esc(M.test_seasons)}</div><h2>Performance</h2><p>Each season is predicted by a model trained only on earlier seasons, and compared with the closing line. ESPN's public data has no lines for 2023-24 and 2024-25, so those seasons are skipped in the betting results.</p></div></div>
    <div class="kpis"><div><b class="num">${M.margin_mae.model}</b><small>avg margin miss, model</small><em>Vegas ${M.margin_mae.vegas}</em></div><div><b class="num">${M.total_mae.model}</b><small>avg total miss, model</small><em>Vegas ${M.total_mae.vegas}</em></div>
      <div><b class="num">${pct(M.winner_acc.model, 1)}</b><small>picks the winner</small><em>Vegas ${pct(M.winner_acc.vegas, 1)}</em></div><div><b class="num">${M.brier.model}</b><small>Brier score</small><em>Vegas ${M.brier.vegas}</em></div><div><b class="num">${M.corr_with_vegas_spread}</b><small>correlation with the spread</small></div></div>
    <div class="note">Vegas is a little more accurate game by game, which is expected. The model's edge is in <b>totals</b>: at ${ML.thresholds.Total}+ points of disagreement it beat the 52.4% break-even in every test season. Spread edges were about break-even after the juice (52.5% at 7–12 points), so spreads are shown as leans, not plays; spread gaps over ${ML.max_edge.Spread} points were usually bad line data. Spread results use only lines from named sportsbooks that agree with that book's moneyline; ESPN's older consensus records sometimes swap home and away.</div>
    <div class="cols">
      <div class="card"><h3>Units by season</h3><p class="sub">Flat 1-unit bets on every play at the site's thresholds (totals ${ML.thresholds.Total}+), cumulative. The spread line shows 7–12 point spread edges for comparison; clean spread lines start in 2019.</p><div class="chart-box"><canvas id="c1"></canvas></div></div>
      <div class="card"><h3>Win rate by edge</h3><p class="sub">How often a pick won, by the size of the model's disagreement with the line. 52.4% breaks even at −110.</p><div class="chart-box"><canvas id="c2"></canvas></div></div>
    </div>
    <div class="card"><h3>Backtest by market</h3><div class="tbl"><table><thead><tr><th>Market</th><th>Min edge</th><th class="n">Bets</th><th class="n">Win %</th><th class="n">Units</th><th class="n">ROI</th></tr></thead><tbody>
      ${ML.backtest.map(x => `<tr><td>${x.market}</td><td class="num">${esc(x.edge)}</td><td class="n num">${x.bets.toLocaleString()}</td><td class="n num">${pct(x.win_pct, 1)}</td><td class="n num ${x.units >= 0 ? 'pos' : 'neg'}">${sgn(x.units, 1)}</td><td class="n num ${x.roi >= 0 ? 'pos' : 'neg'}">${pct(x.roi, 1)}</td></tr>`).join('')}</tbody></table></div></div>
    <div class="card"><h3>Strategy simulator</h3><p class="sub">Pick a market, a minimum edge and a range of seasons to see how flat $${stake} bets would have done in the backtest.</p>
      <div class="body controls"><label class="f">Market<select id="sm">${['Spread', 'Total', 'Moneyline'].map(m => `<option ${m === sim.market ? 'selected' : ''}>${m}</option>`).join('')}</select></label>
        <label class="rng">Minimum edge <input id="se" type="range" min="0" max="12" step="0.5" value="${sim.edge}"><output id="seo">${sim.edge}</output></label>
        <label class="f">From<select id="sf">${seasons.map(s => `<option ${s === sim.from ? 'selected' : ''}>${s}</option>`).join('')}</select></label>
        <label class="f">To<select id="st">${seasons.map(s => `<option ${s === sim.to ? 'selected' : ''}>${s}</option>`).join('')}</select></label></div>
      <div class="sim-out" id="so"><div class="empty">Loading every backtest game…</div></div><div class="chart-box"><canvas id="c3"></canvas></div></div>
    <div class="card"><h3>Calibration</h3><p class="sub">When the model says a team wins X% of the time, how often did it? Market probabilities shown for comparison.</p><div class="chart-box"><canvas id="c4"></canvas></div></div>
    ${foot()}`;
  killCharts();
  const mod = css('--model'), acc = css('--accent'), book = css('--book');
  chart($('#c1'), {type: 'line', data: {labels: seasons, datasets: [{label: 'Spread', data: ML.curve.map(x => x.spread_cum), borderColor: mod, backgroundColor: mod, tension: .2},
    {label: 'Total', data: ML.curve.map(x => x.total_cum), borderColor: acc, backgroundColor: acc, tension: .2}]}, options: {scales: {y: {title: {display: true, text: 'units'}}}}});
  chart($('#c2'), {type: 'line', data: {labels: ML.edge_curve.spread.map(x => x.edge), datasets: [{label: 'Spread', data: ML.edge_curve.spread.map(x => 100 * x.prob), borderColor: mod, backgroundColor: mod},
    {label: 'Total', data: ML.edge_curve.total.map(x => 100 * x.prob), borderColor: acc, backgroundColor: acc}, {label: 'Break-even', data: ML.edge_curve.spread.map(() => 52.38), borderColor: book, borderDash: [4, 4], pointRadius: 0}]},
    options: {scales: {x: {title: {display: true, text: 'model edge (points)'}}, y: {title: {display: true, text: 'win %'}}}}});
  chart($('#c4'), {type: 'scatter', data: {datasets: [{label: 'Model', data: ML.calibration.map(x => ({x: 100 * x.pred, y: 100 * x.actual})), backgroundColor: mod, pointRadius: 5},
    {label: 'Market', data: ML.calibration.map(x => ({x: 100 * x.mkt, y: 100 * x.actual})), backgroundColor: book, pointRadius: 4},
    {label: 'Perfect', type: 'line', data: [{x: 0, y: 0}, {x: 100, y: 100}], borderColor: css('--faint'), borderDash: [4, 4], pointRadius: 0}]},
    options: {scales: {x: {min: 0, max: 100, title: {display: true, text: 'predicted win %'}}, y: {min: 0, max: 100, title: {display: true, text: 'actual win %'}}}}});
  let c3 = null;
  const run = () => {
    if (!HI) return;
    const r = simulate(), n = r.length, nn = r.filter(x => !x.push).length, w = r.filter(x => x.won && !x.push).length, u = r.reduce((s, x) => s + x.pnl, 0);
    $('#so').innerHTML = [[n.toLocaleString(), 'bets'], [pct(w / (nn || 1), 1), 'win rate'], [`<span class="${u >= 0 ? 'pos' : 'neg'}">${money(u * stake)}</span>`, `profit at $${stake}/bet`], [pct(u / (n || 1), 1), 'ROI']]
      .map(([b, s]) => `<div><b class="num">${b}</b><small>${s}</small></div>`).join('');
    const byS = seasons.filter(s => s >= sim.from && s <= sim.to).map(s => ({s, u: r.filter(x => x.s === s).reduce((a, x) => a + x.pnl, 0)}));
    let c = 0; const cum = byS.map(x => (c += x.u * stake));
    if (c3) { c3.destroy(); charts.splice(charts.indexOf(c3), 1); }
    c3 = chart($('#c3'), {type: 'bar', data: {labels: byS.map(x => x.s), datasets: [{label: 'Profit by season', data: byS.map(x => x.u * stake), backgroundColor: byS.map(x => x.u >= 0 ? css('--pos') : css('--neg'))},
      {label: 'Cumulative', type: 'line', data: cum, borderColor: mod, backgroundColor: mod}]}, options: {scales: {y: {title: {display: true, text: '$'}}}}});
  };
  const upd = () => { sim.market = $('#sm').value; sim.edge = +$('#se').value; sim.from = +$('#sf').value; sim.to = +$('#st').value; $('#seo').textContent = sim.edge; store.set('sim', sim); run(); };
  ['#sm', '#sf', '#st'].forEach(s => $(s).onchange = upd); $('#se').oninput = upd;
  if (HI) run(); else fetch('data/history.json').then(r => r.json()).then(d => { HI = d; run(); }).catch(() => { $('#so').innerHTML = '<div class="empty">Could not load the backtest games.</div>'; });
}

// ---------------------------------------------------------------- about
function pageAbout() {
  $('#hero').innerHTML = '';
  const imp = Object.entries(ML.importance), mx = Math.max(...imp.map(x => x[1]));
  const lbl = {exp_pts: 'Expected points (ratings × tempo)', exp_eff: 'Expected efficiency', exp_poss: 'Expected possessions', adj_o: 'Own adj. offense', adj_d: 'Own adj. defense',
    opp_adj_o: 'Opponent adj. offense', opp_adj_d: 'Opponent adj. defense', adj_t: 'Own tempo', opp_adj_t: 'Opponent tempo', elo_diff: 'Elo difference', loc: 'Home / away / neutral',
    mu: 'League scoring level', mu_t: 'League tempo', day: 'Point in the season', games: 'Games played', opp_games: 'Opponent games played', rest: 'Rest days', opp_rest: 'Opponent rest'};
  $('#app').innerHTML = `<div class="head"><div><h2>How the model works</h2></div></div>
    <div class="cols"><div class="card"><div class="body prose">
      <p><b>Data.</b> Every Division I game since 2011-12 from ESPN via the sportsdataverse project: schedules, team box scores and the pregame betting lines ESPN publishes for each game.</p>
      <p><b>Adjusted efficiency.</b> For every game day, each team's offense and defense (points per 100 possessions) are solved together with home-court advantage as a weighted ridge regression over the games played so far, so beating good teams counts for more than beating bad ones. Recent games weigh more, and each team starts the season anchored to 60% of last season's rating. Tempo is rated the same way.</p>
      <p><b>Four factors.</b> Effective FG%, turnover rate, offensive rebound rate and free-throw rate, for and against, plus 3-point rate and FT%, exponentially weighted toward recent games.</p>
      <p><b>Model.</b> Ridge regression and LightGBM, blended 60/40, each predict a team's points from its ratings against the opponent's, both tempos, the four factors, Elo, venue, rest and point in the season. A logistic layer turns the projected margin into win probability.</p>
      <p><b>Testing.</b> Walk-forward by season: each year is predicted by a model trained only on earlier years and compared with the closing line.</p>
      <p><b>Edges.</b> The gap between the model and a line becomes a win probability using how often gaps that size actually won in the backtest; that sets expected value and a quarter-Kelly stake. Totals are de-biased against the market's running total so the model can't drift into betting one side.</p>
      <p><b>Locked plays.</b> Each play is recorded once, at the best price when it's first flagged, and graded at that price, with closing line value tracked.</p></div></div>
      <div class="card"><h3>What drives the model</h3><p class="sub">Share of the tree model's total gain, top 20 inputs.</p><div class="body imp">
        ${imp.map(([k, v]) => `<div><span>${esc(lbl[k] || k.replace(/^oppd_/, 'opp. defense ').replace(/^oppo_/, 'opp. offense ').replace(/^o_/, 'offense ').replace(/^d_/, 'defense ').replace('efg', 'eFG%').replace('tov', 'TO rate').replace('orb', 'OR%').replace('ftr', 'FT rate').replace('tpr', '3PA rate').replace('tpp', '3P%').replace('ftp', 'FT%'))}</span><span class="b" style="width:${100 * v / mx}%"></span><span class="num muted">${pct(v, 1)}</span></div>`).join('')}</div></div></div>
    ${foot()}`;
}

// ---------------------------------------------------------------- router
function route() {
  killCharts();
  const h = location.hash.replace(/^#\/?/, '').split('/'), r = h[0] || '';
  document.querySelectorAll('.nav a').forEach(a => a.toggleAttribute('aria-current', a.getAttribute('href') === '#/' + r || (r === 'day' && a.getAttribute('href') === '#/') || (r === 'game' && a.getAttribute('href') === '#/')));
  document.querySelectorAll('.nav a[aria-current]').forEach(a => a.setAttribute('aria-current', 'page'));
  ({'': () => pageDay(), day: () => pageDay(h[1]), game: () => pageGame(h[1]), rankings: pageRankings, lines: pageLines, record: pageRecord, performance: pagePerformance, about: pageAbout}[r] || (() => pageDay()))();
  if (!route.keep) scrollTo(0, 0);
  route.keep = false;
}
addEventListener('hashchange', route);
const picksReq = fetch('data/picks.json').then(r => r.ok ? r.json() : null).catch(() => null);
Promise.all(['ml', 'rankings'].map(f => fetch(`data/${f}.json`).then(r => { if (!r.ok) throw new Error(f); return r.json(); })).concat(picksReq))
  .then(([a, b, c]) => { ML = a; RK = b; PK = c; RK.teams.forEach(t => RANK[t.team] = t); route(); })
  .catch(e => { $('#app').innerHTML = `<div class="card empty">Couldn't load the model data (${esc(e.message)}). Try refreshing.</div>`; });
})();
