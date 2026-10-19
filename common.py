"""Shared configuration, data loading and cached risk calculations.

Every page imports this module, so the carbon benchmark, the commodity list and the
core risk numbers (volatility, VaR, GARCH, regimes) are defined and computed in one
place. The heavy calculations are wrapped in st.cache_data, so a page that needs a
number another page already computed gets it from the cache.
"""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from arch import arch_model
from scipy.stats import chi2

# ─── Carbon Benchmark Resolution ───
# KEUA (KraneShares European Carbon Allowance ETF) was liquidated on 20 March 2026,
# so the original EUA proxy no longer returns data. We probe a cascade of candidates
# at runtime and label the dashboard from whichever one actually resolves, so the
# stated benchmark can never drift from the series being plotted.
CARBON_CANDIDATES = [
    {
        "ticker": "CARB.L",
        "label": "EU Carbon ETC (CARB.L, USD)",
        "short": "EU Carbon",
        "name": "EU Carbon (EUA ETC, USD)",
        "unit": "ETC price, USD (tracks EUA futures)",
        "pure_eu": True,
        "note": ("WisdomTree Carbon ETC, USD line on the LSE (the GBP line is CARP). It tracks ICE "
                 "EUA futures, but the quote is an ETC share price in USD, not €/tCO2, so its "
                 "returns also carry EUR/USD moves, roll yield and fees."),
    },
    {
        "ticker": "KRBN",
        "label": "Global Carbon (KRBN, EUA-weighted)",
        "short": "Carbon",
        "name": "Carbon Allowances (Global)",
        "unit": "ETF price, USD",
        "pure_eu": False,
        "note": ("KraneShares Global Carbon Strategy ETF. EUA carries the dominant index weight, "
                 "but California (CCA), RGGI, UK (UKA) and Washington (WCA) allowances are also "
                 "included: a correlated proxy for EU carbon, not a pure EUA price."),
    },
    {
        "ticker": "ICLN",
        "label": "Clean Energy Proxy (ICLN)",
        "short": "Clean Energy",
        "name": "Clean Energy Proxy",
        "unit": "ETF price, USD",
        "pure_eu": False,
        "note": ("Clean-energy equity ETF. This is NOT a carbon allowance price; it is a "
                 "last-resort proxy used only when no carbon instrument resolves."),
    },
]



@st.cache_data(ttl=3600, show_spinner="Resolving carbon benchmark...")
def resolve_carbon_benchmark():
    """Return the first carbon candidate that yfinance can actually serve."""
    probe_start = (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    for cand in CARBON_CANDIDATES:
        try:
            probe = yf.download(cand["ticker"], start=probe_start, progress=False)
            if probe is None or probe.empty:
                continue
            if probe["Close"].squeeze().notna().sum() > 30:
                return cand
        except Exception:
            continue
    return CARBON_CANDIDATES[-1]



def build_commodities(carbon):
    """Commodity definitions. The carbon entry follows whichever benchmark resolved."""
    return {
        "TTF Natural Gas": {
            "ticker": "TTF=F",
            "unit": "€/MWh",
            "color": "steelblue",
            "keywords": ['gas', 'TTF', 'LNG', 'pipeline', 'natural gas', 'methane'],
            "rss_query": "natural+gas+Europe+price",
            "news_term": "gas",   # one word for country and 30-day news queries
        },
        "WTI Crude Oil": {
            "ticker": "CL=F",
            "unit": "$/barrel",
            "color": "saddlebrown",
            "keywords": ['oil', 'crude', 'WTI', 'OPEC', 'petroleum', 'barrel', 'refinery'],
            "rss_query": "crude+oil+Europe+price",
            "news_term": "oil",   # one word for country and 30-day news queries
        },
        "Brent Crude Oil": {
            "ticker": "BZ=F",
            "unit": "$/barrel",
            "color": "darkred",
            "keywords": ['oil', 'crude', 'Brent', 'OPEC', 'petroleum', 'barrel', 'North Sea'],
            "rss_query": "brent+oil+Europe+price",
            "news_term": "oil",   # one word for country and 30-day news queries
        },
        carbon["name"]: {
            "ticker": carbon["ticker"],
            "unit": carbon["unit"],
            "color": "seagreen",
            "keywords": ['carbon', 'ETS', 'emission', 'EU ETS', 'EUA', 'allowance', 'CBAM'],
            "rss_query": "EU+carbon+ETS+emission+price",
            "news_term": "carbon",   # one word for country and 30-day news queries
        },
    }


DEFAULT_START = dt.date(2020, 1, 1)


def context():
    """Everything a page needs to know about the current selection.

    The commodity selector and date inputs live in app.py (keys: commodity,
    start_date, end_date), so the choice carries across pages.
    """
    carbon = resolve_carbon_benchmark()
    commodities = build_commodities(carbon)
    name = st.session_state.get("commodity")
    if name not in commodities:
        name = next(iter(commodities))
    commodity = commodities[name]
    is_carbon = commodity["ticker"] == carbon["ticker"]
    return SimpleNamespace(
        carbon=carbon,
        commodities=commodities,
        selected_commodity=name,
        commodity=commodity,
        is_carbon=is_carbon,
        compare_ticker="TTF=F" if is_carbon else carbon["ticker"],
        compare_label="TTF Natural Gas (€/MWh)" if is_carbon else carbon["label"],
        start_date=st.session_state.get("start_date", DEFAULT_START),
        end_date=st.session_state.get("end_date", dt.date.today()),
    )


# ─── Data Download (cached) ───
STALE_RUN = 5   # this many identical consecutive closes = a stale feed (suspension, no trading)


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def _raw_close(ticker, start, end):
    """Daily closing prices for one ticker exactly as downloaded (empty if unavailable).

    yfinance treats `end` as exclusive, so one day is added: the end date chosen in the
    sidebar is included. On a trading day the last value is the latest price, not yet a
    settlement.
    """
    try:
        data = yf.download(ticker, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                           progress=False)
    except Exception:
        return pd.Series(dtype=float)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    # Yahoo sometimes adds a Saturday or Sunday row with an intraday price from the Sunday
    # evening session (e.g. CL=F, BZ=F). It is not a settlement, so weekend rows are dropped.
    close = close[close.index.dayofweek < 5]
    return close.astype(float)


def _stale_mask(close):
    """True on the repeated days of any run of STALE_RUN or more identical closes.

    The first day of a run is a real traded price and is kept; the repeats after it are
    not market data (e.g. CARB.L was suspended from June 2020 to August 2021 and the
    feed repeated the last price).
    """
    if close.empty:
        return pd.Series(False, index=close.index)
    same = close.eq(close.shift())
    run_id = (~same).cumsum()
    run_len = close.groupby(run_id).transform('size')
    return same & (run_len >= STALE_RUN)


@st.cache_data(ttl=3600)
def stale_periods(ticker, start, end):
    """(first, last) dates of every excluded stale run in a ticker's series."""
    close = _raw_close(ticker, start, end)
    mask = _stale_mask(close)
    if not mask.any():
        return []
    run_id = (~mask).cumsum()[mask]
    return [(g.index[0].date(), g.index[-1].date()) for _, g in run_id.groupby(run_id)]


def load_close(ticker, start, end):
    """Daily closes with stale repeats set to NaN.

    The dates stay in the index, so a return computed with pct_change(fill_method=None)
    is NaN across the gap instead of one fake return spanning the whole suspension.
    """
    close = _raw_close(ticker, start, end)
    return close.mask(_stale_mask(close))


def break_gaps(series, max_gap_days=7):
    """For line charts: insert a NaN wherever two observations are more than `max_gap_days`
    apart, so the chart shows the gap (e.g. CARB.L's stale 2020-21 period) instead of a
    straight line drawn across it. Weekends and single holidays are left joined."""
    s = series.dropna()
    if len(s) < 2:
        return s
    gaps = s.index.to_series().diff() > pd.Timedelta(days=max_gap_days)
    if not gaps.any():
        return s
    breaks = pd.Series(np.nan, index=s.index[gaps.values] - pd.Timedelta(days=1))
    return pd.concat([s, breaks]).sort_index()


def simple_returns(close):
    """Daily simple returns on the series' own trading days; NaN after a gap or a
    non-positive price (WTI settled negative on 2020-04-20)."""
    return close.where(close > 0).pct_change(fill_method=None)


# ─── Core Calculations ───
VAR_WINDOW = 250     # trading days behind the headline VaR and Expected Shortfall


def _var_es(returns):
    """Historical VaR 95%, VaR 99% and ES 97.5% of daily returns, in % (losses negative).

    Quantiles use numpy's default linear interpolation between the two nearest days. ES is
    the mean of the returns at or below the 2.5% quantile (7 of 250 days, about 2.8%).
    """
    q95, q99, q975 = np.percentile(returns, [5, 1, 2.5])
    return q95 * 100, q99 * 100, returns[returns <= q975].mean() * 100


@st.cache_data(ttl=3600, show_spinner="Computing risk metrics...")
def compute_core(ticker, compare_ticker, start, end):
    """Returns, rolling volatility/correlation, VaR, the current regime and risk signal.

    Returns None when fewer than 30 overlapping observations are available.
    """
    price = load_close(ticker, start, end)
    compare = load_close(compare_ticker, start, end)
    df = pd.DataFrame({
        'Price': price,
        'Compare': compare,
        # Returns on each series' own trading days, so a stale gap or a non-positive
        # price gives NaN instead of a return that spans the gap.
        'Returns': simple_returns(price),
        'Compare_Returns': simple_returns(compare),
    }).dropna(subset=['Price'])
    has_compare = df['Compare'].notna().sum() > 30

    # Percentage returns are undefined across a sign change in the price level.
    # WTI (CL=F) settled negative on 2020-04-20, so non-positive prices are excluded.
    # Only the commodity's own price decides which days are kept: a gap in the comparison
    # series (CARB.L was stale from June 2020 to August 2021) must not remove days from the
    # commodity's volatility and VaR. The comparison is simply missing on those days.
    df_analysis = df[df['Price'] > 0].copy()
    df_analysis['Compare'] = df_analysis['Compare'].where(df_analysis['Compare'] > 0)
    if len(df_analysis) < 30:
        return None

    df_analysis['Volatility'] = df_analysis['Returns'].rolling(30, min_periods=20).std() * 100

    # Rolling correlation on returns (not price levels) to avoid spurious correlation.
    # min_periods=20, as for volatility: without it one missing day in either series (e.g. a
    # London holiday for the carbon ETC) blanks the next 30 days of the line. pandas uses the
    # days on which both returns exist.
    df_analysis['Rolling Correlation'] = (
        df_analysis['Returns'].rolling(30, min_periods=20).corr(df_analysis['Compare_Returns'])
    )

    latest_vol = df_analysis['Volatility'].dropna().iloc[-1]
    avg_vol = df_analysis['Volatility'].dropna().mean()

    returns_clean = df_analysis['Returns'].dropna()
    # Headline VaR / ES from the last VAR_WINDOW trading days; full-period figures alongside.
    var_returns = returns_clean.tail(VAR_WINDOW)
    var_95, var_99, es_975 = _var_es(var_returns)
    var_95_full, var_99_full, es_975_full = _var_es(returns_clean)

    returns_corr = df_analysis[['Returns', 'Compare_Returns']].dropna()
    overall_corr = returns_corr['Returns'].corr(returns_corr['Compare_Returns'])

    regime_thr = regime_thresholds(ticker, df_analysis['Volatility'])
    current_regime = regime_label(latest_vol, regime_thr)

    return dict(
        df_analysis=df_analysis, has_compare=has_compare,
        latest_vol=latest_vol, avg_vol=avg_vol,
        returns_clean=returns_clean, var_95=var_95, var_99=var_99, es_975=es_975,
        var_days=len(var_returns), var_95_full=var_95_full, var_99_full=var_99_full,
        es_975_full=es_975_full,
        overall_corr=overall_corr,
        regime_thr=regime_thr, current_regime=current_regime,
    )


# ─── GARCH(1,1) ───
@st.cache_data(ttl=3600, show_spinner="Fitting GARCH model...")
def fit_garch(returns_clean):
    """Fit GARCH(1,1) on daily % returns and forecast 10 days ahead.

    Returns None if the fit fails (e.g. too short a window).
    """
    garch_returns = returns_clean.dropna() * 100
    try:
        # Student-t shocks: daily energy returns are fat-tailed, a normal distribution
        # understates how often large moves happen.
        model = arch_model(garch_returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
        result = model.fit(disp='off')
        forecast = result.forecast(horizon=10)
    except Exception:
        return None
    forecast_vol = np.sqrt(forecast.variance.iloc[-1])

    # ── Bootstrap 90% CI for GARCH(1,1) 10-day forecast ─────────────────────
    # Resample standardised residuals (ẑ_t = ε_t / σ_t) from the fitted model.
    # For each draw, propagate the GARCH recursion forward 10 steps to obtain a
    # distribution of forecast volatilities; take the 5th / 95th percentiles.
    _N_BOOT   = 500
    _std_z    = (result.resid / result.conditional_volatility).dropna().values
    _omega_b  = result.params['omega']
    _alpha_b  = result.params['alpha[1]']
    _beta_b   = result.params['beta[1]']
    _s2_init  = float(result.conditional_volatility.iloc[-1] ** 2)
    _e2_init  = float(result.resid.iloc[-1] ** 2)   # last shock, net of the fitted mean

    _boot_vols = np.zeros((_N_BOOT, 10))
    _rng = np.random.default_rng(42)
    for _b in range(_N_BOOT):
        _z  = _rng.choice(_std_z, size=10, replace=True)
        _s2 = _s2_init
        _e2 = _e2_init
        for _h in range(10):
            _s2 = _omega_b + _alpha_b * _e2 + _beta_b * _s2
            _boot_vols[_b, _h] = np.sqrt(max(_s2, 1e-8))  # guard against numerical zero
            _e2 = _s2 * _z[_h] ** 2
    # ─────────────────────────────────────────────────────────────────────────

    # Long-run (unconditional) volatility the forecast reverts to: sqrt(ω / (1 − α − β)).
    # Undefined when α + β ≥ 1, since shocks then never fade.
    persistence = _alpha_b + _beta_b
    garch_long_run = float(np.sqrt(_omega_b / (1 - persistence))) if persistence < 1 else None
    # With α + β close to 1 (TTF, Brent and carbon all sit at 0.98 to 0.99) the formula divides
    # by almost zero: small changes in the fit move it a lot, and it can flip between sensible
    # and absurd from one day to the next. So the reference level for comparisons is always the
    # sample average of the 30-day rolling volatility; the GARCH value is shown for information
    # only, and only when it is stable (α + β < 0.98 and within 1.5x the sample average).
    sample_avg_vol = float((returns_clean.dropna().rolling(30).std() * 100).mean())
    garch_long_run_ok = (garch_long_run is not None and persistence < 0.98
                         and garch_long_run <= 1.5 * sample_avg_vol)

    return dict(
        params={k: float(result.params[k]) for k in ('omega', 'alpha[1]', 'beta[1]')},
        long_run_vol=sample_avg_vol,         # pages display core['avg_vol'] as the reference
        garch_long_run=garch_long_run,       # √(ω / (1 − α − β)), information only
        garch_long_run_ok=garch_long_run_ok,
        persistence=float(persistence),
        sample_avg_vol=sample_avg_vol,
        nu=float(result.params['nu']) if 'nu' in result.params else None,
        loglikelihood=float(result.loglikelihood),
        conditional_volatility=result.conditional_volatility,
        # conditional_volatility is already a volatility series, no need for sqrt(x**2)
        current_cond_vol=float(result.conditional_volatility.iloc[-1]),
        forecast_vol=forecast_vol.values,
        # Daily volatility forecast for day 5 and day 10 ahead (not holding-period volatility).
        forecast_5d=float(forecast_vol.iloc[4]),
        forecast_10d=float(forecast_vol.iloc[9]),
        # Volatility of the 10-day holding-period return: sqrt of the sum of the 10 daily
        # forecast variances (returns are close to uncorrelated from day to day).
        forecast_cum_10d=float(np.sqrt(forecast.variance.iloc[-1].iloc[:10].sum())),
        ci_lo=np.percentile(_boot_vols, 5, axis=0),
        ci_hi=np.percentile(_boot_vols, 95, axis=0),
    )


# ─── Regime Detection (per-commodity volatility thresholds) ───
# Thresholds come from the last REGIME_WINDOW trading days only (about 3 years). The full
# history since 2010 includes the 2021-22 energy crisis, when TTF's daily volatility was
# several times its normal level; percentiles taken over it set the "normal" line so high
# that a market under real stress (e.g. 2026, with Qatari LNG disrupted) was still labelled
# Normal. A recent window compares today with the market as it has recently been.
REGIME_WINDOW = 750               # trading days, about 3 years
# Elevated = top 25%, Extreme = top 10% of the commodity's own 30-day volatility. There is no
# industry-standard regime cut-off; desks usually quote where volatility sits in its own history
# (its percentile), which the page shows next to the label.
ELEVATED_PCT, EXTREME_PCT = 75, 90
_MIN_HISTORY = 250
REGIME_COLORS = {'Normal': 'green', 'Elevated': 'orange', 'Extreme': 'red'}
ANNUAL = np.sqrt(252)   # daily volatility x sqrt(252) = annualised volatility


@st.cache_data(ttl=86400, show_spinner="Computing regime thresholds...")
def regime_thresholds(ticker, fallback_vol):
    """Elevated / Extreme thresholds from the commodity's own recent 30-day volatility.

    Elevated starts at the ELEVATED_PCT-th percentile and Extreme at the EXTREME_PCT-th, both
    taken over the last REGIME_WINDOW trading days up to today (so they do not move with
    the selected date range). If that history is too short, `fallback_vol` (the 30-day
    volatility of the selected range) is used instead. The mean and standard deviation
    of the same window are returned for the volatility anomaly check.
    """
    start = dt.date.today() - dt.timedelta(days=int(REGIME_WINDOW * 365 / 250) + 90)
    close = load_close(ticker, start, dt.date.today())
    vol = (simple_returns(close).rolling(30, min_periods=20).std() * 100).dropna().tail(REGIME_WINDOW)
    source = f"last {len(vol):,} trading days"
    if len(vol) < _MIN_HISTORY:
        vol, source = fallback_vol.dropna(), "selected date range only; recent history unavailable"
    elevated, extreme = np.percentile(vol, [ELEVATED_PCT, EXTREME_PCT])
    return dict(elevated=float(elevated), extreme=float(extreme), source=source, hist=np.sort(vol.values),
                start=vol.index[0], end=vol.index[-1], n=len(vol),
                mean=float(vol.mean()), std=float(vol.std()))


# Fixed pre-war reference: the last full quarter before the US-Israeli strikes on Iran on 28 Feb 2026.
# A fixed window, like the fixed event dates of the historical scenarios, so the reference does not
# drift with the date range; percentile windows alone cannot say "twice as volatile as before the war".
PREWAR_START, PREWAR_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")
_CONTEXT_START = dt.date(2017, 1, 1)


@st.cache_data(ttl=3600, show_spinner=False)
def vol_context(ticker, as_of, vol_now):
    """Where `vol_now` (30-day volatility of daily returns, %) sits against the commodity's own history
    up to `as_of`: percentile rank over the last 250 and 750 trading days and since 2017, and the ratio
    to the median 30-day volatility of the pre-war quarter (Oct to Dec 2025). Same volatility definition
    as the page (simple returns, 30-day window, stale repeats excluded)."""
    close = load_close(ticker, _CONTEXT_START, dt.date.today())
    vol = (simple_returns(close).rolling(30, min_periods=20).std() * 100).dropna()
    vol = vol[vol.index <= pd.Timestamp(as_of)]
    if len(vol) < 250:
        return None

    def rank(window):
        h = np.sort(window.values)
        return 100.0 * np.searchsorted(h, vol_now, side='left') / len(h)

    prewar = vol[(vol.index >= PREWAR_START) & (vol.index <= PREWAR_END)]
    pre_med = float(prewar.median()) if len(prewar) >= 20 else None
    return dict(p1y=rank(vol.tail(250)), p3y=rank(vol.tail(750)), p_all=rank(vol), all_start=vol.index[0],
                start_3y=vol.tail(750).index[0],
                prewar_median=pre_med, prewar_ratio=(vol_now / pre_med) if pre_med else None)


def fmt_vol(daily, decimals=0):
    """A daily volatility (%) written the one way the pages use: annualised first, daily in brackets."""
    return f"{daily * ANNUAL:.{decimals}f}% ann. ({daily:.2f}% daily)"


def _ordinal(x):
    n = int(round(x))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


_WAR_START = pd.Timestamp("2026-03-01")
_PRICE_TOL = 1e-4   # a continuous close and a single-contract close from the same download are the same quote


@st.cache_data(ttl=3600, show_spinner="Checking contract rolls...")
def window_rolls(ticker, window_returns):
    """Contract rolls of a continuous front-month future (TTF=F, CL=F, BZ=F) inside a volatility window.

    `window_returns` are the daily returns behind the latest 30-day volatility. The continuous series and the
    individual monthly contracts are downloaded together, and each day is matched to the one contract whose
    close equals the continuous close. A return day is a roll day when the continuous close equals contract
    M on that day but did not equal M's close on the day before; its return then contains the gap between two
    contracts, and M's own return that day replaces it. A day is not identified when the continuous close
    matches no single contract on Yahoo Finance (expired contracts are no longer listed) or M has no close on
    the day before; nothing is guessed for those days. For Brent the ICE rule (brent_roll_days) is the page's
    roll calendar, so an unmatched Brent day counts as a roll day only if the rule makes it one.

    Returns None for a ticker without monthly contracts, else dict(rolls, unknown, rule_only, adjusted_vol):
    rolls = [(date, new contract symbol, continuous return, new contract return)], unknown = unidentified
    return days, rule_only = Brent rule days with no contract price to adjust, adjusted_vol = 30-day
    volatility (daily, %) with the new contract's return on the roll days (None if there is no roll)."""
    if ticker not in CURVE_ROOTS:
        return None
    r = window_returns.dropna()
    if r.empty:
        return None
    root = CURVE_ROOTS[ticker][0]
    first, last = r.index[0], r.index[-1]
    months = pd.period_range(first.to_period('M'), last.to_period('M') + 4, freq='M')
    syms = [f"{root}{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM" for p in months]
    try:
        data = yf.download([ticker] + syms, start=first - pd.Timedelta(days=14),
                           end=last + pd.Timedelta(days=1), progress=False)['Close']
    except Exception:
        data = pd.DataFrame()
    if isinstance(data, pd.Series):
        data = data.to_frame(ticker)
    data = data[data.index.dayofweek < 5] if len(data) else data
    cont = data[ticker].dropna() if ticker in data else pd.Series(dtype=float)
    syms = [s for s in syms if s in data and data[s].notna().any()]

    def match(day):
        hits = [s for s in syms if pd.notna(data.at[day, s]) and abs(data.at[day, s] - cont[day]) < _PRICE_TOL]
        return hits[0] if len(hits) == 1 else None

    rule = set(brent_roll_days(cont.index)) if ticker == "BZ=F" and len(cont) else set()
    rolls, unknown, rule_only = [], [], []
    adj = r.copy()
    for t in r.index:
        pos = cont.index.get_indexer([t])[0]
        m = match(t) if pos > 0 else None
        if m is not None:
            p = cont.index[pos - 1]
            if match(p) == m:
                continue                                  # same contract on both days: no roll
            m_prev = data.at[p, m]
            if pd.notna(m_prev) and abs(m_prev - cont[p]) >= _PRICE_TOL:
                new_ret = data.at[t, m] / m_prev - 1
                rolls.append((t, m, float(r[t]), float(new_ret)))
                adj[t] = new_ret
                continue
        if ticker == "BZ=F":
            if t in rule:
                rule_only.append(t)
            continue                                      # Brent: not a rule day, so not a roll day
        unknown.append(t)
    return dict(rolls=rolls, unknown=unknown, rule_only=rule_only,
                adjusted_vol=float(adj.std() * 100) if rolls else None)


def _contract_month(sym, root):
    code = sym[len(root)]
    return f"{dt.date(2000 + int(sym[len(root) + 1:len(root) + 3]), MONTH_CODES.index(code) + 1, 1):%b %y}"


def _day_ranges(days, index):
    """Consecutive runs of `days` within `index`, written '08 Sep to 22 Sep 2026' or '19 Oct 2026'."""
    pos = sorted(index.get_indexer(days))
    runs, start = [], pos[0]
    for a, b in zip(pos, pos[1:] + [None]):
        if b != a + 1:
            runs.append((index[start], index[a]))
            start = b
    return ", ".join(f"{a:%d %b %Y}" if a == b else
                     (f"{a:%d %b} to {b:%d %b %Y}" if a.year == b.year else f"{a:%d %b %Y} to {b:%d %b %Y}")
                     for a, b in runs)


def roll_sentence(ticker, window_returns):
    """The line under the volatility heading when the 30-day window contains contract roll days, or ''."""
    rc = window_rolls(ticker, window_returns)
    if rc is None or not (rc['rolls'] or rc['unknown'] or rc['rule_only']):
        return ""
    root = CURVE_ROOTS[ticker][0]
    idx = window_returns.dropna().index
    parts = []
    if rc['rolls']:
        days = [f"{t:%d %b %Y} ({ticker} {cr * 100:+.2f}%, the {_contract_month(m, root)} contract it switched to "
                f"{nr * 100:+.2f}%)" for t, m, cr, nr in rc['rolls']]
        parts.append(f"Contract roll days in this 30-day window: {' and '.join(days)}. On a roll day the "
                     "series' return includes the price gap between two contracts. With the new contract's own "
                     f"return on {'that day' if len(days) == 1 else 'those days'}, the 30-day volatility is "
                     f"{rc['adjusted_vol'] * ANNUAL:.0f}% annualised ({rc['adjusted_vol']:.2f}% daily).")
    if rc['rule_only']:
        parts.append(f"{_day_ranges(rc['rule_only'], idx)} {'is a roll day' if len(rc['rule_only']) == 1 else 'are roll days'} "
                     "by the ICE Brent rule, but the series' close matches no single monthly contract on Yahoo "
                     "Finance, so no adjusted figure is given.")
    if rc['unknown']:
        parts.append(f"Roll days not identified for {_day_ranges(rc['unknown'], idx)}: the series' close matches "
                     "no single monthly contract on Yahoo Finance (expired contracts are no longer listed), so a "
                     "roll on those days is neither ruled out nor adjusted.")
    if rc['rolls']:
        parts.append("The 3-year label and the percentiles use the unadjusted figure.")
    return " ".join(parts)


def vol_summary(ticker, as_of, vol_now, name=None, window_returns=None):
    """The single source of every volatility statement on the Risk and Market pages and in the AI input.

    Returns dict(headline, headline2, roll, detail, label, tile, regime, ...): `headline` and `headline2` are
    the two lines of the large neutral heading, `roll` the line on contract roll days in the 30-day window
    (only when `window_returns`, the returns behind `vol_now`, are given), `detail` the paragraph under it,
    `label` the small grey line with the 3-year label, `tile` the value for a metric tile."""
    ctx = vol_context(ticker, as_of, vol_now)
    close = load_close(ticker, _CONTEXT_START, dt.date.today())
    hist = (simple_returns(close).rolling(30, min_periods=20).std() * 100).dropna()
    thr = regime_thresholds(ticker, hist)
    regime = regime_label(vol_now, thr)
    out = dict(regime=regime, thr=thr, ctx=ctx, tile=fmt_vol(vol_now, 0),
               roll=roll_sentence(ticker, window_returns) if window_returns is not None else "",
               headline=f"{name + ' ' if name else ''}30-day volatility: {vol_now * ANNUAL:.0f}% annualised "
                        f"({vol_now:.2f}% daily)",
               label=(f"3-year label: {regime} (Elevated above the {ELEVATED_PCT}th percentile, "
                      f"{fmt_vol(thr['elevated'])}; Extreme above the {EXTREME_PCT}th, {fmt_vol(thr['extreme'])})."))
    if ctx is None:
        out.update(headline2="", detail="")
        return out
    r = ctx['prewar_ratio']
    if r is None:
        compare = "no pre-war reference"
    elif r >= 1.1:
        compare = f"{r:.2f}x its pre-war level"
    elif r >= 0.9:
        compare = f"close to its pre-war level ({r:.2f}x)"
    else:
        compare = f"below its pre-war level ({r:.2f}x)"
    out['headline2'] = f"{compare} · {_ordinal(ctx['p3y'])} percentile of the last 3 years"
    detail = ""
    if ctx['prewar_median'] is not None:
        detail += (f"Pre-war level: {ctx['prewar_median'] * ANNUAL:.0f}% annualised ({ctx['prewar_median']:.2f}% "
                   "daily), the median of Oct to Dec 2025, the last full quarter before the strikes on Iran on "
                   "28 Feb 2026. ")
    detail += (f"Percentiles: {_ordinal(ctx['p1y'])} over the last year, {_ordinal(ctx['p3y'])} over 3 years, "
               f"{_ordinal(ctx['p_all'])} since {ctx['all_start']:%b %Y}.")
    if ctx['start_3y'] <= _WAR_START:
        detail += " The 3-year window includes the volatility after the war began in March 2026."
    out['detail'] = detail
    return out


def regime_label(vol, thr):
    """Regime for one volatility value (%): a higher volatility never gets a lower regime."""
    if vol > thr['extreme']:
        return 'Extreme'
    if vol > thr['elevated']:
        return 'Elevated'
    return 'Normal'


def vol_percentile(vol, thr):
    """Share of the threshold window's 30-day volatilities below `vol`, in % (its percentile rank)."""
    return 100.0 * np.searchsorted(thr['hist'], vol, side='left') / len(thr['hist'])


def regime_caption(thr):
    """One-line description of the thresholds, for page captions."""
    period = f"{thr['start']:%b %Y} to {thr['end']:%b %Y}"
    return (f"Thresholds for this commodity: Normal up to {fmt_vol(thr['elevated'])}, Elevated up to "
            f"{fmt_vol(thr['extreme'])}, Extreme above that (the {ELEVATED_PCT}th and {EXTREME_PCT}th "
            f"percentiles of its own 30-day rolling volatility, {period}, {thr['source']}). It compares volatility "
            "with the commodity's own last 3 years; it is not a measure of overall risk.")


def compute_regimes(features, thr):
    """Label each day Normal / Elevated / Extreme from its 30-day volatility alone.

    `features` has a Volatility column (other columns are kept for the statistics);
    `thr` comes from regime_thresholds().
    """
    features = features.dropna(subset=['Volatility']).copy()
    features['Regime'] = features['Volatility'].apply(regime_label, thr=thr)
    return features


# ─── Cross-commodity returns (correlation matrix, portfolio VaR, positions) ───
@st.cache_data(ttl=3600, show_spinner="Computing cross-commodity returns...")
def returns_panel(start, end, carbon_short, carbon_ticker):
    """Daily returns of TTF, WTI, Brent and carbon, one column each (not NaN-aligned).

    Callers drop NaNs over the columns they use, so a short carbon history does not
    truncate a calculation that does not involve carbon.
    """
    tickers = {"TTF Gas": "TTF=F", "WTI Oil": "CL=F", "Brent Oil": "BZ=F", carbon_short: carbon_ticker}
    returns = {}
    for name, ticker in tickers.items():
        close = load_close(ticker, start, end)
        if len(close) > 30:
            # WTI (CL=F) settled negative on 2020-04-20; percentage returns are
            # undefined across a sign change, so those observations are excluded.
            returns[name] = simple_returns(close)
    if len(returns) < 2:
        return None
    return pd.DataFrame(returns)


# ─── Position risk: € VaR / ES, VaR backtest ───
# Conventions: P&L in €, VaR and ES reported as positive loss amounts.
def book_pnl(returns, positions):
    """Daily € P&L of static positions: sum of (€ notional × daily return)."""
    cols = list(positions)
    return (returns[cols] * pd.Series(positions)).sum(axis=1)


def hist_var(pnl, conf):
    """Historical-simulation VaR at confidence `conf`, as a positive € loss."""
    return -np.percentile(pnl, (1 - conf) * 100)


def hist_es(pnl, conf):
    """Expected Shortfall: average loss on the days at or beyond the VaR quantile."""
    q = np.percentile(pnl, (1 - conf) * 100)
    return -pnl[pnl <= q].mean()


def var_backtest(pnl, window=250, test_days=250, confs=(0.95, 0.99)):
    """Rolling historical VaR computed from the previous `window` days only (no look-ahead),
    compared with the realised P&L over the last `test_days` days."""
    out = pd.DataFrame({'PnL': pnl})
    for c in confs:
        out[f'VaR{int(c * 100)}'] = -pnl.rolling(window).quantile(1 - c).shift(1)
        out[f'Exc{int(c * 100)}'] = out['PnL'] < -out[f'VaR{int(c * 100)}']
    return out.dropna().tail(test_days)


def kupiec_pof(n, x, p):
    """Kupiec proportion-of-failures test.

    n observations, x exceptions, p expected exception rate (0.01 for 99% VaR).
    Returns (likelihood ratio, p-value); LR ~ chi-squared with 1 degree of freedom.
    """
    phat = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = ((n - x) * np.log(1 - phat) if x < n else 0.0) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - chi2.cdf(lr, df=1)


def basel_zone(exceptions, n=250):
    """Basel traffic light for 99% VaR over 250 days: green 0-4, yellow 5-9, red 10+.

    The Basel zones (BCBS 1996) are defined for a 250-day backtest only. Scaling the
    thresholds by n / 250 is not the Basel method, so for fewer days None is returned
    and the page says there are not enough days.
    """
    if n < 250:
        return None
    if exceptions <= 4:
        return 'Green'
    if exceptions <= 9:
        return 'Yellow'
    return 'Red'


# ─── Price range over a short horizon (risk of waiting) ───
@st.cache_data(ttl=3600, show_spinner="Simulating price paths...")
def garch_price_range(close, horizon=5, simulations=10000, conf=0.90):
    """Range of the price `horizon` business days ahead from a GARCH(1,1) fit.

    Fitted on daily log returns with zero mean (no view on direction) and Student-t
    shocks (fat tails). Simulated paths give, for each day ahead, the lower, median
    and upper percentile of the price. Returns None if the fit fails.
    """
    close = close.where(close > 0)
    # Log returns on the series' own days; NaN across a stale gap rather than one jump.
    log_ret = (100 * np.log(close).diff()).dropna()
    close = close.dropna()
    try:
        res = arch_model(log_ret, mean='Zero', vol='GARCH', p=1, q=1, dist='t',
                         rescale=False).fit(disp='off')
        fc = res.forecast(horizon=horizon, method='simulation', simulations=simulations,
                          reindex=False, random_state=np.random.RandomState(42))
    except Exception:
        return None
    cum = np.cumsum(fc.simulations.values[-1], axis=1) / 100   # (simulations, horizon)
    paths = close.iloc[-1] * np.exp(cum)
    tail = (1 - conf) / 2 * 100
    return dict(
        price_now=float(close.iloc[-1]),
        date=close.index[-1],
        lo=np.percentile(paths, tail, axis=0),
        median=np.percentile(paths, 50, axis=0),
        hi=np.percentile(paths, 100 - tail, axis=0),
        horizon_vol=float(np.std(cum[:, -1]) * 100),   # % std of the horizon log return
        params={k: float(v) for k, v in res.params.items()},
    )


def brent_roll_days(index):
    """Days on which Yahoo's continuous Brent series (BZ=F) switches contract.

    ICE Brent stops trading on the last business day of the second month before delivery, so the
    front month changes on the first trading day of each month. In December the expiry moves one
    business day earlier (ICE's rule for the business day before New Year's Day), so the January
    roll happens on the last trading day of December instead. The first day of the series is
    skipped (it may be mid-month).
    """
    months = pd.Series(index.to_period('M'), index=index)
    first = months.ne(months.shift())
    first.iloc[0] = False
    first &= index.month != 1                      # no roll on the first trading day of January
    last_dec = months.ne(months.shift(-1)) & (index.month == 12)
    last_dec.iloc[-1] = False                      # the series may end mid-December
    return index[(first | last_dec).values]


def ttf_expiry_days(index):
    """Trading days in `index` that are, or follow, a TTF front-month expiry.

    ICE Endex TTF futures stop trading two UK business days before the first calendar day of the
    delivery month. UK bank holidays are not modelled (weekdays only), so a date can be one day
    off around a holiday. Yahoo's continuous TTF=F may switch contract on either of these days, so
    a large move on them can be partly the gap between two contracts.
    """
    if len(index) == 0:
        return index
    out = set()
    months = pd.period_range(index.min().to_period('M'), index.max().to_period('M') + 1, freq='M')
    for m in months:
        first = m.to_timestamp()
        expiry = first - pd.offsets.BDay(2)
        nxt = expiry + pd.offsets.BDay(1)
        out.update([expiry.normalize(), nxt.normalize()])
    idx_norm = index.normalize()
    return index[idx_norm.isin(list(out))]


# ─── Which monthly contract a continuous front-month series is tracking ───
@st.cache_data(ttl=3600, show_spinner=False)
def front_month_contract(continuous, root, last_date, last_price):
    """Delivery month of the monthly contract that `continuous` (e.g. TTF=F) tracked on `last_date`.

    Yahoo Finance does not label the continuous series reliably, so the next two monthly
    contracts are downloaded and the one whose close on `last_date` equals the continuous
    close is taken. Returns (dt.date of the delivery month, contract symbol), or (None, None)
    if no contract matches.
    """
    d = pd.Timestamp(last_date)
    for k in (1, 2, 0):
        month_index = d.month - 1 + k
        year, month = d.year + month_index // 12, month_index % 12 + 1
        sym = f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}.NYM"
        try:
            data = yf.download(sym, start=d - pd.Timedelta(days=10), end=d + pd.Timedelta(days=1),
                               progress=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if d in close.index and abs(float(close.loc[d]) - float(last_price)) < 1e-6:
            return dt.date(year, month, 1), sym
    return None, None


# ─── Forward curves from individual monthly futures on Yahoo Finance ───
MONTH_CODES = "FGHJKMNQUVXZ"
# Continuous ticker → (monthly contract root on NYMEX, unit). EU carbon has no monthly
# contracts on Yahoo Finance: EUA futures are annual December contracts on ICE Endex.
CURVE_ROOTS = {"CL=F": ("CL", "$/barrel"), "BZ=F": ("BZ", "$/barrel"), "TTF=F": ("TTF", "€/MWh")}
CURVE_FRESH_DAYS = 5     # a contract needs a close within this many trading days


@st.cache_data(ttl=3600, show_spinner="Fetching monthly futures contracts...")
def forward_curve(root, n=12, today=None):
    """Latest daily closes of the next `n` monthly contracts, e.g. CLX26.NYM, CLZ26.NYM, ...

    Symbols are generated from the current month onwards (with spares for contracts
    that have already expired) and downloaded in one batch. A contract is kept if it
    has a close within the last CURVE_FRESH_DAYS trading days of the batch (the dates
    on which any contract traded), so expired or stale contracts drop out while a
    contract that missed only the latest day stays. Its price is its latest daily close
    on Yahoo Finance (for a finished day normally the settlement; on a trading day the
    latest price). Weekend rows (Sunday evening session) are dropped. Nothing is
    interpolated: missing contracts are simply absent. Returns a DataFrame with
    Contract, Delivery, Price and Last trade, sorted by delivery month.
    """
    today = today or dt.date.today()
    symbols = {}
    for k in range(n + 3):
        month_index = today.month - 1 + k
        year, month = today.year + month_index // 12, month_index % 12 + 1
        symbols[f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}.NYM"] = dt.date(year, month, 1)

    empty = pd.DataFrame(columns=['Contract', 'Delivery', 'Price', 'Last trade'])
    try:
        data = yf.download(list(symbols), start=today - dt.timedelta(days=21), progress=False)
    except Exception:
        return empty
    if data is None or data.empty or 'Close' not in data:
        return empty
    closes = data['Close']
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(list(symbols)[0])
    # Same rule as _raw_close: a Saturday or Sunday row is an intraday price from the
    # Sunday evening session, not a daily close.
    closes = closes[closes.index.dayofweek < 5]

    rows = []
    for sym in closes.columns:
        s = closes[sym].dropna()
        if not s.empty and sym in symbols:
            rows.append({'Contract': sym, 'Delivery': symbols[sym],
                         'Price': float(s.iloc[-1]), 'Last trade': s.index[-1]})
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    trading_days = closes.dropna(how='all').index
    cutoff = trading_days[-min(CURVE_FRESH_DAYS, len(trading_days))]
    df = df[df['Last trade'] >= cutoff]
    return df.sort_values('Delivery').head(n).reset_index(drop=True)


def gas_season(delivery):
    """Gas-year season of a delivery month: Winter = Oct to Mar, Summer = Apr to Sep."""
    y = delivery.year % 100
    if delivery.month >= 10:
        return f"Winter {y:02d}/{(y + 1) % 100:02d}"
    if delivery.month <= 3:
        return f"Winter {(y - 1) % 100:02d}/{y:02d}"
    return f"Summer {y:02d}"
