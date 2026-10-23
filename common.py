"""Shared configuration, data loading and cached risk calculations.

Every page imports this module, so the carbon benchmark, the commodity list and the
core risk numbers (volatility, VaR, GARCH, stress period) are defined and computed in one
place. The heavy calculations are wrapped in st.cache_data, so a page that needs a
number another page already computed gets it from the cache.
"""
import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from arch import arch_model
from arch.univariate import GARCH, StudentsT, ZeroMean
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



OIL_BARREL_TERMS = ['a barrel', 'per barrel', 'barrels a day', 'barrels per day', 'bpd']


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
            # "barrel" alone also matches stories such as "Tank barrel production returns to UK", so it counts
            # only as an oil-price unit ("a barrel", "per barrel", "barrels a day").
            "keywords": ['oil', 'crude', 'WTI', 'OPEC', 'petroleum', 'refinery'] + OIL_BARREL_TERMS,
            "rss_query": "crude+oil+Europe+price",
            "news_term": "oil",   # one word for country and 30-day news queries
        },
        "Brent Crude Oil": {
            "ticker": "BZ=F",
            "unit": "$/barrel",
            "color": "darkred",
            "keywords": ['oil', 'crude', 'Brent', 'OPEC', 'petroleum', 'North Sea'] + OIL_BARREL_TERMS,
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


def berlin_today():
    """Today's date in Berlin. The default end date and the Power page's "is it today" checks use it, so the
    page agrees with the power market's calendar day whatever time zone the server runs in."""
    return pd.Timestamp.now(tz="Europe/Berlin").date()


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
        end_date=st.session_state.get("end_date", berlin_today()),
    )


# ─── Data Download (cached) ───
STALE_RUN = 5   # this many identical consecutive closes = a stale feed (suspension, no trading)


_DATA_START = dt.date(2017, 1, 1)   # every price series is downloaded once from here and sliced


def _batch_contracts(ticker, today):
    """Monthly contracts downloaded with a continuous future, for the roll check: from two months before the
    month of `today` to five months after it (contracts that have expired come back empty from Yahoo)."""
    if ticker not in CURVE_ROOTS:
        return []
    root = CURVE_ROOTS[ticker][0]
    months = pd.period_range(pd.Timestamp(today).to_period('M') - 2, pd.Timestamp(today).to_period('M') + 5,
                             freq='M')
    return [f"{root}{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM" for p in months]


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def _download_batch(ticker, first, today):
    """Daily closes from `first` to `today` exactly as downloaded: one column for `ticker` and, for a continuous
    front-month future (TTF=F, CL=F, BZ=F), one column for each monthly contract of _batch_contracts. Empty if
    unavailable.

    One download per ticker is shared by every calculation (volatility, percentiles, thresholds, VaR and the
    contract roll check), so they all use the same last price: during and after the trading session the latest
    row keeps changing (after the settlement it can be a price from the evening electronic session), and two
    separate downloads a minute apart could disagree. yfinance treats `end` as exclusive, so one day is added.
    """
    tickers = [ticker] + _batch_contracts(ticker, today)
    try:
        data = yf.download(tickers if len(tickers) > 1 else ticker, start=first,
                           end=pd.Timestamp(today) + pd.Timedelta(days=1), progress=False)
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    close = data['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(ticker)
    if ticker not in close:
        return pd.DataFrame()
    # Yahoo sometimes adds a Saturday or Sunday row with an intraday price from the Sunday
    # evening session (e.g. CL=F, BZ=F). It is not a settlement, so weekend rows are dropped.
    close = close[close.index.dayofweek < 5]
    return close[close[ticker].notna()].astype(float)


def _download_close(ticker, first, today):
    """The continuous series of the shared download (_download_batch), empty if unavailable."""
    batch = _download_batch(ticker, first, today)
    return batch[ticker] if ticker in batch else pd.Series(dtype=float)


def _slice(close, start, end):
    return close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]


def _download_day():
    """The `today` of the shared download (_download_batch): the Berlin date (berlin_today), the same day the
    default end date and the Power page use. shared_download and compute_core both call this, so they always
    hit the same cached download, whatever time zone the server runs in."""
    return berlin_today()


def shared_download(ticker, start):
    """The shared download (_download_batch) that load_close(ticker, start, ...) reads: the continuous series
    and, for TTF=F, CL=F and BZ=F, the monthly contracts of _batch_contracts. A page that needs a monthly
    contract next to the continuous series calls this, so both come from the same download."""
    return _download_batch(ticker, min(pd.Timestamp(start).date(), _DATA_START), _download_day())


def _raw_close(ticker, start, end):
    """Daily closes from `start` to `end` (both included), cut from the one shared download."""
    batch = shared_download(ticker, start)
    close = batch[ticker] if ticker in batch else pd.Series(dtype=float)
    return _slice(close, start, end)


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


VOL_DAYS, VOL_MIN = 30, 20   # 30-day volatility: a window of 30 trading days, at least 20 returns in it


def rolling_vol(close):
    """The one definition of 30-day volatility (daily, %) behind every figure on the Risk and Market pages: the
    standard deviation of simple daily returns over the last VOL_DAYS rows of the series' own trading days, given
    only when at least VOL_MIN of those rows have a return.

    `close` is the series on its own trading days with stale repeats set to NaN and the rows kept (load_close).
    A stale run therefore stays inside the window as rows without a return, so a window never reaches back
    across a suspension to fill itself up: after CARB.L's 2020-21 suspension the first figure comes once 20
    returns after the gap exist. compute_core (the page series, the latest figure and the selected-range
    average), the percentile history (vol_context), the Market page's check window (three_year_reference) and
    the GARCH reference all take their 30-day volatility from here; vol_gap explains a missing latest figure."""
    return simple_returns(close).rolling(VOL_DAYS, min_periods=VOL_MIN).std() * 100


def rolling_corr(close, compare):
    """The one definition of 30-day correlation behind every figure on the Risk and Market pages: the Pearson
    correlation of simple daily returns of `close` and `compare` over the last VOL_DAYS rows of `close`'s own
    trading days, given only when at least VOL_MIN of those rows have a return in both series. The same window
    rule as rolling_vol.

    `close` and `compare` are each on their own trading days with stale repeats set to NaN and the rows kept
    (load_close). The returns of `compare` are taken on its own days and placed on the rows of `close`, so a stale
    run or a non-positive price of either series stays inside the window as rows without a pair: a window never
    reaches back across a suspension of the commodity itself (CARB.L in 2020-21 on the carbon page) to fill
    itself up. compute_core (the page line, the latest figure) and three_year_reference (the Market page's
    check window) both take their 30-day correlation from here; corr_gap explains a missing latest figure."""
    pairs = pd.DataFrame({'R': simple_returns(close),
                          'CR': simple_returns(compare).reindex(close.index)})
    return pairs['R'].rolling(VOL_DAYS, min_periods=VOL_MIN).corr(pairs['CR'])


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
    """Returns, rolling volatility/correlation, VaR, the volatility context (pre-war multiple, percentile since
    the start of the data), the 3-year reference window of the Market page's checks and the contract roll check
    of the latest 30-day window.

    Everything about `ticker` comes from one read of its shared download (_download_batch), inside this one
    cached result, so the percentile, the pre-war multiple, the 3-year reference, VaR and the roll check can
    never come from two different downloads, even after the download cache expires. Returns None when fewer
    than 30 overlapping observations are available. `latest_vol` is the 30-day volatility (rolling_vol) of the
    last day with a price (`vol_asof`); it is None when that day's window (`vol_window`, its last VOL_DAYS
    returns on the series' own trading days) holds fewer than VOL_MIN returns, and `vol_gap` then gives the count
    and the stale runs in the window, so the pages can say why instead of showing an older figure. `latest_corr`
    and `corr_gap` say the same of the 30-day correlation with the comparison series (rolling_corr): the figure of
    `vol_asof`, or None with the count of days with both returns in the window and the stale runs of either series.
    """
    today = _download_day()
    batch = _download_batch(ticker, min(pd.Timestamp(start).date(), _DATA_START), today)
    raw = batch[ticker] if ticker in batch else pd.Series(dtype=float)
    price = _slice(raw, start, end)
    price = price.mask(_stale_mask(price))
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

    # 30-day volatility on the commodity's own trading days, stale days kept as rows without a return
    # (rolling_vol), then read on the days with a price.
    own_vol = rolling_vol(price)
    df_analysis['Volatility'] = own_vol.reindex(df_analysis.index)

    # 30-day correlation of returns (not price levels, to avoid spurious correlation), on the same rows and with
    # the same window rule as the volatility (rolling_corr), then read on the days with a price.
    own_corr = rolling_corr(price, compare)
    df_analysis['Rolling Correlation'] = own_corr.reindex(df_analysis.index)

    # The latest figure is the one of the last day with a price. When that day's 30-day window holds fewer than
    # VOL_MIN returns (after a suspension), there is no latest figure: an older one is not shown in its place.
    vol_asof = df_analysis.index[-1]
    latest_vol = df_analysis['Volatility'].iloc[-1]
    latest_vol = None if pd.isna(latest_vol) else float(latest_vol)
    window_returns = simple_returns(price).loc[:vol_asof].tail(VOL_DAYS)
    # The same for the correlation: the pairs of returns in the last VOL_DAYS rows up to vol_asof.
    latest_corr = df_analysis['Rolling Correlation'].iloc[-1]
    latest_corr = None if pd.isna(latest_corr) else float(latest_corr)
    corr_window = pd.DataFrame({'R': window_returns,
                                'CR': simple_returns(compare).reindex(window_returns.index)}).dropna()
    avg_vol = df_analysis['Volatility'].dropna().mean()

    returns_clean = df_analysis['Returns'].dropna()
    # Headline VaR / ES from the last VAR_WINDOW trading days; full-period figures alongside.
    var_returns = returns_clean.tail(VAR_WINDOW)
    var_95, var_99, es_975 = _var_es(var_returns)
    var_95_full, var_99_full, es_975_full = _var_es(returns_clean)

    returns_corr = df_analysis[['Returns', 'Compare_Returns']].dropna()
    overall_corr = returns_corr['Returns'].corr(returns_corr['Compare_Returns'])

    # 30-day volatility since the start of the data (2017) up to today, from the same download: the source of
    # the percentile and the pre-war multiple (vol_context cuts it at the end date).
    hist = _slice(raw, _CONTEXT_START, today)
    hist_vol = rolling_vol(hist.mask(_stale_mask(hist))).dropna()
    # The Market page's 3-year check window ends on the end date, never later, and comes from this same download
    # (the comparison series from its own shared download, the one `compare` above is cut from).
    ref_price = _slice(raw, _CONTEXT_START, end)
    ref_mask = _stale_mask(ref_price)
    # load_close, written out so the stale days of the comparison are known too.
    cmp_raw = _raw_close(compare_ticker, min(pd.Timestamp(start).date(), _CONTEXT_START), end)
    cmp_mask = _stale_mask(cmp_raw)
    ref3y = three_year_reference(ref_price.mask(ref_mask), cmp_raw.mask(cmp_mask), df_analysis,
                                 stale=(ticker, ref_mask), compare_stale=(compare_ticker, cmp_mask))

    vol_gap = None
    if latest_vol is None:
        first = window_returns.index[0]
        runs = [(t, a, b) for t, a, b in _stale_runs([(ticker, ref_mask)], ref_mask.index[0], vol_asof)
                if b >= first]
        vol_gap = dict(n=int(window_returns.notna().sum()), need=VOL_MIN, days=len(window_returns), stale=runs)
    corr_gap = None
    if latest_corr is None and has_compare and len(window_returns):
        first = window_returns.index[0]
        runs = sorted([(t, a, b) for t, a, b in _stale_runs([(ticker, ref_mask), (compare_ticker, cmp_mask)],
                                                            ref_mask.index[0], vol_asof) if b >= first],
                      key=lambda r: r[2])
        # A stale run of the comparison that lasts up to the commodity's last price has not ended.
        _cmp_rows = cmp_mask.index[cmp_mask.index <= vol_asof]
        ongoing = bool(runs) and runs[-1][0] == compare_ticker and len(_cmp_rows) > 0 and runs[-1][2] == _cmp_rows[-1]
        corr_gap = dict(n=len(corr_window), need=VOL_MIN, days=len(window_returns), stale=runs, ongoing=ongoing)
    return dict(
        ticker=ticker,
        df_analysis=df_analysis, has_compare=has_compare,
        latest_vol=latest_vol, avg_vol=avg_vol, vol_asof=vol_asof, vol_window=window_returns, vol_gap=vol_gap,
        latest_corr=latest_corr, corr_gap=corr_gap,
        returns_clean=returns_clean, var_95=var_95, var_99=var_99, es_975=es_975,
        var_days=len(var_returns), var_95_full=var_95_full, var_99_full=var_99_full,
        es_975_full=es_975_full,
        overall_corr=overall_corr,
        ref3y=ref3y,
        vol_ctx=vol_context(hist_vol, vol_asof, latest_vol) if latest_vol is not None else None,
        rolls=window_rolls(ticker, batch, window_returns) if latest_vol is not None else None,
        last_download=raw.index[-1] if len(raw) else None,
    )


# ─── GARCH(1,1) ───
@st.cache_data(ttl=3600, show_spinner="Fitting GARCH model...")
def fit_garch(returns_clean, ref_vol):
    """Fit GARCH(1,1) on daily % returns and forecast 10 days ahead. `ref_vol` is the page's reference level,
    an average 30-day volatility (daily, %) from rolling_vol: the Risk page's selected-range average, the Market
    page's check-window average. It only decides whether the GARCH long-run level is shown.

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
    # page's average 30-day volatility (`ref_vol`, rolling_vol, not a second definition here); the GARCH
    # value is shown for information only, and only when it is stable (α + β < 0.98 and within 1.5x
    # that average).
    sample_avg_vol = float(ref_vol)
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


# ─── Volatility reference windows ───
# The volatility heading compares today's 30-day volatility with two references that do not roll with the
# market: the pre-war quarter (a calm baseline) and the commodity's whole history since the start of the data
# (its percentile). A rolling 3-year Normal / Elevated / Extreme label was dropped: its cut-offs had no
# industry source and its thresholds moved with the market (low after calm years, high after a crisis).
# The Market page's statistical checks still ask "unusual compared with the last 3 years?", so the 3-year
# window below is kept for them only.
CHECK_WINDOW = 750               # trading days, about 3 years: the Market page's check window
_MIN_HISTORY = 250
ANNUAL = np.sqrt(252)   # daily volatility x sqrt(252) = annualised volatility


def _stale_runs(stales, first, last):
    """[(ticker, first day, last day)] of every run of stale repeats between `first` and `last`, from `stales`
    ((ticker, _stale_mask of its close) pairs; None is skipped)."""
    out = []
    for item in stales:
        if item is None:
            continue
        ticker, mask = item
        m = mask[(mask.index >= first) & (mask.index <= last)]
        if not m.any():
            continue
        run_id = (~m).cumsum()[m]
        out += [(ticker, g.index[0], g.index[-1]) for _, g in run_id.groupby(run_id)]
    return out


def _long_span(first, last):
    """True when a window of CHECK_WINDOW trading days spans more than 3 years and 30 days, i.e. days without a
    price (a suspension masked as stale) pushed its start back."""
    return pd.Timestamp(first) < pd.Timestamp(last) - pd.DateOffset(years=3) - pd.Timedelta(days=30)


def three_year_reference(price, compare, fallback, stale=None, compare_stale=None):
    """The reference window of the Market page's statistical checks: the last CHECK_WINDOW trading days up to
    the end date of the selected range, never later. A past end date is judged against the 3 years before it,
    with no data from after it.

    `price` is the commodity's close from compute_core's download (the only caller) from the start of the data
    to the end date, stale repeats masked (rows kept); `compare` the comparison series over the same span. The
    definitions are compute_core's: simple returns on each series' own trading days, non-positive prices
    dropped, 30-day volatility from rolling_vol (the same figure on the same day as the page's latest value and
    the percentile history) and 30-day correlation from rolling_corr, both read on the days with a price. Returns dict(vol, corr, rets, start, end, n, mean, std, fallback, rows, days,
    short): the last CHECK_WINDOW values of 30-day volatility (daily, %) and of the 30-day correlation of
    returns up to the last day with a price (`end`), and the daily returns from the window's first day to `end`
    (for the 3-year VaR99, the GARCH fit and the counts of days fired); `mean` (of vol) is the GARCH check's
    reference level; `days` are the days with a price from `start` to `end` and `rows` their number, which is
    larger than `n` when some of them have no 30-day volatility (the first days after a suspension); `short` is
    True when the history up to the end date has fewer than CHECK_WINDOW values. If it has fewer than
    _MIN_HISTORY, the selected range (`fallback`, compute_core's df_analysis) is used instead and `fallback` is
    True.

    Stale repeats have no return, so where a series was suspended the CHECK_WINDOW values reach back further
    than 3 years (CARB.L to 2 Apr 2024: from Dec 2019, across its 2020-21 suspension). `long` is True when
    the volatility window spans more than 3 years and 30 days, and `stale` then lists the stale runs inside it
    ((ticker, first, last), from the argument `stale`, (ticker, stale mask of `price`)); `corr_start`,
    `corr_end`, `corr_long` and `corr_stale` say the same of the correlation window, which also loses the stale
    days of the comparison series (`compare_stale`). `corr_own` is True when the correlation window's first day,
    last day or number of values differs from the volatility window's (its values, not its rows), so the page
    names it separately.
    """
    df = pd.DataFrame({'Price': price, 'R': simple_returns(price), 'V': rolling_vol(price),
                       'C': rolling_corr(price, compare)}).dropna(subset=['Price'])
    df = df[df['Price'] > 0]
    vol = df['V'].dropna().tail(CHECK_WINDOW)
    corr = df['C'].dropna().tail(CHECK_WINDOW)
    rets = df['R'].dropna()
    used_fallback = len(vol) < _MIN_HISTORY
    if used_fallback:
        vol, corr = fallback['Volatility'].dropna(), fallback['Rolling Correlation'].dropna()
        rets = fallback['Returns'].dropna()
        days = fallback.index[fallback.index >= vol.index[0]]
    else:
        rets = rets.loc[vol.index[0]:]
        days = df.index[df.index >= vol.index[0]]
    out = dict(vol=vol, corr=corr, rets=rets, start=vol.index[0], end=days[-1], n=len(vol),
               mean=float(vol.mean()), std=float(vol.std()), fallback=used_fallback,
               rows=len(days), days=days, short=(not used_fallback and len(vol) < CHECK_WINDOW),
               long=False, stale=[], corr_start=None, corr_end=None, corr_long=False, corr_stale=[],
               corr_own=False)
    if not used_fallback:
        out.update(long=_long_span(out['start'], out['end']))
        if out['long']:
            out['stale'] = _stale_runs([stale], out['start'], out['end'])
    if len(corr):
        out.update(corr_start=corr.index[0], corr_end=corr.index[-1])
        if not used_fallback:
            # The correlation window is named on its own whenever its first day, last day or number of values
            # differs from the volatility window's (TTF to 30 Jun 2020: 605 values from 09 Jan 2018 to 24 Jun 2020,
            # after CARB.L went stale on 11 Jun, against 656 volatility values from 20 Nov 2017 to 30 Jun 2020).
            out['corr_own'] = (corr.index[0] != vol.index[0] or corr.index[-1] != vol.index[-1]
                               or len(corr) != len(vol))
            out['corr_long'] = _long_span(corr.index[0], corr.index[-1])
            if out['corr_long']:
                out['corr_stale'] = _stale_runs([stale, compare_stale], corr.index[0], corr.index[-1])
    return out


# Fixed pre-war reference: the last full quarter before the US-Israeli strikes on Iran on 28 Feb 2026.
# A fixed window, like the fixed event dates of the historical scenarios, so the reference does not
# drift with the date range; a percentile alone cannot say "twice as volatile as before the war".
PREWAR_START, PREWAR_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")
_CONTEXT_START = dt.date(2017, 1, 1)


@st.cache_data(ttl=3600, show_spinner=False)
def vol_context(hist_vol, as_of, vol_now):
    """Where `vol_now` (30-day volatility of daily returns, %) sits against the commodity's own history
    up to `as_of`: its percentile rank since the start of the data (Jan 2017, or later where the series starts
    later; TTF=F starts in Oct 2017) and the ratio to the median 30-day volatility of the pre-war quarter
    (Oct to Dec 2025). `hist_vol` is the 30-day volatility from compute_core's download (simple returns, 30-day
    window, stale repeats excluded). The pre-war median is given only when `as_of` is on or after 31 Dec 2025:
    before that the quarter is not complete, and a median of part of it would be labelled as the whole
    quarter."""
    vol = hist_vol[hist_vol.index <= pd.Timestamp(as_of)]
    if len(vol) < 250:
        return None
    # Share of the history strictly below vol_now. Both sides are rounded to 10 decimals first: today's
    # volatility is computed on two paths (compute_core's window and the history here) that can differ by
    # ~1e-14, which would otherwise count today itself as "below today" on some days and not on others.
    h = np.sort(np.round(vol.values, 10))
    now = round(float(vol_now), 10)
    complete = pd.Timestamp(as_of) >= PREWAR_END
    prewar = vol[(vol.index >= PREWAR_START) & (vol.index <= PREWAR_END)]
    pre_med = float(prewar.median()) if complete and len(prewar) >= 20 else None
    return dict(p_all=100.0 * np.searchsorted(h, now, side='left') / len(h), top_all=bool(now >= h.max()),
                all_start=vol.index[0], prewar_incomplete=not complete,
                prewar_median=pre_med, prewar_ratio=(vol_now / pre_med) if pre_med else None)


def fmt_vol(daily, decimals=0):
    """A daily volatility (%) written the one way the pages use: annualised first, daily in brackets."""
    return f"{daily * ANNUAL:.{decimals}f}% ann. ({daily:.2f}% daily)"


def _ordinal(x):
    n = int(round(x))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


_WAR_START = pd.Timestamp("2026-03-01")
# Oil's 2020 crash, named in the volatility history from March 2020 (WTI and Brent only).
_CRASH_START = pd.Timestamp("2020-03-01")
_OIL_TICKERS = ('CL=F', 'BZ=F')
_PRICE_TOL = 1e-4   # a settled continuous close and a single-contract close are the same quote
# The download's last day can be a live price, and Yahoo updates the continuous row and the contract rows at
# slightly different moments (on 20 Oct 2026, 00:16 UTC: CL=F 98.09, CLZ26 98.10). On that day the closest
# contract is taken if it is within 0.05% and every other contract is at least 0.05 percentage points farther.
_LAST_TOL = 5e-4
# A continuous front-month series on a given day follows a contract delivering a fixed number of months later:
# TTF and WTI 1 or 2 (the next month, or the one after once the next has expired), Brent 2 or 3 (two months
# ahead, three after the December expiry moved before New Year). Only those contracts are compared, so a
# back-month contract that happens to close at the same price is never taken as the front.
_FRONT_AHEAD = {"TTF": (1, 2), "CL": (1, 2), "BZ": (2, 3)}


def match_front_contract(batch, ticker, day):
    """The monthly contract whose close in the shared download (_download_batch) equals the continuous
    series' close on `day`, or None. Only contracts that can be the front month on that day are compared
    (_FRONT_AHEAD). On the download's last day, which can be a live price, the closest contract counts if it is
    within _LAST_TOL and every other contract is at least _LAST_TOL farther; on earlier days the closes must
    agree within _PRICE_TOL and exactly one contract may match. window_rolls and front_month_contract both use
    this rule."""
    if ticker not in CURVE_ROOTS or ticker not in batch or day not in batch.index:
        return None
    cont = batch.at[day, ticker]
    if pd.isna(cont):
        return None
    root = CURVE_ROOTS[ticker][0]
    lo, hi = _FRONT_AHEAD[root]
    m0 = day.year * 12 + day.month
    quotes = {}
    for s in batch.columns:
        if s == ticker:
            continue
        d = contract_delivery(s, root)
        if lo <= d.year * 12 + d.month - m0 <= hi and pd.notna(batch.at[day, s]):
            quotes[s] = batch.at[day, s]
    if day == batch[ticker].dropna().index[-1]:
        gaps = sorted((abs(q / cont - 1), s) for s, q in quotes.items())
        ok = gaps and gaps[0][0] < _LAST_TOL and all(g >= gaps[0][0] + _LAST_TOL for g, _ in gaps[1:])
        return gaps[0][1] if ok else None
    hits = [s for s, q in quotes.items() if abs(q - cont) < _PRICE_TOL]
    return hits[0] if len(hits) == 1 else None


def window_rolls(ticker, batch, window_returns):
    """Contract rolls of a continuous front-month future (TTF=F, CL=F, BZ=F) inside a volatility window.

    `window_returns` are the daily returns behind the latest 30-day volatility and `batch` is the shared
    download (_download_batch): the continuous series and the individual monthly contracts from one download,
    the same one behind the volatility. Each day is matched to the one contract whose
    close equals the continuous close (on the download's last day, which can be a live price, the closest
    contract within _LAST_TOL when all others are at least _LAST_TOL farther). Only contracts that can be the
    front month on that day are compared (_FRONT_AHEAD). A return day is a roll day when the
    continuous close equals contract
    M on that day but did not equal M's close on the day before; its return then contains the gap between two
    contracts, and M's own return that day replaces it. A day is not identified when the continuous close
    matches no single contract on Yahoo Finance (expired contracts are no longer listed) or M has no close on
    the day before; nothing is guessed for those days. For Brent the ICE rule (brent_roll_days) is the page's
    roll calendar, so an unmatched Brent day counts as a roll day only if the rule makes it one.

    Returns None for a ticker without monthly contracts, else dict(rolls, unknown, rule_only, adjusted_vol):
    rolls = [(date, new contract symbol, continuous return, new contract return)], unknown = unidentified
    return days, rule_only = Brent rule days with no contract price to adjust, adjusted_vol = 30-day
    volatility (daily, %) with the new contract's return on the roll days (None if there is no roll)."""
    if ticker not in CURVE_ROOTS or ticker not in batch:
        return None
    r = window_returns.dropna()
    if r.empty:
        return None
    first, last = r.index[0], r.index[-1]
    data = batch[(batch.index >= first - pd.Timedelta(days=14)) & (batch.index <= last)]
    cont = data[ticker].dropna()

    def match(day):
        return match_front_contract(batch, ticker, day)

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
    """Consecutive runs of `days` within `index`, written '08 Sep to 22 Sep 2026' or '19 Oct 2026'; two runs are
    joined with 'and', three or more with commas and 'and' before the last ('02 Feb 2026 and 02 Mar 2026')."""
    pos = sorted(index.get_indexer(days))
    runs, start = [], pos[0]
    for a, b in zip(pos, pos[1:] + [None]):
        if b != a + 1:
            runs.append((index[start], index[a]))
            start = b
    parts = [f"{a:%d %b %Y}" if a == b else
             (f"{a:%d %b} to {b:%d %b %Y}" if a.year == b.year else f"{a:%d %b %Y} to {b:%d %b %Y}")
             for a, b in runs]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def contract_delivery(sym, root):
    """Delivery month (first day) of a monthly contract symbol such as CLZ26.NYM."""
    return dt.date(2000 + int(sym[len(root) + 1:len(root) + 3]), MONTH_CODES.index(sym[len(root)]) + 1, 1)


def rule_roll_days(root, days):
    """For WTI (CL) and Dutch TTF (TTF): the days among `days` that are a front contract's last trading day by the
    exchange rule (contract_expiry) or the next business day on the same calendar, as {day: (delivery month,
    last trading day)}. Yahoo's continuous series switches contract on or near the expiry, so these are the days
    on which a roll is possible when the roll check (window_rolls) cannot match a contract; they are candidates,
    not identified roll days. Brent has its own rule calendar (brent_roll_days). Empty for other roots."""
    if root not in ("CL", "TTF") or len(days) == 0:
        return {}
    days = pd.DatetimeIndex(days).normalize()
    cal = EXPIRY_CALENDARS[root]
    out = {}
    lo, hi = days.min().to_period('M'), days.max().to_period('M') + 3
    for m in pd.period_range(lo, hi, freq='M'):
        exp = contract_expiry(root, m.to_timestamp())
        nxt = pd.Timestamp(np.busday_offset(exp.date(), 1, busdaycal=cal))
        for d in (exp.normalize(), nxt.normalize()):
            if d in days:
                out[d] = (m.to_timestamp(), exp)
    return out


def roll_sentence(ticker, rc, window_returns):
    """The line under the volatility heading when the 30-day window contains contract roll days, or ''.
    `rc` is compute_core's window_rolls result for `window_returns`."""
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
        cand = rule_roll_days(root, rc['unknown'])
        # A contract whose roll was already identified (on its expiry or the day after) needs no candidate.
        found = {pd.Timestamp(t).normalize() for t, _, _, _ in rc['rolls']}
        cal = EXPIRY_CALENDARS.get(root)
        cand = {d: (m, exp) for d, (m, exp) in cand.items()
                if not ({exp.normalize(), pd.Timestamp(np.busday_offset(exp.date(), 1, busdaycal=cal))} & found)}
        if cand:
            parts.append(rule_roll_sentence(cand))
    if rc['rolls']:
        parts.append("The heading, the percentile and the pre-war multiple use the unadjusted figure, because "
                     "the history they are compared with contains roll days too.")
    return " ".join(parts)


def rule_roll_sentence(cand):
    """'By the exchange rule ...' for rule_roll_days' candidates: which contract stopped trading when, and that the
    day may be a roll day (not verified)."""
    by_exp = {}
    for d, (m, exp) in sorted(cand.items()):
        by_exp.setdefault((m, exp), []).append(d)
    bits = [f"the {m:%b %y} contract's last trading day was {exp:%d %b %Y}, so the series may have switched "
            "contract on " + " or ".join(f"{d:%d %b %Y}" for d in ds)
            for (m, exp), ds in by_exp.items()]
    return ("By the exchange rule, " + "; ".join(bits) + " (not verified: the expired contract is no longer listed "
            "on Yahoo Finance, so its close cannot be compared).")


def _gap_sentence(what, counted, g, as_of):
    """'No 30-day <what> on <day>: ...' from a vol_gap or corr_gap of compute_core (count, window, stale runs)."""
    as_of = pd.Timestamp(as_of)
    count = f"{g['n']} of the {g['need']} {counted} needed in the last {g['days']} trading days"
    if g['stale']:
        t, a, b = g['stale'][-1]
        if g.get('ongoing'):
            return (f"No 30-day {what} on {as_of:%d %b %Y}: the {t} prices from {a:%d %b %Y} are stale "
                    f"({count}).")
        return (f"No 30-day {what} on {as_of:%d %b %Y}: not enough prices since the stale {t} prices of "
                f"{a:%d %b %Y} to {b:%d %b %Y} ended ({count}).")
    return f"No 30-day {what} on {as_of:%d %b %Y}: not enough prices ({count})."


def vol_gap_sentence(core):
    """Why there is no latest 30-day volatility (compute_core's vol_gap), or '' when there is one. The count, the
    window and the stale run come from the data."""
    g = core.get('vol_gap')
    return _gap_sentence("volatility", "returns", g, core['vol_asof']) if g else ""


def corr_gap_sentence(core):
    """Why there is no latest 30-day correlation (compute_core's corr_gap), or '' when there is one, in the words
    of vol_gap_sentence. Written here once for the Risk and Market pages."""
    g = core.get('corr_gap')
    return _gap_sentence("correlation", "days with both returns", g, core['vol_asof']) if g else ""


# The US-Israeli strikes on Iran that started the war. Before this date the AI input names neither the war
# nor the strikes, and calls the fixed reference quarter by its months only ("the Oct to Dec 2025 level").
_STRIKES = pd.Timestamp("2026-02-28")


def vol_summary(core, name=None, with_roll=False, past_end=False, today=None):
    """The single source of every volatility statement on the Risk and Market pages and in the AI input.

    Everything comes from `core` (compute_core): the latest 30-day volatility, its multiple of the pre-war
    level, its percentile since the start of the data and the roll check, all from one download. Neither
    reference rolls with the market: the pre-war quarter is fixed and the percentile uses the whole history.
    Returns dict(headline, headline2, roll, detail, tile, ctx, ai, prewar_ai): `headline` and `headline2` are
    the two lines of the large neutral heading, `roll` the line on contract roll days in the 30-day window (only
    with `with_roll`), `detail` the paragraph under it, `tile` the value for a metric tile. With `past_end` (the
    end date chosen is before today) the percentile names the end date. When the last day with a price has no
    30-day volatility (vol_gap), the heading says so and why, and no figure is given.

    `ai` is the same statement for the AI input, which is written as of the data date: when that date is before
    the strikes of 28 Feb 2026 it names neither the strikes nor a war (the reference quarter is "the Oct to Dec
    2025 level"), and when the quarter is not complete by then it leaves the quarter out altogether. The page
    keeps its own wording ("no pre-war reference"). `prewar_ai` is the reference's name for the prompt ('' when
    there is none). `today` (default berlin_today(), the date the default end date uses) only decides whether the
    latest figure is today's: when the last 30-day volatility is of an earlier day (TTF and carbon most mornings),
    the percentile sentence names its date."""
    as_of = pd.Timestamp(core['vol_asof'])
    ctx = core['vol_ctx']
    window_returns = core['vol_window']
    rc = core['rolls']
    lead = f"{name + ' ' if name else ''}30-day volatility"
    if core['latest_vol'] is None:
        why = vol_gap_sentence(core)
        return dict(ctx=None, tile="n/a", roll="", headline=f"{lead}: not available", headline2=why, detail="",
                    ai=f"{lead}: not available. {why}", prewar_ai="")
    vol_now = float(core['latest_vol'])
    out = dict(ctx=ctx, tile=fmt_vol(vol_now, 0),
               roll=(roll_sentence(core['ticker'], rc, window_returns) if with_roll else ""),
               headline=f"{lead}: {vol_now * ANNUAL:.0f}% annualised ({vol_now:.2f}% daily)")
    if ctx is None:
        out.update(headline2="", detail="", ai=out['headline'], prewar_ai="")
        return out
    before_strikes = as_of < _STRIKES
    r = ctx['prewar_ratio']
    level, level_ai = "its pre-war level", ("its Oct to Dec 2025 level" if before_strikes else "its pre-war level")
    if r is None:
        compare, compare_ai = "no pre-war reference", (None if before_strikes else "no pre-war reference")
    elif r >= 1.1:
        compare, compare_ai = f"{r:.2f}x {level}", f"{r:.2f}x {level_ai}"
    elif r >= 0.9:
        compare, compare_ai = f"close to {level} ({r:.2f}x)", f"close to {level_ai} ({r:.2f}x)"
    else:
        compare, compare_ai = f"below {level} ({r:.2f}x)", f"below {level_ai} ({r:.2f}x)"
    span = f"since {ctx['all_start']:%b %Y}" + (f" (to {as_of:%d %b %Y})" if past_end else "")
    pct = f"highest {span}" if ctx['top_all'] else f"{_ordinal(ctx['p_all'])} percentile {span}"
    # A roll day in the 30-day window lifts or lowers the large figures; say so in the heading itself.
    roll_flag = (" · includes a contract roll day (see below)"
                 if with_roll and rc is not None and rc['rolls'] else "")
    out['headline2'] = f"{compare} · {pct}{roll_flag}"
    headline2_ai = (f"{compare_ai} · " if compare_ai else "") + pct + roll_flag
    detail, detail_ai = "", ""
    if ctx['prewar_median'] is not None:
        _lvl = f"{ctx['prewar_median'] * ANNUAL:.0f}% annualised ({ctx['prewar_median']:.2f}% daily)"
        detail += (f"Pre-war level: {_lvl}, the median of Oct to Dec 2025, the last full quarter before the strikes on "
                   "Iran on 28 Feb 2026. ")
        detail_ai += (f"Oct to Dec 2025 level: {_lvl}, the median of that quarter. " if before_strikes else
                      detail)
    elif ctx['prewar_incomplete']:
        detail += ("No pre-war multiple: the reference quarter (Oct to Dec 2025, the last full quarter before the "
                   f"strikes on Iran on 28 Feb 2026) is not complete by {as_of:%d %b %Y}. ")
    # Name only the episodes that lie inside the history up to the end date (the war began with the strikes of
    # 28 Feb 2026, _WAR_START is the first day of the month after).
    # For oil the largest 30-day volatility in the history is the 2020 crash (Brent 141% annualised on
    # 21 Apr 2020, WTI higher), not the 2021-22 crisis, so it is named for WTI and Brent.
    events = []
    if core['ticker'] in _OIL_TICKERS:
        if as_of >= pd.Timestamp("2021-01-01"):
            events.append("the 2020 crash")
        elif as_of >= _CRASH_START:
            events.append(f"the 2020 crash up to {as_of:%b %Y}")
    if as_of >= pd.Timestamp("2023-01-01"):
        events.append("the 2021-22 crisis")
    elif as_of >= pd.Timestamp("2021-01-01"):
        events.append(f"the 2021-22 crisis up to {as_of:%b %Y}")
    if as_of >= _WAR_START:
        events.append("the 2026 war")
    # Whose figure the percentile ranks: with a past end date the span already names the date; otherwise the
    # latest figure is today's only when it is of today's date.
    today = pd.Timestamp(berlin_today() if today is None else today).normalize()
    whose = ("the latest figure" if past_end else
             f"the latest figure ({as_of:%d %b})" if as_of.normalize() < today else "today's figure")
    listed = (", ".join(events[:-1]) + " and " + events[-1]) if len(events) > 1 else "".join(events)
    ranks = (f"The percentile ranks {whose} among every 30-day volatility {span}"
             + (f", a history that includes {listed}" if events else "")
             + "; it does not move with a rolling window.")
    out['detail'] = detail + ranks
    out['ai'] = (f"{out['headline']}; {headline2_ai.replace(' (see below)', '')}. {detail_ai + ranks}"
                 + (f" {out['roll']}" if out['roll'] else ""))
    out['prewar_ai'] = ("" if r is None else
                        "the Oct to Dec 2025 level" if before_strikes else "the pre-war level (Oct to Dec 2025)")
    return out


NEW_YORK = ZoneInfo("America/New_York")


def new_york_today(now=None):
    """Today's date in New York (midnight-normalised Timestamp). `now` is a time (any time zone; naive means UTC),
    so the rules below can be tested without the clock."""
    now = pd.Timestamp.now(tz=NEW_YORK) if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return pd.Timestamp(now.tz_convert(NEW_YORK).date())


def futures_live_from(now=None):
    """The earliest row date of a NYMEX future (TTF=F, CL=F, BZ=F and their monthly contracts) that is not yet
    treated as a close (futures_row_state): the day before today's New York date when that day is Monday to
    Thursday, else today's New York date itself. A Friday row is a close from Saturday: CME Globex trades these
    contracts from Sunday to Friday, 18:00 to 17:00 ET the next day, so nothing trades from Friday 17:00 to
    Sunday 18:00 and no evening session is written into Friday's row after Friday ends in New York."""
    today = new_york_today(now)
    before = today - pd.Timedelta(days=1)
    return before if before.dayofweek <= 3 else today


def futures_row_state(day, now=None):
    """What a futures row dated `day` holds, by today's New York date (new_york_today):
    'live'      the row is of today's New York date or later: the current session or, after the settlement, the
                evening electronic session, not the settlement.
    'unsettled' the row is of the day before today's New York date and that day is Monday to Thursday
                (futures_live_from): it may not yet be the settlement. Yahoo Finance writes the evening session into the day's row until New York midnight
                and can rewrite it afterwards (at 03:17 UTC on 23 October 2026, 23:17 on 22 October in New York, the
                22 October rows of the December WTI and Brent contracts were 100.14 and 113.81 against settlements
                of 99.47 and 113.17, reported by The Nation (Thailand) on 23 October 2026; after New York midnight
                Yahoo showed 99.47 and 113.17 and opened 23 October rows for the evening session, but that was seen
                on three downloads only, so the page does not say either way).
    'close'     an older row: a full New York day has passed after it, or it is of a Friday and today's New York
                date is Saturday or later (CME Globex: Sunday to Friday, 18:00 to 17:00 ET the next day, so there
                is no session from Friday 17:00 to Sunday 18:00; a new row opened by the Sunday evening session is
                of today's New York date or later and so is 'live').
    The Risk page's forward curve, its "Data up to" line and last_value_note all use this, so the label is
    decided in one place."""
    d = pd.Timestamp(day).normalize()
    if d >= new_york_today(now):
        return 'live'
    return 'unsettled' if d >= futures_live_from(now) else 'close'


# The wording of the three states, written once (futures_row_state).
_LIVE_WHY = ("the current session or the evening electronic session after the settlement, not the settlement")
_UNSETTLED_WHY = ("Yahoo Finance writes the evening session into the day's row until New York midnight and can "
                  "rewrite it afterwards")


def futures_curve_words(trades, front_day, settled, now=None):
    """The forward curve's wording for its contracts' last rows (`trades`, the dates of 'Last trade') and the
    front contract's (`front_day`): dict(state, title, hover, caption). `title` is the phrase on the prices in
    the chart title, with `settled` (their dates as written there): 'latest prices 22 Oct 2026', 'prices of
    22 Oct 2026 (may not yet be settlements)' or 'latest closes 22 Oct 2026'; `hover` the help of the front-back
    spread when both ends are of `front_day`; `caption` the clause on prices in the caption. The state is the one
    of the latest row (futures_row_state)."""
    days = sorted({pd.Timestamp(t).normalize() for t in trades})
    states = {futures_row_state(t, now) for t in days}
    state = futures_row_state(days[-1], now)
    front_state = futures_row_state(front_day, now)
    fd = f"{pd.Timestamp(front_day):%d %b}"
    title = {'live': f"latest prices {settled}", 'unsettled': f"prices of {settled} (may not yet be settlements)",
             'close': f"latest closes {settled}"}[state]
    # Contracts from different days: name the days whose rows may not yet be settlements, so a title over
    # "22 Oct to 23 Oct" does not call every price the latest one.
    unsettled = [t for t in days if futures_row_state(t, now) == 'unsettled']
    if unsettled and len(states) > 1:
        ud = ", ".join(f"{t:%d %b}" for t in unsettled)
        title = (f"latest prices {settled} ({ud} prices may not yet be settlements)" if state == 'live'
                 else f"prices of {settled} ({ud} prices may not yet be settlements)")
    hover = {'live': f"Latest prices of {fd} (the last row can be the evening electronic session after the "
                     "settlement, not the settlement).",
             'unsettled': f"Prices of {fd}, which may not yet be settlements: {_UNSETTLED_WHY}.",
             'close': f"Both closes of {fd}."}[front_state]
    if state == 'close':
        caption = ("at its latest close (for a finished day normally the settlement), so prices can be from "
                   "different days (see Last trade). ")
    else:
        parts = []
        if 'live' in states:
            parts.append(f"a row of {new_york_today(now):%d %b} can be {_LIVE_WHY}, so the curve can change within "
                         "the hour")
        if 'unsettled' in states:
            parts.append(f"a row of {futures_live_from(now):%d %b} may not yet be the settlement ({_UNSETTLED_WHY})")
        caption = ("at its " + ("latest price" if state == 'live' else "last price") + ": " + "; ".join(parts)
                   + ". Prices can be from different days (see Last trade). ")
    return dict(state=state, title=title, hover=hover, caption=caption)


def ttf_row_late(ttf_last, bz_close, now=None):
    """The day of the latest NYMEX Brent (BZ=F) row dated before today's New York date when it is later than
    TTF=F's last row `ttf_last`, else None: then Yahoo's TTF=F row for that day is late, not a holiday (both follow
    the NYMEX trading calendar). A BZ=F row of today's New York date is not compared: after New York midnight
    Yahoo opens it for the evening session that has just started, before TTF has traded that day."""
    days = pd.DatetimeIndex(bz_close.dropna().index).normalize()
    days = days[days < new_york_today(now)]
    if len(days) and days.max() > pd.Timestamp(ttf_last).normalize():
        return days.max()
    return None


def last_value_note(ticker, end_date, last_day, now=None):
    """The sentence on the last price of `ticker`, whose last row is `last_day`. For a future, by
    futures_row_state: 'live', the value is the latest price, which can be the evening electronic session after
    the settlement; 'unsettled', it may not yet be the settlement; 'close', ''. For the carbon ETC and other
    non-futures: a note on the latest close, or '' when the end date is more than three days before today in
    Berlin (berlin_today, the date the default end date uses). The Risk and Market pages both call this, so the
    note is written in one place."""
    if ticker.endswith("=F"):
        state = futures_row_state(last_day, now)
        if state == 'live':
            return (f" The last value, of {pd.Timestamp(last_day):%d %b}, is the latest price: it can be "
                    f"{_LIVE_WHY}, so it can change within the hour.")
        if state == 'unsettled':
            return (f" The last value, of {pd.Timestamp(last_day):%d %b}, may not yet be the settlement: "
                    f"{_UNSETTLED_WHY}.")
        return ""
    if (pd.Timestamp(berlin_today()) - pd.Timestamp(end_date)).days > 3:
        return ""
    if ticker.endswith(".L"):
        return " The last value is the latest close on Yahoo Finance (London Stock Exchange)."
    return " The last value is the latest close."


# ─── Cross-commodity returns (correlation matrix, portfolio VaR, positions) ───
def panel_tickers(carbon_short, carbon_ticker):
    """Column name -> Yahoo ticker of returns_panel, so a page can go back from a position to its prices."""
    return {"TTF Gas": "TTF=F", "WTI Oil": "CL=F", "Brent Oil": "BZ=F", carbon_short: carbon_ticker}


@st.cache_data(ttl=3600, show_spinner="Computing cross-commodity returns...")
def returns_panel(start, end, carbon_short, carbon_ticker):
    """Daily returns of TTF, WTI, Brent and carbon, one column each (not NaN-aligned).

    Callers drop NaNs over the columns they use, so a short carbon history does not
    truncate a calculation that does not involve carbon.
    """
    tickers = panel_tickers(carbon_short, carbon_ticker)
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


STRESS_WINDOW = 250      # common trading days in a candidate stress period (about 12 months)
STRESS_MAX_SPAN = 400    # calendar days those 250 days may span at most (12 months plus holiday slack)
STRESS_FROM = dt.date(2017, 1, 1)


@st.cache_data(ttl=3600, show_spinner="Searching for the stress period...")
def stress_period(panel, positions, window=STRESS_WINDOW, max_span=STRESS_MAX_SPAN):
    """The 12-month stress period of today's book and its VaR and ES, after the Basel method (BCBS d457,
    MAR33.7): the window of past data in which the current positions lose the most, equally weighted.

    `panel` holds the daily returns of the instruments (returns_panel, from STRESS_FROM to the end date) and
    `positions` maps instrument to € notional. Only days on which every instrument held has a return count, as
    for the book VaR above. A candidate is `window` such consecutive days spanning at most `max_span` calendar
    days: 250 common days of instruments on different holiday calendars normally span 360 to 380 days, while a long gap
    (the carbon ETC's suspension from June 2020 to August 2021, its sparse prices before 2018) makes the span far
    longer, so such a stretch cannot be passed off as 12 months. The stress period is the candidate with the
    largest ES 97.5% (same definition as hist_es). The ES of 250 days depends on its worst 7 days only, so
    neighbouring windows containing the same worst days share the largest ES exactly; the latest of them is
    taken (Basel asks for the stress period to be refreshed at least quarterly, and the latest moves least when
    it is).

    Returns None when no candidate qualifies, else dict(start, end, ties, first_tie_end, var95, var99, es975,
    search_from, search_to, var95_ties, n_tail), losses as positive €. The tied windows share the worst 7 days,
    so their ES and VaR 99% (between the 3rd and 4th worst day) are the same, but VaR 95% (between the 13th and
    14th worst) can differ: var95_ties is its (lowest, highest) over the tied windows. n_tail is the number of
    days the ES averages.
    """
    cols = list(positions)
    sub = panel[cols].dropna()
    if len(sub) < window:
        return None
    pnl = (sub * pd.Series(positions)).sum(axis=1)
    idx = sub.index

    def es(w):
        q = np.percentile(w, 2.5)
        return -w[w <= q].mean()

    found = _stress_search(pnl.values, idx, es, window, max_span)
    if found is None:
        return None
    ties = found['ties']
    k = int(ties[-1])
    w = pnl.iloc[k - window + 1:k + 1]
    tie_var95 = [float(hist_var(pnl.iloc[int(t) - window + 1:int(t) + 1], 0.95)) for t in ties]
    return dict(start=w.index[0], end=w.index[-1], ties=len(ties), first_tie_end=idx[int(ties[0])],
                var95=float(hist_var(w, 0.95)), var99=float(hist_var(w, 0.99)), es975=float(hist_es(w, 0.975)),
                search_from=idx[0], search_to=idx[-1], var95_ties=(min(tie_var95), max(tie_var95)),
                n_tail=int((w <= np.percentile(w, 2.5)).sum()))


def _stress_search(values, index, score, window=STRESS_WINDOW, max_span=STRESS_MAX_SPAN):
    """The window search behind stress_period and stress_period_fall, written once: every run of `window`
    consecutive observations spanning at most `max_span` calendar days is scored with `score` (larger = more
    stressed). Scores within 1e-9 (relative) of the largest count as tied. Returns None when no run qualifies,
    else dict(scores = score by window end position (NaN where the span is too long), top, ties = end positions of
    the tied windows, in date order)."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return None
    span = np.full(len(values), np.inf)
    span[window - 1:] = (index[window - 1:] - index[:len(index) - window + 1]).days
    scores = np.full(len(values), np.nan)
    for k in np.where(span <= max_span)[0]:
        scores[k] = score(values[k - window + 1:k + 1])
    if np.all(np.isnan(scores)):
        return None
    top = float(np.nanmax(scores))
    ties = np.where(scores >= top - 1e-9 * max(abs(top), 1.0))[0]
    return dict(scores=scores, top=top, ties=ties)


@st.cache_data(ttl=3600, show_spinner="Searching for the stress period...")
def stress_period_fall(changes, window=STRESS_WINDOW, max_span=STRESS_MAX_SPAN):
    """Stress period of a series of day-on-day changes that can be negative (the Power page's spread in €/MWh
    and the flexible plant's daily margin in €), by the same rule as stress_period: the run of `window`
    consecutive changes, spanning at most `max_span` calendar days, with the largest 1-in-20 fall (the 5th
    percentile of the changes, numpy's default linear interpolation, shown as a positive number), the latest of
    tied runs. `changes` runs from the start of the data to the end date chosen, so nothing after the end date
    is searched. The 5th percentile of 250 changes depends only on the 13th and 14th largest falls, so
    neighbouring windows containing the same falls tie exactly.

    Returns None when no run qualifies, else dict(start, end, fall, ties, first_tie_end, search_from, search_to,
    runner_up): runner_up is None or dict(fall, n, first_end, last_end) for the next-highest value, so the page
    can say when a window far from the stress period comes close."""
    s = changes.dropna()
    idx = s.index
    found = _stress_search(s.values, idx, lambda w: -np.percentile(w, 5), window, max_span)
    if found is None:
        return None
    scores, top, ties = found['scores'], found['top'], found['ties']
    k = int(ties[-1])
    w = s.iloc[k - window + 1:k + 1]
    rest = np.where(scores < top - 1e-9 * max(abs(top), 1.0), scores, np.nan)
    runner = None
    if not np.all(np.isnan(rest)):
        r2 = float(np.nanmax(rest))
        r2k = np.where(np.abs(rest - r2) <= 1e-9 * max(abs(r2), 1.0))[0]
        runner = dict(fall=r2, n=len(r2k), first_end=idx[int(r2k[0])], last_end=idx[int(r2k[-1])])
    return dict(start=w.index[0], end=w.index[-1], fall=top, ties=len(ties), first_tie_end=idx[int(ties[0])],
                search_from=idx[0], search_to=idx[-1], runner_up=runner)


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
GARCH_SEED = 42   # seed of the simulated shocks in garch_price_range


@st.cache_data(ttl=3600, show_spinner="Simulating price paths...")
def garch_price_range(close, horizon=5, simulations=10000, conf=0.90, drop=None):
    """Range of the price `horizon` business days ahead from a GARCH(1,1) fit.

    Fitted on daily log returns with zero mean (no view on direction) and Student-t
    shocks (fat tails). Simulated paths give, for each day ahead, the lower, median
    and upper percentile of the price. Returns None if the fit fails.

    The shocks are drawn from a Student-t distribution seeded with GARCH_SEED, so the same prices give the
    same range on every run. (arch 8.0 uses forecast(random_state=...) only for bootstrap forecasts, not for
    method='simulation', so the seed has to be given to the distribution.) The fit is the same as
    arch_model(..., mean='Zero', vol='GARCH', p=1, q=1, dist='t').

    `drop`: days whose returns are left out of the fit (the Power page's check of a contract roll day among the
    last returns); the paths still start from the last price. None leaves every return in.
    """
    close = close.where(close > 0)
    # Log returns on the series' own days; NaN across a stale gap rather than one jump.
    log_ret = (100 * np.log(close).diff()).dropna()
    if drop is not None:
        log_ret = log_ret.drop(pd.DatetimeIndex(list(drop)), errors='ignore')
    close = close.dropna()
    try:
        res = ZeroMean(log_ret, volatility=GARCH(p=1, q=1), distribution=StudentsT(seed=GARCH_SEED),
                       rescale=False).fit(disp='off')
        fc = res.forecast(horizon=horizon, method='simulation', simulations=simulations, reindex=False)
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

    TTF=F and the curve are NYMEX Dutch TTF futures: trading stops on the second London business day before
    the first calendar day of the delivery month, or the first earlier day that is both a London and a NYMEX
    business day (CME rule 1159102.E); ICE Endex TTF, the European benchmark, stops two UK business days
    before the delivery month, normally the same day. The expiry comes from contract_expiry("TTF", ...), the
    one place the rule is written, on the exchange calendars (EXPIRY_CALENDARS); the day after is the next NYMEX
    business day. Yahoo's continuous TTF=F may switch contract on either of these days, so a large move on
    them can be partly the gap between two contracts.
    """
    if len(index) == 0:
        return index
    out = set()
    months = pd.period_range(index.min().to_period('M'), index.max().to_period('M') + 1, freq='M')
    for m in months:
        expiry = contract_expiry("TTF", m.to_timestamp())
        nxt = pd.Timestamp(np.busday_offset(expiry.date(), 1, busdaycal=NYMEX_BDAYS))
        out.update([expiry.normalize(), nxt.normalize()])
    idx_norm = index.normalize()
    return index[idx_norm.isin(list(out))]


# ─── Which monthly contract a continuous front-month series is tracking ───
def contract_symbol(root, delivery):
    """Yahoo Finance symbol of a NYMEX monthly contract, e.g. ("TTF", 2026-12-01) -> TTFZ26.NYM."""
    d = pd.Timestamp(delivery)
    return f"{root}{MONTH_CODES[d.month - 1]}{d.year % 100:02d}.NYM"


def front_by_rule(root, day):
    """Delivery month (dt.date, first day) of the front contract on `day` by the exchange expiry rule: the
    earliest delivery month whose last trading day (contract_expiry) is not before `day`."""
    d = pd.Timestamp(day).normalize()
    for k in range(0, 4):
        delivery = (d.to_period('M') + k).to_timestamp()
        if not contract_expired(root, delivery, d):
            return delivery.date()
    return None


def front_month_contract(continuous, day):
    """The monthly contract that `continuous` (TTF=F, CL=F or BZ=F) tracked on `day`.

    Read from the shared download (_download_batch), the same one load_close uses, so the continuous price and
    the contract prices come from one download and nothing is downloaded again. The contract is the one whose
    close matches the continuous close on that day (match_front_contract: the same rule and tolerances as the
    Risk page's roll check). When none matches (Yahoo has dropped expired contracts, the date is older than the
    contracts downloaded, or a live price moved between rows), the delivery month comes from the exchange
    expiry rule (front_by_rule) instead, and `matched` is False so the page can say so.

    Returns dict(delivery=dt.date, symbol=str, matched=bool).
    """
    root = CURVE_ROOTS[continuous][0]
    d = pd.Timestamp(day).normalize()
    batch = shared_download(continuous, _DATA_START)
    sym = match_front_contract(batch, continuous, d) if len(batch) else None
    if sym is not None:
        return dict(delivery=contract_delivery(sym, root), symbol=sym, matched=True)
    delivery = front_by_rule(root, d)
    return dict(delivery=delivery, symbol=contract_symbol(root, delivery), matched=False)


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
    that have already expired) and downloaded in one batch. A contract is kept if
      1. it has a close within the last CURVE_FRESH_DAYS trading days of the batch (the
         dates on which any contract traded), so stale contracts drop out while a
         contract that missed only the latest day stays; and
      2. its last trading day by contract_expiry is not earlier than the batch's last
         trading day (trading_days[-1]). Yahoo keeps listing a contract for some days
         after it expires (CLX26 on 21 Oct 2026, the day after its last trading day), and
         without this rule it would stay on the curve as the front month. A contract on
         its last trading day is kept.
    contract_expiry uses exchange calendars (EXPIRY_CALENDARS); a closure missing from them (an
    unscheduled one, such as a national day of mourning) could still make an expiry one day late. As a
    safeguard for the first row only: the first contract is also dropped when it has no
    price on the batch's last trading day, a later contract has one, and its last trading
    day by the rule is that day or the next business day.
    A contract's price is its latest daily close on Yahoo Finance (for a finished day
    normally the settlement; on a trading day the latest price). Weekend rows (Sunday
    evening session) are dropped. Nothing is interpolated: missing contracts are simply
    absent. Returns a DataFrame with Contract, Delivery, Price and Last trade, sorted by
    delivery month.
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
    last_day = trading_days[-1]
    cutoff = trading_days[-min(CURVE_FRESH_DAYS, len(trading_days))]
    df = df[df['Last trade'] >= cutoff]
    df = df[~df['Delivery'].map(lambda d: contract_expired(root, d, last_day))]
    df = df.sort_values('Delivery').reset_index(drop=True)
    if len(df) > 1:
        # Safeguard for an expiry moved earlier by a holiday (see the docstring).
        priced = closes.loc[last_day].notna()
        first_exp = contract_expiry(root, df.loc[0, 'Delivery'])
        if (not bool(priced.get(df.loc[0, 'Contract'], False))
                and any(bool(priced.get(c, False)) for c in df['Contract'].iloc[1:])
                and first_exp is not None
                and 0 <= int(np.busday_count(last_day.date(), first_exp.date())) <= 1):
            df = df.iloc[1:]
    return df.head(n).reset_index(drop=True)


EXPIRY_RULES = {
    "CL": "NYMEX WTI: 3 NYMEX business days before the 25th of the month before delivery, 4 if the 25th is not a "
          "business day",
    "BZ": "Brent: last London business day of the second month before delivery, one business day earlier in "
          "December",
}

# Exchange calendars for the expiry rules (the `holidays` package): London business days are England's bank
# holidays (ICE Brent, NYMEX BZ and the London leg of NYMEX Dutch TTF count London business days), NYMEX
# business days are CME's holidays (calendar "XCME"). Checked on 2026-10-23 against CME's published last
# trading days for every listed contract (CME product calendars 425 WTI, 424 Brent Last Day Financial and 8378
# Dutch TTF: 124, 100 and 122 contracts from Nov 2026 to Feb 2037): all equal.
_CAL_YEARS = range(2000, 2041)
LONDON_BDAYS = np.busdaycalendar(holidays=sorted(holidays.UK(subdiv="ENG", years=_CAL_YEARS).keys()))
NYMEX_BDAYS = np.busdaycalendar(holidays=sorted(holidays.financial_holidays("XCME", years=_CAL_YEARS).keys()))
_LONDON_NYMEX_BDAYS = np.busdaycalendar(holidays=sorted(
    set(holidays.UK(subdiv="ENG", years=_CAL_YEARS).keys())
    | set(holidays.financial_holidays("XCME", years=_CAL_YEARS).keys())))
EXPIRY_CALENDARS = {"CL": NYMEX_BDAYS, "BZ": LONDON_BDAYS, "TTF": _LONDON_NYMEX_BDAYS}
# How the pages describe the calendars behind an expiry date, per root.
_CAL_WORDS = {"CL": "CME's holiday calendar", "BZ": "England's bank holidays (London business days)",
              "TTF": "England's bank holidays and CME's holiday calendar"}


def expiry_cal_note(root):
    """'exchange holidays from ..., checked against CME's published last trading days' for `root`."""
    return (f"exchange holidays from {_CAL_WORDS.get(root, 'weekdays only')}, checked against CME's published "
            "last trading days")


def contract_expiry(root, delivery):
    """Last trading day of a WTI (CL), Brent (BZ) or Dutch TTF (TTF) monthly contract, on the exchange
    calendars above (England's bank holidays for London business days, CME's holidays for NYMEX business days).
    This is the only place the expiry rules are written: the forward curve, the front-month label and
    ttf_expiry_days all call it.

    WTI (CME rule 200102.F): trading ends 3 business days before the 25th calendar day of the month before
    the delivery month; if the 25th is not a business day, 4 business days before it (3 before the business day
    preceding it). Brent (ICE rule, which NYMEX BZ follows, 698102.E): the last London business day of the
    second month before delivery, moved one business day earlier when that month is December. Dutch TTF (CME
    rule 1159102.E, NYMEX contract used by TTF=F and the curve): the second London business day before the first
    calendar day of the delivery month; if that day is not a NYMEX business day, the first earlier day that is
    both (ICE Endex TTF: two UK business days before the delivery month, normally the same day). None for other
    roots. Checked against CME's published calendars (see EXPIRY_CALENDARS)."""
    d = pd.Timestamp(delivery)
    if root == "TTF":
        first = d.replace(day=1).date()
        day = np.busday_offset(first, -2, roll='forward', busdaycal=LONDON_BDAYS)
        if not np.is_busday(day, busdaycal=NYMEX_BDAYS):
            day = np.busday_offset(day, 0, roll='backward', busdaycal=_LONDON_NYMEX_BDAYS)
        return pd.Timestamp(day)
    if root == "CL":
        d25 = (d - pd.DateOffset(months=1)).replace(day=25).date()
        if np.is_busday(d25, busdaycal=NYMEX_BDAYS):
            return pd.Timestamp(np.busday_offset(d25, -3, busdaycal=NYMEX_BDAYS))
        prev = np.busday_offset(d25, 0, roll='backward', busdaycal=NYMEX_BDAYS)
        return pd.Timestamp(np.busday_offset(prev, -3, busdaycal=NYMEX_BDAYS))
    if root == "BZ":
        m2 = d - pd.DateOffset(months=2)
        last = np.busday_offset((m2 + pd.offsets.MonthEnd(0)).date(), 0, roll='backward', busdaycal=LONDON_BDAYS)
        return pd.Timestamp(np.busday_offset(last, -1, busdaycal=LONDON_BDAYS) if m2.month == 12 else last)
    return None


def exchange_busday_count(root, start, end):
    """Business days from `start` (included) to `end` (excluded) on the calendar of `root`'s expiry rule
    (EXPIRY_CALENDARS); weekdays only for other roots."""
    cal = EXPIRY_CALENDARS.get(root)
    if cal is None:
        return int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date()))
    return int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date(), busdaycal=cal))


def contract_expired(root, delivery, as_of):
    """True when the contract's last trading day (contract_expiry) is earlier than `as_of`. A contract on its
    last trading day is not expired. False when the root has no rule."""
    expiry = contract_expiry(root, delivery)
    return expiry is not None and expiry.normalize() < pd.Timestamp(as_of).normalize()


def gas_season(delivery):
    """Gas-year season of a delivery month: Winter = Oct to Mar, Summer = Apr to Sep."""
    y = delivery.year % 100
    if delivery.month >= 10:
        return f"Winter {y:02d}/{(y + 1) % 100:02d}"
    if delivery.month <= 3:
        return f"Winter {(y - 1) % 100:02d}/{y:02d}"
    return f"Summer {y:02d}"
