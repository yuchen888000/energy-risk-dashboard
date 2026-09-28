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
        "name": "EU Carbon Allowance",
        "unit": "ETC price, USD (tracks EUA futures)",
        "pure_eu": True,
        "note": ("WisdomTree Carbon ETC, USD line on the LSE (the GBP line is CARP). It tracks ICE "
                 "EUA futures, but the quote is an ETC share price in USD, not €/tCO2, so its "
                 "returns also carry EUR/USD moves, roll yield and fees."),
    },
    {
        "ticker": "KRBN",
        "label": "Global Carbon (KRBN — EUA-weighted)",
        "short": "Carbon",
        "name": "Carbon Allowances (Global)",
        "unit": "ETF price, USD",
        "pure_eu": False,
        "note": ("KraneShares Global Carbon Strategy ETF. EUA carries the dominant index weight, "
                 "but California (CCA), RGGI, UK (UKA) and Washington (WCA) allowances are also "
                 "included — a correlated proxy for EU carbon, not a pure EUA price."),
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
        },
        "WTI Crude Oil": {
            "ticker": "CL=F",
            "unit": "$/barrel",
            "color": "saddlebrown",
            "keywords": ['oil', 'crude', 'WTI', 'OPEC', 'petroleum', 'barrel', 'refinery'],
            "rss_query": "crude+oil+Europe+price",
        },
        "Brent Crude Oil": {
            "ticker": "BZ=F",
            "unit": "$/barrel",
            "color": "darkred",
            "keywords": ['oil', 'crude', 'Brent', 'OPEC', 'petroleum', 'barrel', 'North Sea'],
            "rss_query": "brent+oil+Europe+price",
        },
        carbon["name"]: {
            "ticker": carbon["ticker"],
            "unit": carbon["unit"],
            "color": "seagreen",
            "keywords": ['carbon', 'ETS', 'emission', 'EU ETS', 'EUA', 'allowance', 'CBAM'],
            "rss_query": "EU+carbon+ETS+emission+price",
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
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_close(ticker, start, end):
    """Daily closing prices for one ticker as a float Series (empty if unavailable)."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


# ─── Core Calculations ───
VAR_WINDOW = 250     # trading days behind the headline VaR and Expected Shortfall


def _var_es(returns):
    """Historical VaR 95%, VaR 99% and ES 97.5% of daily returns, in % (losses negative)."""
    q95, q99, q975 = np.percentile(returns, [5, 1, 2.5])
    return q95 * 100, q99 * 100, returns[returns <= q975].mean() * 100


@st.cache_data(ttl=3600, show_spinner="Computing risk metrics...")
def compute_core(ticker, compare_ticker, start, end):
    """Returns, rolling volatility/correlation, VaR, the current regime and risk signal.

    Returns None when fewer than 30 overlapping observations are available.
    """
    df = pd.DataFrame({
        'Price': load_close(ticker, start, end),
        'Compare': load_close(compare_ticker, start, end),
    }).dropna(subset=['Price'])
    has_compare = df['Compare'].notna().sum() > 30

    df_analysis = df[['Price', 'Compare']].dropna()
    if len(df_analysis) < 30:
        return None

    df_analysis = df_analysis.copy()
    # Percentage returns are undefined across a sign change in the price level.
    # WTI (CL=F) settled negative on 2020-04-20, so non-positive prices are excluded.
    df_analysis = df_analysis[(df_analysis['Price'] > 0) & (df_analysis['Compare'] > 0)]
    df_analysis['Returns'] = df_analysis['Price'].pct_change()
    df_analysis['Compare_Returns'] = df_analysis['Compare'].pct_change()
    df_analysis['Volatility'] = df_analysis['Returns'].rolling(30).std() * 100

    # Rolling correlation on returns (not price levels) to avoid spurious correlation
    df_analysis['Rolling Correlation'] = (
        df_analysis['Returns'].rolling(30).corr(df_analysis['Compare_Returns'])
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
    risk_level, risk_color = RISK_SIGNAL[current_regime]

    return dict(
        df_analysis=df_analysis, has_compare=has_compare,
        latest_vol=latest_vol, avg_vol=avg_vol,
        returns_clean=returns_clean, var_95=var_95, var_99=var_99, es_975=es_975,
        var_days=len(var_returns), var_95_full=var_95_full, var_99_full=var_99_full,
        es_975_full=es_975_full,
        overall_corr=overall_corr, risk_level=risk_level, risk_color=risk_color,
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
        model = arch_model(garch_returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
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
    _e2_init  = float(garch_returns.iloc[-1] ** 2)

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
    long_run_vol = float(np.sqrt(_omega_b / (1 - persistence))) if persistence < 1 else None

    return dict(
        params={k: float(result.params[k]) for k in ('omega', 'alpha[1]', 'beta[1]')},
        long_run_vol=long_run_vol,
        loglikelihood=float(result.loglikelihood),
        conditional_volatility=result.conditional_volatility,
        # conditional_volatility is already a volatility series, no need for sqrt(x**2)
        current_cond_vol=float(result.conditional_volatility.iloc[-1]),
        forecast_vol=forecast_vol.values,
        forecast_5d=float(forecast_vol.iloc[4]),
        forecast_10d=float(forecast_vol.iloc[9]),
        ci_lo=np.percentile(_boot_vols, 5, axis=0),
        ci_hi=np.percentile(_boot_vols, 95, axis=0),
    )


# ─── Regime Detection (per-commodity volatility thresholds) ───
REGIME_HISTORY_START = dt.date(2010, 1, 1)
CALM_PCT, CRISIS_PCT = 50, 90     # percentiles of the commodity's own 30-day volatility
_MIN_HISTORY = 250
REGIME_COLORS = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
# The Risk Signal headline is the regime under another name, so the two cannot disagree.
RISK_SIGNAL = {'Calm': ("🟢 LOW RISK", "green"),
               'Volatile': ("🟡 MEDIUM RISK", "orange"),
               'Crisis': ("🔴 HIGH RISK", "red")}


@st.cache_data(ttl=86400, show_spinner="Computing regime thresholds...")
def regime_thresholds(ticker, fallback_vol):
    """Volatile / Crisis thresholds from the commodity's own 30-day volatility history.

    Volatile starts at the CALM_PCT-th percentile and Crisis at the CRISIS_PCT-th, both
    taken over the ticker's full history since REGIME_HISTORY_START (so they do not move
    with the selected date range). If that history is too short, `fallback_vol` (the
    30-day volatility of the selected range) is used instead.
    """
    close = load_close(ticker, REGIME_HISTORY_START, dt.date.today())
    close = close[close > 0]
    vol = (close.pct_change().rolling(30).std() * 100).dropna()
    source = "full history"
    if len(vol) < _MIN_HISTORY:
        vol, source = fallback_vol.dropna(), "selected date range only; full history unavailable"
    calm, crisis = np.percentile(vol, [CALM_PCT, CRISIS_PCT])
    return dict(calm=float(calm), crisis=float(crisis), source=source,
                start=vol.index[0], end=vol.index[-1], n=len(vol))


def regime_label(vol, thr):
    """Regime for one volatility value (%): a higher volatility never gets a lower regime."""
    if vol > thr['crisis']:
        return 'Crisis'
    if vol > thr['calm']:
        return 'Volatile'
    return 'Calm'


def regime_caption(thr):
    """One-line description of the thresholds, for page captions."""
    period = f"{thr['start']:%b %Y}–{thr['end']:%b %Y}"
    return (f"Thresholds for this commodity: Calm < {thr['calm']:.2f}% · "
            f"Volatile {thr['calm']:.2f}–{thr['crisis']:.2f}% · Crisis > {thr['crisis']:.2f}% "
            f"(the {CALM_PCT}th and {CRISIS_PCT}th percentiles of its own 30-day rolling volatility, "
            f"{period}, {thr['n']:,} days — {thr['source']}). The Risk Signal at the top of the page "
            "is the same classification: Calm = low, Volatile = medium, Crisis = high risk.")


def compute_regimes(features, thr):
    """Label each day Calm / Volatile / Crisis from its 30-day volatility alone.

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
            close = close.where(close > 0)
            returns[name] = close.pct_change()
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
    """Basel traffic light for 99% VaR over 250 days: green 0–4, yellow 5–9, red 10+.
    For shorter samples the thresholds are scaled by n / 250."""
    scale = n / 250
    if exceptions <= 4 * scale:
        return 'Green'
    if exceptions <= 9 * scale:
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
    close = close[close > 0].dropna()
    log_ret = 100 * np.log(close).diff().dropna()
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


# ─── Forward curves from individual monthly futures on Yahoo Finance ───
MONTH_CODES = "FGHJKMNQUVXZ"
# Continuous ticker → (monthly contract root on NYMEX, unit). EU carbon has no monthly
# contracts on Yahoo Finance: EUA futures are annual December contracts on ICE Endex.
CURVE_ROOTS = {"CL=F": ("CL", "$/barrel"), "BZ=F": ("BZ", "$/barrel"), "TTF=F": ("TTF", "€/MWh")}
CURVE_FRESH_DAYS = 5     # a contract needs a settlement within this many trading days


@st.cache_data(ttl=3600, show_spinner="Fetching monthly futures contracts...")
def forward_curve(root, n=12, today=None):
    """Latest prices of the next `n` monthly contracts, e.g. CLX26.NYM, CLZ26.NYM, ...

    Symbols are generated from the current month onwards (with spares for contracts
    that have already expired) and downloaded in one batch. A contract is kept if it
    has a settlement within the last CURVE_FRESH_DAYS trading days of the batch (the
    dates on which any contract settled), so expired or stale contracts drop out while
    a contract that missed only the latest day stays. Its price is its latest
    settlement. Nothing is interpolated: missing contracts are simply absent. Returns a
    DataFrame with Contract, Delivery, Price and Last trade, sorted by delivery month.
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
    """Gas-year season of a delivery month: Winter = Oct–Mar, Summer = Apr–Sep."""
    y = delivery.year % 100
    if delivery.month >= 10:
        return f"Winter {y:02d}/{(y + 1) % 100:02d}"
    if delivery.month <= 3:
        return f"Winter {(y - 1) % 100:02d}/{y:02d}"
    return f"Summer {y:02d}"
