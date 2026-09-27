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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
@st.cache_data(ttl=3600, show_spinner="Computing risk metrics...")
def compute_core(ticker, compare_ticker, start, end):
    """Returns, rolling volatility/correlation, VaR and the relative risk signal.

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
    var_95 = np.percentile(returns_clean, 5) * 100
    var_99 = np.percentile(returns_clean, 1) * 100

    returns_corr = df_analysis[['Returns', 'Compare_Returns']].dropna()
    overall_corr = returns_corr['Returns'].corr(returns_corr['Compare_Returns'])

    if latest_vol > avg_vol * 1.5:
        risk_level, risk_color = "🔴 HIGH RISK", "red"
    elif latest_vol > avg_vol:
        risk_level, risk_color = "🟡 MEDIUM RISK", "orange"
    else:
        risk_level, risk_color = "🟢 LOW RISK", "green"

    return dict(
        df_analysis=df_analysis, has_compare=has_compare,
        latest_vol=latest_vol, avg_vol=avg_vol,
        returns_clean=returns_clean, var_95=var_95, var_99=var_99,
        overall_corr=overall_corr, risk_level=risk_level, risk_color=risk_color,
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

    return dict(
        params={k: float(result.params[k]) for k in ('omega', 'alpha[1]', 'beta[1]')},
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


# ─── Hybrid Regime Detection ───
@st.cache_data(ttl=3600, show_spinner=False)
def compute_regimes(features):
    """Label each day Calm / Volatile / Crisis from 30-day volatility and correlation.

    `features` has columns Volatility and Rolling Correlation.
    """
    features = features.dropna().copy()
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features['Cluster'] = kmeans.fit_predict(scaled)

    # Step 1: Map K-Means clusters → regime labels by cluster-mean volatility.
    # Sorting by mean vol assigns: lowest cluster = Calm, middle = Volatile, highest = Crisis.
    _cluster_vol_means = features.groupby('Cluster')['Volatility'].mean().sort_values()
    _cluster_to_regime = dict(zip(_cluster_vol_means.index.tolist(), ['Calm', 'Volatile', 'Crisis']))
    features['KMeans_Regime'] = features['Cluster'].map(_cluster_to_regime)

    # Step 2: Absolute threshold labels — reliable at extremes, ambiguous near boundaries.
    def _threshold_regime(vol):
        if vol > 12:
            return 'Crisis'
        elif vol > 6:
            return 'Volatile'
        else:
            return 'Calm'

    features['Threshold_Regime'] = features['Volatility'].apply(_threshold_regime)

    # Step 3: Weighted hybrid vote.
    # - Both agree  → unanimous (high confidence)
    # - Boundary zone 4–9% vol → K-Means wins: it uses *both* volatility and correlation,
    #   so it captures regime character that pure vol thresholds miss (e.g. a low-vol period
    #   with extreme negative correlation behaving like early-stage Volatile).
    # - Outside boundary zone → threshold wins: at extremes the threshold is unambiguous
    #   and K-Means adds no useful information.
    _BOUNDARY_LO, _BOUNDARY_HI = 4.0, 9.0

    def _hybrid_regime(row):
        t, k = row['Threshold_Regime'], row['KMeans_Regime']
        if t == k:
            return t
        return k if _BOUNDARY_LO <= row['Volatility'] <= _BOUNDARY_HI else t

    features['Regime'] = features.apply(_hybrid_regime, axis=1)
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
