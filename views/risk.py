"""Risk page: market risk of the selected commodity and of a multi-commodity book.

Designed with a trading-house middle office (risk control) in mind.
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common

def _ord(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]
CARBON_LABEL = ctx.carbon["label"]
CARBON_SHORT = ctx.carbon["short"]
CARBON_NOTE = ctx.carbon["note"]
ANNUAL = np.sqrt(252)   # daily volatility x sqrt(252) = annualised volatility
# The carbon ETC tracks EUR-denominated EUA futures but is quoted in USD, so its returns
# contain the EUR/USD move. Other fallbacks (KRBN, ICLN) are USD funds.
if CARBON_TICKER == "CARB.L":
    CARBON_FX_NOTE = ("the carbon ETC (CARB.L) is quoted in USD while the EUA futures it tracks are in euros, "
                      "so its returns include the EUR/USD move")
else:
    CARBON_FX_NOTE = f"the carbon proxy ({CARBON_TICKER}) is a USD fund, not a euro EUA price"

with st.sidebar.expander("Methodology: Risk page"):
    st.markdown("""
    - **30-Day Rolling Volatility**: equal-weighted standard deviation of simple daily returns
      over 30 trading days, shown as a daily figure (annualised = daily x √252). This is a
      simplification: RiskMetrics uses exponentially weighted (EWMA) log returns.
    - **Rolling Correlation**: Pearson correlation of daily returns with the comparison asset over 30 days.
    - **Cross-Commodity Matrix**: 4×4 correlation heatmap (full period vs last 30 common trading days).
    - **Value at Risk (VaR)**: 95% and 99% historical VaR and 97.5% Expected Shortfall
      from the last 250 trading days, with the full-period figures alongside. Quantiles use
      linear interpolation between the two nearest days; ES averages the days at or below
      the 2.5% quantile.
    - **GARCH(1,1) Forecast**: Student-t GARCH; daily volatility forecast for each of the next
      10 days, and the 10-day holding-period volatility.
    - **Regime Detection**: per-commodity volatility thresholds only. Elevated above the
      75th and Extreme above the 90th percentile of the commodity's own 30-day volatility
      over the last 3 years (750 trading days), so the 2021-22 crisis does not set today's
      "normal". The headline signal is the same classification, and a lower volatility can
      never get a higher regime.
    - **Historical scenarios**: today's positions replayed through three past windows
      (Russia's invasion of Ukraine, the August 2022 gas peak, the 2026 Iran war). It shows
      what happened, not a forecast.
    """)

st.title("Market Risk")
st.markdown(f"Analyzing **{selected_commodity}** vs {compare_label}")
if ctx.is_carbon:
    st.caption(f"**{CARBON_LABEL}**: every return, volatility and VaR on this page is for this exchange-traded "
               f"product's USD price, not for EUA futures in €/tCO2. {CARBON_NOTE}")
else:
    st.caption(f"Carbon benchmark: **{CARBON_LABEL}**. {CARBON_NOTE}")

core = common.compute_core(commodity['ticker'], ctx.compare_ticker, start_date, end_date)
if core is None:
    st.warning("Please select a longer time range (at least 30 days of data required).")
    st.stop()
if not ctx.is_carbon and not core['has_compare']:
    st.caption(f"{CARBON_LABEL} has limited coverage over the selected range, "
               "widen the date window for a fuller comparison.")

_stale = [(n, a, b) for n, t in ((selected_commodity, commodity['ticker']),
                                   (compare_label, ctx.compare_ticker))
          for a, b in common.stale_periods(t, start_date, end_date)]
if _stale:
    st.caption("Excluded stale prices (5+ identical closes in a row, treated as missing, no return "
               "computed across the gap): "
               + "; ".join(f"{n} {a:%d %b %Y} to {b:%d %b %Y}" for n, a, b in _stale) + ".")

df_analysis = core['df_analysis']
latest_vol, avg_vol = core['latest_vol'], core['avg_vol']
returns_clean = core['returns_clean']
var_95, var_99, es_975 = core['var_95'], core['var_99'], core['es_975']
var_days = core['var_days']
overall_corr = core['overall_corr']
risk_level, risk_color = core['risk_level'], core['risk_color']
regime_thr, current_regime = core['regime_thr'], core['current_regime']

# ─── Section 1: Risk Signal ───
st.subheader(f"Volatility Regime: {selected_commodity}")
st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>",
            unsafe_allow_html=True)
_vctx = common.vol_context(commodity['ticker'], df_analysis['Volatility'].dropna().index[-1], float(latest_vol))
st.markdown(f"**30-day volatility {latest_vol:.2f}% daily, {latest_vol * ANNUAL:.0f}% annualised.** "
            + common.vol_context_text(_vctx, ANNUAL))
st.caption(f"The label ({current_regime}) uses the last 3 years only: Elevated above {regime_thr['elevated']:.2f}% "
           f"(the 75th percentile), Extreme above {regime_thr['extreme']:.2f}% (the 90th). Read it with the line above: "
           "a market can be ordinary for its last 3 years and still well above where it was before the war. "
           "This compares volatility with the commodity's own past. "
           "It is not a measure of overall risk: low volatility says nothing about supply, storage or position size."
           + (f" The end date is in the past, but the thresholds are always the last 3 years to today "
              f"({regime_thr['start']:%b %Y} to {regime_thr['end']:%b %Y}), so this label is applied with hindsight."
              if end_date < pd.Timestamp.today().date() else ""))
_last_px = df_analysis['Price'].dropna().index.max()
_last_cmp = df_analysis['Compare'].dropna().index.max() if df_analysis['Compare'].notna().any() else None
st.caption(f"Data up to {_last_px:%d %b %Y} for {selected_commodity}"
           + (f", {_last_cmp:%d %b %Y} for {compare_label}" if _last_cmp is not None else "")
           + ". On a trading day the last value is the latest price, not yet a settlement.")

_vol_series = df_analysis['Volatility'].dropna()
_corr_pairs = df_analysis[['Returns', 'Compare_Returns']].dropna()
_corr_30 = df_analysis['Rolling Correlation'].dropna()
_partner = ctx.compare_ticker

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Current Volatility (daily)", f"{latest_vol:.2f}%",
           help=f"Standard deviation of the last 30 daily returns. Annualised: {latest_vol * ANNUAL:.1f}% "
                "(daily x √252).")
mc2.metric("Average Volatility (daily)", f"{avg_vol:.2f}%",
           help=f"Average 30-day volatility over the selected period ({_vol_series.index.min():%d %b %Y} to "
                f"{_vol_series.index.max():%d %b %Y}). Annualised: {avg_vol * ANNUAL:.1f}%. The regime "
                "thresholds use a different period (the last 3 years), so a value below this average can "
                "still be Elevated.")
if len(_corr_pairs):
    mc3.metric(f"Correlation vs {_partner}", f"{overall_corr:.2f}",
               help=f"Pearson correlation of daily returns, {_corr_pairs.index.min():%d %b %Y} to "
                    f"{_corr_pairs.index.max():%d %b %Y} ({len(_corr_pairs):,} common days).")
    _corr_note = f"Since {_corr_pairs.index.min():%b %Y}"
    if len(_corr_30):
        _corr_note += f"; last 30 days: **{_corr_30.iloc[-1]:.2f}**"
    mc3.caption(_corr_note)
else:
    mc3.metric(f"Correlation vs {_partner}", "n/a", help="No common days with returns in the selected range.")
mc4.metric("VaR 95% (1-day)", f"{var_95:.2f}%",
           help=f"Historical, last {var_days} trading days: the daily loss exceeded on 5% of days, shown as a "
                "negative return. The € figures in Positions & Limits are shown as positive losses.")

# ─── Section 2: Price Chart ───
# Each date checked against a primary or contemporaneous source (see README).
macro_events = {
    "2021-07-14": "EU Fit for 55 package",
    "2022-02-24": "Russia invades Ukraine",
    "2022-06-03": "EU adopts Russian oil embargo",
    "2022-09-26": "Nord Stream sabotage",
    "2022-12-05": "EU ban on Russian seaborne crude applies",
    "2023-02-15": "EU gas price cap can apply",
    "2023-04-18": "EP adopts ETS reform incl. ETS 2",
    "2024-01-01": "EU ETS extended to shipping",
    "2026-01-01": "CBAM definitive period starts",
    "2026-02-28": "US-Israel strikes on Iran",
    "2026-03-02": "Qatar halts LNG; IRGC says Hormuz closed",
    "2028-01-01": "EU ETS 2 starts (postponed)",
}

st.subheader("Price Trends with Key Policy and Market Events")
fig, ax = plt.subplots(figsize=(14, 5))
# Each series is drawn from its own prices, with gaps (stale periods) left blank.
_px_line = common.break_gaps(df_analysis['Price'])
_cmp_line = common.break_gaps(common.load_close(ctx.compare_ticker, start_date, end_date))
ax.plot(_px_line.index, _px_line, color=commodity['color'],
        label=f'{selected_commodity} ({commodity["unit"]})', linewidth=1.5)
ax2 = ax.twinx()
ax2.plot(_cmp_line.index, _cmp_line, color='gray',
         label=compare_label, linewidth=1.2, alpha=0.6)

_prev_event, _flip = None, False
for date_str, label in macro_events.items():
    event_date = pd.to_datetime(date_str)
    if df_analysis.index.min() <= event_date <= df_analysis.index.max():
        # Two events a few days apart: put the labels on opposite sides of the lines.
        _flip = (not _flip) if (_prev_event is not None and (event_date - _prev_event).days < 30) else False
        ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.4)
        ax.text(event_date, ax.get_ylim()[1] * 0.9, label, rotation=90, fontsize=7, color='gray',
                va='top', ha='left' if _flip else 'right')
        _prev_event = event_date

ax.set_ylabel(f'{selected_commodity} ({commodity["unit"]})', color=commodity['color'])
ax2.set_ylabel(compare_label, color='gray')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
ax.set_title(f'{selected_commodity} vs {compare_label}')
plt.tight_layout()
st.pyplot(fig)

# ─── Section 2b: Forward Curve ───
st.subheader(f"Forward Curve: {selected_commodity}")
_MIN_CONTRACTS = 3
_MIN_SEASON = 3   # monthly contracts a season needs before its average is shown
if commodity['ticker'] not in common.CURVE_ROOTS:
    st.caption("No forward curve for carbon: Yahoo Finance has no free monthly EUA contracts (EUA futures "
               "trade on ICE Endex, with the December contracts as the benchmark), and the carbon series on this page is an "
               "exchange-traded product, not a futures strip.")
else:
    curve_root, curve_unit = common.CURVE_ROOTS[commodity['ticker']]
    curve = common.forward_curve(curve_root)
    if len(curve) < _MIN_CONTRACTS:
        st.caption(f"No forward curve shown: Yahoo Finance returned fresh prices for only {len(curve)} monthly "
                   f"{curve_root} contracts right now (at least {_MIN_CONTRACTS} needed). "
                   "Missing contracts are not filled in.")
    else:
        front, back = curve.iloc[0], curve.iloc[-1]
        n_curve = len(curve)
        first_trade, last_trade = curve['Last trade'].min(), curve['Last trade'].max()
        settled = (f"{last_trade:%d %b %Y}" if first_trade == last_trade
                   else f"{first_trade:%d %b} to {last_trade:%d %b %Y}")
        # A real monthly axis: months with no contract stay empty and the line is not drawn across them.
        labels_fc = list(curve['Delivery'])
        _all_months = pd.date_range(curve['Delivery'].min(), curve['Delivery'].max(), freq='MS')
        _line = curve.set_index('Delivery')['Price'].reindex(_all_months)
        _gaps = [f"{m:%b %y}" for m in _all_months[_line.isna()]]
        fig_fc, ax_fc = plt.subplots(figsize=(14, 3.8))
        ax_fc.plot(_all_months, _line.values, color=commodity['color'], linewidth=1.5, zorder=1)
        ax_fc.set_xticks(_all_months)
        ax_fc.set_xticklabels([f"{m:%b %y}" for m in _all_months])

        if curve_root == "TTF":
            # Gas curves are seasonal: winter delivery carries a heating premium, so front
            # vs. back month says little. Compare winter and summer contracts instead.
            curve['Season'] = curve['Delivery'].map(common.gas_season)
            seasons = (curve.groupby('Season', sort=False)
                       .agg(Price=('Price', 'mean'), Contracts=('Price', 'size'),
                            From=('Delivery', 'min'), To=('Delivery', 'max')))
            # The next winter against the summer that follows it (not a leftover summer month
            # before that winter); without one, the nearest summer on the curve.
            winter = next((k for k in seasons.index if k.startswith("Winter")), None)
            summers = [k for k in seasons.index if k.startswith("Summer")]
            summer = next((k for k in summers if winter is None or seasons.loc[k, 'From'] > seasons.loc[winter, 'From']),
                          summers[0] if summers else None)
            curve_shape = "Seasonal curve"

            # Contracts of a season still trading: a TTF month expires two business days before it starts.
            _asof = curve['Last trade'].max()

            def _open_months(key):
                _y = int(key.split()[-1].split('/')[0]) + 2000
                months = (pd.date_range(f"{_y}-10-01", f"{_y + 1}-03-01", freq='MS') if key.startswith("Winter")
                          else pd.date_range(f"{_y}-04-01", f"{_y}-09-01", freq='MS'))
                return sum(pd.Timestamp(np.busday_offset(m.date(), -2, roll='forward')) > _asof for m in months)

            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.markdown("<h3 style='color:steelblue; margin-top:0'>Seasonal curve</h3>",
                         unsafe_allow_html=True)
            _season_ok = {}
            for col_fc, key in ((fc2, winter), (fc3, summer)):
                if key is None:
                    continue
                n_season = int(seasons.loc[key, 'Contracts'])
                _season_ok[key] = n_season >= _MIN_SEASON
                if _season_ok[key]:
                    _d = curve.loc[curve['Season'] == key, 'Delivery']
                    col_fc.metric(f"{key}: {_d.min():%b %y} to {_d.max():%b %y} average ({n_season} of {_open_months(key)} contracts still trading)",
                                  f"{seasons.loc[key, 'Price']:.2f} {curve_unit}")
                else:
                    _months = ", ".join(f"{d:%b %y}" for d in curve.loc[curve['Season'] == key, 'Delivery'])
                    col_fc.metric(f"{key} average", "n/a",
                                  help=f"Only {n_season} of {_open_months(key)} monthly contracts still trading available ({_months}); "
                                       f"an average needs at least {_MIN_SEASON}.")
                    col_fc.caption(f"Only {n_season} contract{'s' if n_season > 1 else ''} ({_months}): "
                                   "no season average.")
            if winter is not None and summer is not None and _season_ok.get(winter) and _season_ok.get(summer):
                fc4.metric(f"{winter} − {summer}",
                           f"{seasons.loc[winter, 'Price'] - seasons.loc[summer, 'Price']:+.2f} {curve_unit}",
                           help="Average winter contract minus average summer contract: the seasonal premium.")
            else:
                fc4.caption(f"No winter-summer spread: each season needs at least {_MIN_SEASON} monthly "
                            "contracts. The contracts available are plotted below.")

            season_colors = ['steelblue' if k.startswith("Winter") else 'darkorange' for k in curve['Season']]
            ax_fc.scatter(labels_fc, curve['Price'], c=season_colors, s=36, zorder=2)
            ax_fc.scatter([], [], c='steelblue', label='Winter (Oct to Mar)')
            ax_fc.scatter([], [], c='darkorange', label='Summer (Apr to Sep)')
            ax_fc.legend(fontsize=8, loc='upper right')
            shape_note = (
                (f"No contract for {', '.join(_gaps)}: the gap is left empty, not interpolated. " if _gaps else "") +
                "TTF is not labelled contango or backwardation: gas curves are seasonal, with winter "
                "deliveries (Oct to Mar) usually priced above summer (Apr to Sep) because of heating demand "
                "and storage costs, so comparing the front month with the last month mostly measures "
                f"which seasons they fall in. A season average is shown only when at least {_MIN_SEASON} "
                "of its monthly contracts still trading are available. These are NYMEX Dutch TTF contracts; the "
                "European benchmark is ICE Endex TTF, whose prices can differ slightly. ")
        else:
            fb_spread = front['Price'] - back['Price']
            if abs(fb_spread) < 0.005 * front['Price']:
                curve_shape, shape_color = "Flat", "gray"
            elif fb_spread < 0:
                curve_shape, shape_color = "Contango", "steelblue"
            else:
                curve_shape, shape_color = "Backwardation", "darkorange"
            back_label = "12th month" if n_curve == 12 else f"month {n_curve} (furthest available)"

            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown(f"<h3 style='color:{shape_color}; margin-top:0'>{curve_shape}</h3>",
                         unsafe_allow_html=True)
            fc2.metric(f"Front month ({front['Delivery']:%b %y})", f"{front['Price']:.2f} {curve_unit}")
            _mixed = front['Last trade'].normalize() != back['Last trade'].normalize()
            fc3.metric(f"Front − {back_label} ({back['Delivery']:%b %y})", f"{fb_spread:+.2f} {curve_unit}",
                       help=(f"Front month close of {front['Last trade']:%d %b}, {back['Delivery']:%b %y} close of "
                             f"{back['Last trade']:%d %b}: the two ends are from different days." if _mixed
                             else f"Both closes of {front['Last trade']:%d %b}."))
            if _mixed:
                fc3.caption(f"Front {front['Last trade']:%d %b}, back {back['Last trade']:%d %b}")
            ax_fc.scatter(labels_fc, curve['Price'], color=commodity['color'], s=30, zorder=2)
            shape_note = (
                "Contango: later deliveries cost more than the front month (spread negative). "
                "Backwardation: the front month costs more (spread positive), usually a sign of tight "
                "prompt supply. The label compares only the front and the last month shown. "
                f"The continuous series in the price chart ({commodity['ticker']}) can switch to the next "
                "contract a few days before the front contract expires, so its last price can differ "
                "from the first contract on this curve. ")

        for x_fc, y_fc in zip(labels_fc, curve['Price']):
            ax_fc.annotate(f"{y_fc:.2f}", (x_fc, y_fc), textcoords='offset points', xytext=(0, 7),
                           ha='center', fontsize=7)
        ax_fc.set_ylabel(curve_unit)
        ax_fc.set_xlabel('Delivery month')
        ax_fc.set_title(f"{selected_commodity} futures curve, latest closes {settled}: {curve_shape.lower()}")
        plt.tight_layout()
        st.pyplot(fig_fc)

        with st.expander("Contracts used"):
            st.dataframe(curve.assign(Delivery=curve['Delivery'].map(lambda d: f"{d:%b %Y}"),
                                      **{'Last trade': curve['Last trade'].dt.strftime('%Y-%m-%d')}),
                         width="stretch", hide_index=True)
        st.caption(
            shape_note
            + f"Monthly NYMEX contracts ({curve_root}<month code><year>.NYM) from Yahoo Finance, delayed. "
            f"A contract is shown if it has a daily close within the last {common.CURVE_FRESH_DAYS} trading "
            "days, at its latest close (for a finished day normally the settlement; a value for today is "
            "the latest price), so prices can be from different days (see Last trade); older or missing "
            f"contracts are left out and nothing is interpolated. {n_curve} of 12 contracts available."
        )

# ─── Section 3: Volatility & Correlation ───
vcol1, vcol2 = st.columns(2)
with vcol1:
    st.subheader("30-Day Rolling Volatility")
    _vol_line = common.break_gaps(df_analysis['Volatility'])
    fig_v, ax_v = plt.subplots(figsize=(7, 3.5))
    ax_v.plot(_vol_line.index, _vol_line, color=commodity['color'], linewidth=1)
    ax_v.set_ylabel('Daily volatility (%)')
    ax_v.set_title(f'{selected_commodity}: 30-day rolling volatility (daily; x √252 to annualise)', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_v)
with vcol2:
    st.subheader("30-Day Rolling Correlation")
    _corr_line = common.break_gaps(df_analysis['Rolling Correlation'])
    fig_c, ax_c = plt.subplots(figsize=(7, 3.5))
    ax_c.plot(_corr_line.index, _corr_line, color='gray', linewidth=1)
    ax_c.axhline(0, color='black', linewidth=0.5)
    ax_c.set_ylim(-1, 1)
    ax_c.set_ylabel('Correlation of daily returns')
    ax_c.set_title(f'{selected_commodity} vs {compare_label}: 30-day rolling correlation', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_c)
st.caption("Gaps in the lines are periods without usable prices (for example the stale carbon quote), "
           "left blank rather than joined by a straight line.")


# ─── Section 3b: Cross-Commodity Correlation Matrix ───
st.subheader("Cross-Commodity Correlation Matrix")
st.write("How do gas, oil and carbon move relative to each other, over the whole period and recently?")

_panel = common.returns_panel(start_date, end_date, CARBON_SHORT, CARBON_TICKER)
if _panel is not None and _panel.dropna().shape[0] > 30:
    _returns_df = _panel.dropna()
    _recent_df = _returns_df.tail(30)
    corr_full = _returns_df.corr().round(3)
    corr_30d = _recent_df.corr().round(3)
else:
    corr_full, corr_30d = None, None


def _heatmap(corr, title):
    fig_h, ax_h = plt.subplots(figsize=(5, 5))
    im = ax_h.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax_h.set_xticks(range(len(corr.columns)))
    ax_h.set_yticks(range(len(corr.columns)))
    ax_h.set_xticklabels(corr.columns, fontsize=9, rotation=45, ha='right')
    ax_h.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax_h.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=10,
                      fontweight='bold', color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax_h, shrink=0.8)
    ax_h.set_title(title, fontsize=10)
    fig_h.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    return fig_h


if corr_full is not None:
    cm1, cm2 = st.columns(2)
    _full_range = f"{_returns_df.index.min():%d %b %Y} to {_returns_df.index.max():%d %b %Y}"
    _recent_range = f"{_recent_df.index.min():%d %b %Y} to {_recent_df.index.max():%d %b %Y}"
    with cm1:
        st.write(f"**Full period: {_full_range}** ({len(_returns_df):,} common days)")
        st.pyplot(_heatmap(corr_full, f"Full period, {_full_range}"))
    with cm2:
        st.write(f"**Last 30 common trading days: {_recent_range}**")
        st.pyplot(_heatmap(corr_30d, f"Last 30 common days, {_recent_range}"))

    st.caption("Green = positive correlation (move together). Red = negative (move inversely). "
               "Only days on which all four series have a return are used. Compare full period with the "
               "last 30 days to see whether relationships have shifted. The series do not close at the same "
               "time (TTF and the carbon ETC on European hours, Brent and WTI in the New York afternoon), so "
               "news after the European close reaches oil the same day and gas and carbon the next day, "
               "which lowers same-day correlations between the two groups.")
else:
    st.info("Not enough data to compute cross-commodity correlations.")


# ─── Section 4: Value at Risk ───
st.subheader(f"Value at Risk (VaR): {selected_commodity}")
st.write(f"Historical simulation, 1-day horizon, per unit of a long {selected_commodity} position. "
         "VaR 95% is the daily loss exceeded on 5% of days (about 1 day in 20) and VaR 99% the loss exceeded "
         "on 1% of days; neither is the worst case (see Max daily loss and Expected Shortfall). "
         f"Headline figures use the last {var_days} trading days.")

_first, _last = returns_clean.index.min(), returns_clean.index.max()
# The dates, not a named event: the worst days differ by commodity (Brent's is in 2020).
full_label = f"full period ({_first:%b %Y} to {_last:%b %Y})"
_recent_ret = returns_clean.tail(var_days)


_n_tail = int((_recent_ret <= np.percentile(_recent_ret, 2.5)).sum())

vc1, vc2, vc3, vc4 = st.columns(4)
for _col, _name, _recent, _full, _help in (
        (vc1, "VaR 95%", var_95, core['var_95_full'], "Loss exceeded on 5% of days."),
        (vc2, "VaR 99%", var_99, core['var_99_full'], "Loss exceeded on 1% of days."),
        (vc3, "ES 97.5%", es_975, core['es_975_full'],
         f"Average return of the {_n_tail} days (of {var_days}) at or below the 2.5% quantile.")):
    _col.metric(_name, f"{_recent:.2f}%",
                help=f"Last {var_days} trading days, 1-day horizon. {_help} Quantiles use linear "
                     "interpolation between the two nearest days.")
    _col.caption(f"{full_label}: **{_full:.2f}%**")

_raw_px = common.load_close(commodity['ticker'], start_date, end_date)
_nonpos = _raw_px[_raw_px <= 0]
if len(_nonpos):
    st.caption(f"⚠️ {selected_commodity} settled at or below zero on "
               + ", ".join(f"{d:%d %b %Y} ({v:.2f})" for d, v in _nonpos.items())
               + ". A percentage return is not defined across a zero or negative price, so the returns into and "
                 "out of those days are left out, and the full-period max daily loss, VaR and ES do not include "
                 "that move.")

_worst_recent_day = _recent_ret.idxmin()
_worst_full_day = returns_clean.idxmin()
vc4.metric("Max daily loss", f"{_recent_ret.min() * 100:.2f}%",
           help=f"Worst daily return in the last {var_days} trading days, on {_worst_recent_day:%d %b %Y}.")
vc4.caption(f"on {_worst_recent_day:%d %b %Y}; {full_label}: **{returns_clean.min() * 100:.2f}%** "
            f"({_worst_full_day:%d %b %Y})")
st.caption(f"Expected Shortfall (ES) 97.5% is the average return on the days at or below the 2.5% quantile "
           f"({_n_tail} of the last {var_days} days): the average loss in the worst 2.5% of days. The Basel Committee "
           "chose ES 97.5% for FRTB because it gives \"a broadly similar level of risk capture as the existing 99th "
           "percentile VaR threshold\" (BCBS, Fundamental review of the trading book, October 2013, p. 18)"
           + (f". Here ES averages the worst {_n_tail} days while VaR 99% lies between the {int((len(_recent_ret) - 1) * 0.01) + 1}"
              f"{_ord(int((len(_recent_ret) - 1) * 0.01) + 1)} and {int((len(_recent_ret) - 1) * 0.01) + 2}"
              f"{_ord(int((len(_recent_ret) - 1) * 0.01) + 2)} worst, so "
              "ES 97.5% can come out slightly below VaR 99%, as it does now; that is not a calculation error."
              if abs(es_975) < abs(var_99) else "."))


_brent_roll_days = common.brent_roll_days


if commodity['ticker'] == "BZ=F":
    _rolls = _brent_roll_days(df_analysis['Price'].dropna().index)
    _ex_roll = _recent_ret[~_recent_ret.index.isin(_rolls)]
    if _worst_recent_day in _rolls:
        st.caption(f"⚠️ {_worst_recent_day:%d %b %Y}, the worst day, is a Brent roll day: the ICE Brent front "
                   "contract stops trading on the last business day of the month before, so on this day the "
                   "continuous series switches to the next contract, and part or all of the move can be the "
                   "price gap between the two contracts, not a real loss. Excluding roll days, the worst day "
                   f"was {_ex_roll.min() * 100:.2f}% on {_ex_roll.idxmin():%d %b %Y}.")
    else:
        st.caption(f"The worst day ({_worst_recent_day:%d %b %Y}) is not a Brent roll day (the first trading "
                   "day of a month, or the last trading day of December).")
if commodity['ticker'] == "TTF=F":
    _ttf_exp = common.ttf_expiry_days(df_analysis['Price'].dropna().index)
    _flag = [d for d in (_worst_recent_day, _worst_full_day) if d in _ttf_exp]
    if _flag:
        st.caption("⚠️ " + " and ".join(sorted({f"{d:%d %b %Y}" for d in _flag})) + " is on or just after a TTF "
                   "front-month expiry (ICE Endex: trading stops two UK business days before the delivery month), "
                   "so part of that day's move may be the gap between two contracts if the continuous series "
                   "switched on it. Which contract Yahoo's series followed that day is not verified.")
if commodity['ticker'].endswith("=F"):
    st.caption(f"`{commodity['ticker']}` is Yahoo Finance's continuous front-month future. When the front "
               "contract rolls to the next month, the series switches contract and that day's return is "
               "not a real price move: when the next contract trades several dollars below the expiring one, "
               "the series shows a drop that no contract actually had. Roll days are not removed, so they can "
               "distort the max daily loss, the tails, the VaR and the backtest exceptions. Desks use a "
               "back-adjusted series built from individual contracts.")

fig_var, (ax_hist, ax_ts) = plt.subplots(1, 2, figsize=(14, 4))

ax_hist.hist(_recent_ret * 100, bins=50, color=commodity['color'], alpha=0.7, edgecolor='white')
ax_hist.axvline(x=var_95, color='red', linewidth=2, linestyle='--',
                label=f'95% VaR: {var_95:.2f}%')
ax_hist.axvline(x=var_99, color='darkred', linewidth=2, linestyle=':',
                label=f'99% VaR: {var_99:.2f}%')
ax_hist.axvline(x=es_975, color='black', linewidth=1.5, linestyle='-.',
                label=f'97.5% ES: {es_975:.2f}%')
ax_hist.set_xlabel('Daily Returns (%)')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title(f'{selected_commodity} Daily Returns, last {var_days} trading days')
ax_hist.legend(fontsize=8)

rolling_var = common.break_gaps(returns_clean.rolling(60).quantile(0.05) * 100)
ax_ts.plot(rolling_var.index, rolling_var, color='red', linewidth=1, alpha=0.8)
ax_ts.fill_between(rolling_var.index, rolling_var, 0, alpha=0.15, color='red')
ax_ts.set_ylabel('VaR (95%, daily %)')
ax_ts.set_title('60-Day Rolling VaR')
ax_ts.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
st.pyplot(fig_var)


def _backtest_rows(bt, n):
    rows = []
    for conf, col in [(0.95, 'Exc95'), (0.99, 'Exc99')]:
        x = int(bt[col].sum())
        lr, p_val = common.kupiec_pof(n, x, 1 - conf)
        rows.append({'VaR level': f"{conf:.0%}", 'Exceptions': x,
                     'Expected': round(n * (1 - conf), 1), 'Kupiec LR': round(lr, 2),
                     'p-value': round(p_val, 3),
                     'Kupiec result': 'Reject (miscalibrated)' if p_val < 0.05 else 'Accept'})
    return rows


# Backtest of the headline VaR for this commodity alone (per unit of exposure).
_bt = common.var_backtest(returns_clean, window=var_days, test_days=250)
if len(_bt) >= 100:
    _n = len(_bt)
    _rows = _backtest_rows(_bt, _n)
    _zone = common.basel_zone(_rows[1]['Exceptions'], _n)
    _zone_txt = (f"Basel traffic light **{_zone}** ({_rows[1]['Exceptions']} exceptions at 99%)." if _zone
                 else f"Basel traffic light not shown: it is defined for 250 backtest days and only {_n} are "
                      "available; widen the date range.")
    _kup = "; ".join(f"Kupiec at {row['VaR level']}: {'rejected' if row['p-value'] < 0.05 else 'not rejected'} "
                     f"({row['Exceptions']} exceptions vs {row['Expected']} expected, p = {row['p-value']:.3f})"
                     for row in _rows)
    st.write(f"**VaR Backtest ({selected_commodity}), last {_n} trading days "
             f"({_bt.index.min():%d %b %Y} to {_bt.index.max():%d %b %Y}):** " + _zone_txt + " " + _kup + ".")
    st.dataframe(pd.DataFrame(_rows).set_index('VaR level'), width="stretch")
    st.caption(f"Each day's VaR is estimated only from the {var_days} trading days before it (no look-ahead) "
               "and compared with that day's return. Kupiec also rejects too few exceptions. Basel zones: "
               "green 0 to 4, yellow 5 to 9, red 10 or more exceptions at 99% over 250 days.")
else:
    st.caption(f"VaR backtest needs at least {common.VAR_WINDOW + 100} trading days of history; widen the date range.")


# ─── Section 4b: GARCH ───
st.subheader("GARCH Volatility Forecast")
st.write(f"Forward-looking volatility prediction for {selected_commodity} using GARCH(1,1) "
         "with Student-t shocks (fat tails)")

garch = common.fit_garch(returns_clean)
if garch is not None:
    forecast_vol = pd.Series(garch['forecast_vol'])
    current_cond_vol = garch['current_cond_vol']
    params = garch['params']
    cum_10d = garch['forecast_cum_10d']

    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Current GARCH Vol (daily)", f"{current_cond_vol:.2f}%",
               help=f"Fitted conditional volatility on the last day. Annualised: {current_cond_vol * ANNUAL:.1f}%.")
    gc2.metric("Daily vol forecast, day 5", f"{forecast_vol.iloc[4]:.2f}%",
               help="Forecast volatility of the single daily return 5 trading days ahead, not of a 5-day period.")
    gc3.metric("Daily vol forecast, day 10", f"{forecast_vol.iloc[9]:.2f}%",
               help="Forecast volatility of the single daily return 10 trading days ahead, not of a 10-day period.")
    gc4.metric("10-day holding-period vol", f"{cum_10d:.2f}%",
               help="Volatility of the cumulative return over the next 10 trading days: square root of the sum "
                    "of the 10 daily forecast variances.")

    fig_garch, (ax_cv, ax_fc) = plt.subplots(1, 2, figsize=(14, 4))

    cond_vol = common.break_gaps(garch['conditional_volatility'])
    ax_cv.plot(cond_vol.index, cond_vol, color='purple', linewidth=0.8, alpha=0.8)
    ax_cv.set_ylabel('Conditional Volatility (daily %)')
    ax_cv.set_title('GARCH(1,1) Conditional Volatility')
    ax_cv.fill_between(cond_vol.index, cond_vol, 0, alpha=0.1, color='purple')

    _ci_lo, _ci_hi = garch['ci_lo'], garch['ci_hi']

    forecast_days = list(range(1, 11))
    ax_fc.plot(forecast_days, forecast_vol.values, color='purple', linewidth=2, marker='o', markersize=5)
    ax_fc.fill_between(forecast_days, _ci_lo, _ci_hi,
                       alpha=0.18, color='purple', label='Simulated 90% range of future volatility (parameters fixed)')
    ax_fc.set_xlabel('Days Ahead')
    ax_fc.set_ylabel('Forecast Volatility (daily %)')
    ax_fc.set_title('Daily volatility forecast, days 1 to 10')
    ax_fc.set_xticks(forecast_days)
    ax_fc.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_garch)

    with st.expander("GARCH(1,1) Model Parameters"):
        st.write(f"Fitted on daily returns from {returns_clean.index.min():%d %b %Y} to "
                 f"{returns_clean.index.max():%d %b %Y}.")
        st.write(f"**omega (ω):** {params['omega']:.6f}")
        st.write(f"**alpha (α):** {params['alpha[1]']:.4f} (reaction to recent shocks)")
        st.write(f"**beta (β):** {params['beta[1]']:.4f} (persistence of volatility)")
        persistence = params['alpha[1]'] + params['beta[1]']
        st.write(f"**α + β = {persistence:.4f}**, "
                 f"{'high persistence (close to 1)' if persistence > 0.95 else 'moderate persistence'}")
        if garch.get('nu') is not None:
            st.write(f"**Student-t degrees of freedom (ν):** {garch['nu']:.2f} (lower means fatter tails)")
        st.write(f"**Reference level: sample average 30-day volatility = {avg_vol:.2f}%** "
                 "(daily), over the selected range. (The Market page's GARCH check uses the 3-year average instead.)")
        if garch['garch_long_run_ok']:
            st.write(f"GARCH long-run volatility √(ω / (1 − α − β)) = {garch['garch_long_run']:.2f}% "
                     "(daily), the level the forecast reverts to.")
        else:
            st.write(f"The GARCH long-run volatility √(ω / (1 − α − β)) is not shown: with "
                     f"α + β = {garch['persistence']:.4f} the formula divides by almost zero and is unstable.")
        st.write(f"**Log-Likelihood:** {garch['loglikelihood']:.2f}")
        st.caption("Shaded band: 500 paths simulated by resampling the standardised residuals through the GARCH "
                   "recursion with the fitted parameters held fixed; 5th to 95th percentile of the future "
                   "conditional volatility. It shows the spread of future volatility under this model, not "
                   "uncertainty in the estimated parameters.")

else:
    st.caption("GARCH forecast unavailable for this date range: try a longer window.")


# ─── Section 5: Market Regime ───
st.subheader("Market Regime")
st.write("Each day is labelled by its 30-day volatility against this commodity's own thresholds")

features = common.compute_regimes(df_analysis[['Volatility', 'Rolling Correlation']], regime_thr)
regime_color = common.REGIME_COLORS[current_regime]
st.markdown(f"<h3 style='color:{regime_color}'>Current Market Regime: {current_regime}</h3>",
            unsafe_allow_html=True)
st.caption(common.regime_caption(regime_thr))

regime_stats = features.groupby('Regime').agg(
    Days=('Volatility', 'count'),
    Avg_Volatility=('Volatility', 'mean'),
    Avg_Correlation=('Rolling Correlation', 'mean')
).round(2)
regime_stats.columns = ['Trading Days', 'Avg daily volatility (%)', f'Avg correlation vs {_partner}']

rcol1, rcol2 = st.columns([2, 1])
with rcol1:
    fig2, ax3 = plt.subplots(figsize=(12, 4))
    for regime, group in features.groupby('Regime'):
        ax3.scatter(group.index, group['Volatility'],
                   c=common.REGIME_COLORS[regime], label=regime, alpha=0.5, s=10)
    ax3.axhline(y=regime_thr['elevated'], color='orange', linewidth=1, linestyle='--', alpha=0.5,
                label=f"Elevated threshold ({regime_thr['elevated']:.2f}%, p{common.ELEVATED_PCT})")
    ax3.axhline(y=regime_thr['extreme'], color='red', linewidth=1, linestyle='--', alpha=0.5,
                label=f"Extreme threshold ({regime_thr['extreme']:.2f}%, p{common.EXTREME_PCT})")
    ax3.set_ylabel('30-day volatility (daily %)')
    ax3.set_title(f'{selected_commodity}: Market Regime Detection')
    ax3.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig2)
with rcol2:
    st.write("**Regime Statistics:**")
    st.dataframe(regime_stats, width="stretch")
st.caption(f"Every day in the chart is labelled with today's thresholds (from "
           f"{regime_thr['start']:%b %Y} to {regime_thr['end']:%b %Y}), so the labels for earlier years are "
           "applied with hindsight: they are not what a desk would have seen at the time.")


# ─── Section 5a: Positions & Limits ───
st.subheader("Positions & Limits")
st.write("Risk of a book of positions in euros: VaR, Expected Shortfall, limit usage, a VaR backtest and "
         "a replay of past market shocks.")

book_panel = common.returns_panel(start_date, end_date, CARBON_SHORT, CARBON_TICKER)
_LOOKBACK = 250   # days of history behind each VaR estimate
_TEST_DAYS = 250  # days in the backtest
positions = {}

if book_panel is None:
    st.info("Not enough multi-commodity data to compute position risk.")
else:
    st.write("**Positions** (€m notional; positive = long, negative = short):")
    _defaults = {"TTF Gas": 10.0, "WTI Oil": 0.0, "Brent Oil": -5.0, CARBON_SHORT: 3.0}
    _pos_cols = st.columns(len(book_panel.columns) + 1)
    positions_m = {}
    for _col, _name in zip(_pos_cols, book_panel.columns):
        positions_m[_name] = _col.number_input(f"{_name} (€m)", value=_defaults.get(_name, 0.0),
                                               step=0.5, format="%.1f", key=f"pos_{_name}")
    var_limit_m = _pos_cols[-1].number_input("VaR 95% limit (€m)", min_value=0.1, value=1.5,
                                             step=0.1, format="%.1f", key="var_limit")

    positions = {k: v * 1e6 for k, v in positions_m.items() if v != 0}
    book_returns = book_panel[list(positions)].dropna() if positions else None

    if not positions:
        st.info("Enter at least one non-zero position.")
    elif len(book_returns) < _LOOKBACK + 50:
        st.info(f"Position risk needs at least {_LOOKBACK + 50} days of overlapping history for "
                "the instruments held: widen the date range.")
    else:
        pnl = common.book_pnl(book_returns, positions)
        recent = pnl.tail(_LOOKBACK)
        book_var95 = common.hist_var(recent, 0.95)
        book_var99 = common.hist_var(recent, 0.99)
        book_es975 = common.hist_es(recent, 0.975)
        usage = book_var95 / (var_limit_m * 1e6)

        if usage > 1:
            usage_status, usage_color = "🔴 LIMIT BREACH", "red"
        elif usage >= 0.8:
            usage_status, usage_color = "🟡 NEAR LIMIT", "orange"
        else:
            usage_status, usage_color = "🟢 WITHIN LIMIT", "green"

        _n_tail_book = int((recent <= np.percentile(recent, 2.5)).sum())
        pl1, pl2, pl3, pl4 = st.columns(4)
        pl1.metric("VaR 95% (1-day)", f"€{book_var95 / 1e6:,.2f}m",
                   help="Daily book loss exceeded on 5% of the last 250 days.")
        pl2.metric("VaR 99% (1-day)", f"€{book_var99 / 1e6:,.2f}m",
                   help="Daily book loss exceeded on 1% of the last 250 days.")
        pl3.metric("Expected Shortfall 97.5%", f"€{book_es975 / 1e6:,.2f}m",
                   help=f"Average loss on the {_n_tail_book} days (of {len(recent)}) at or below the 2.5% quantile, "
                        "1-day horizon. 97.5% is the confidence level Basel FRTB uses; FRTB also scales by "
                        "liquidity horizon and a stress period, which this does not.")
        pl4.metric("Limit usage (VaR 95%)", f"{usage:.0%}",
                   help=f"VaR 95% of €{book_var95 / 1e6:,.2f}m against a limit of €{var_limit_m:,.1f}m.")
        st.markdown(f"<span style='color:{usage_color}; font-weight:bold'>{usage_status}</span>, "
                    f"€{book_var95 / 1e6:,.2f}m of €{var_limit_m:,.1f}m used",
                    unsafe_allow_html=True)
        st.progress(min(usage, 1.0))

        # Standalone VaR per position vs the book: the gap is the diversification benefit
        standalone = {k: common.hist_var(book_returns[k].tail(_LOOKBACK) * v, 0.95)
                      for k, v in positions.items()}
        contrib_df = pd.DataFrame({
            'Position (€m)': [positions[k] / 1e6 for k in positions],
            'Standalone VaR 95% (€m)': [standalone[k] / 1e6 for k in positions],
        }, index=list(positions)).round(2)
        contrib_df['Position (€m)'] = contrib_df['Position (€m)'].map(lambda v: f"{v:+.1f}")
        contrib_df.loc['Book (diversified)'] = ["", round(book_var95 / 1e6, 2)]
        st.dataframe(contrib_df, width="stretch")
        _gross = sum(abs(v) for v in positions.values()) / 1e6
        _net = sum(positions.values()) / 1e6
        st.caption(f"Book size: gross €{_gross:,.1f}m (sum of absolute positions), net €{_net:+,.1f}m (longs "
                   "minus shorts). Netting notionals across different commodities has no risk meaning; the "
                   "book VaR above does the netting through the joint daily returns. Diversification benefit: "
                   f"€{(sum(standalone.values()) - book_var95) / 1e6:,.2f}m (sum of standalone VaRs minus book VaR).")

        # ── VaR backtest ──
        bt = common.var_backtest(pnl, window=_LOOKBACK, test_days=_TEST_DAYS)
        n_bt = len(bt)
        st.write(f"**VaR Backtest, last {n_bt} trading days on which every instrument held has a return "
                 f"({bt.index.min():%d %b %Y} to {bt.index.max():%d %b %Y})**. Because only common trading days "
                 "count, this window can start earlier than the single-commodity backtest above.")
        rows = _backtest_rows(bt, n_bt)
        exc99 = rows[1]['Exceptions']
        zone = common.basel_zone(exc99, n_bt)
        if zone is not None:
            zone_color = {'Green': 'green', 'Yellow': 'orange', 'Red': 'red'}[zone]
            st.markdown(f"<h4 style='color:{zone_color}'>Basel traffic light: {zone} "
                        f"({exc99} exceptions at 99% over {n_bt} days)</h4>", unsafe_allow_html=True)
        else:
            st.caption(f"Basel traffic light not shown: it is defined for 250 backtest days and only {n_bt} are "
                       "available; widen the date range.")
        st.dataframe(pd.DataFrame(rows).set_index('VaR level'), width="stretch")

        fig_bt, ax_bt = plt.subplots(figsize=(14, 4))
        bar_colors_bt = np.where(bt['Exc99'], 'red', np.where(bt['Exc95'], 'orange', 'lightgray'))
        ax_bt.bar(bt.index, bt['PnL'] / 1e6, color=bar_colors_bt, width=1.0)
        ax_bt.plot(bt.index, -bt['VaR95'] / 1e6, color='orange', linewidth=1, linestyle='--', label='−VaR 95%')
        ax_bt.plot(bt.index, -bt['VaR99'] / 1e6, color='red', linewidth=1.2, label='−VaR 99%')
        ax_bt.axhline(0, color='black', linewidth=0.5)
        ax_bt.set_ylabel('Daily P&L (€m)')
        ax_bt.set_title('Hypothetical daily P&L vs prior-day VaR (orange = beyond VaR 95%, red = beyond VaR 99%)')
        ax_bt.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bt)

        st.caption(
            f"Historical simulation on the last {_LOOKBACK} trading days, 1-day horizon. "
            "Backtest is hypothetical: today's positions applied to past returns, each day's VaR "
            f"estimated only from the {_LOOKBACK} days before it. Positions are held static. Returns are "
            "taken in each instrument's quote currency and applied to the € notional, so EUR/USD is ignored "
            f"for Brent and WTI; {CARBON_FX_NOTE}. Kupiec also rejects too few exceptions (an "
            "over-conservative model). Basel zones: green 0 to 4, yellow 5 to 9, red 10 or more exceptions at 99% over "
            "250 days."
        )


# ─── Section 5b: Historical scenario replay (uses the positions above) ───
# Fixed past windows. Each starts at the close before the first day listed and ends at the
# close of the last day. Dates are checked against prices and news sources (see README).
SCENARIOS = [
    {"name": "Russia invades Ukraine", "first": "2022-02-24", "last": "2022-03-08",
     "why": "from the invasion on 24 Feb 2022 to 8 Mar 2022, the day the US announced a ban on Russian oil "
            "imports (Brent's highest close of the episode; TTF's was on 7 Mar)"},
    {"name": "August 2022 gas peak", "first": "2022-08-01", "last": "2022-08-26",
     "why": "from the first trading day of August 2022 to 26 Aug 2022, the record TTF front-month close (about "
            "€339/MWh) after Gazprom announced a three-day Nord Stream 1 shutdown"},
    {"name": "2026 Iran war: Qatar LNG halt, Hormuz", "first": "2026-03-02", "last": "2026-03-13",
     "why": "the first two trading weeks after the US-Israeli strikes on Iran (Sat 28 Feb 2026): QatarEnergy "
            "halted LNG output and an IRGC official said Hormuz was closed on 2 Mar"},
]
_SCEN_FROM = dt.date(2022, 1, 1)
_SCEN_TO = dt.date(2026, 3, 31)


def replay_scenario(panel, positions, first, last):
    """Replay static positions through one past window.

    Each position is entered at the close before `first` and held unchanged (same quantity)
    to the close of `last`. Returns a dict with per-instrument cumulative return and € P&L,
    the book P&L, and the book's worst day; an instrument with no or too few returns in the
    window is reported as missing.
    """
    window = panel.loc[pd.Timestamp(first):pd.Timestamp(last)]
    window = window.dropna(how='all')
    base = panel.index[panel.index < pd.Timestamp(first)]
    out = {"days": len(window), "base": base[-1] if len(base) else None,
           "end": window.index.max() if len(window) else None, "inst": {}}
    if not len(window):
        return out
    value_paths = []
    for name in panel.columns:
        r = window[name]
        n_ret = int(r.notna().sum())
        # Fewer returns than 70% of the window's trading days: a gap (stale quote, missing data),
        # so the cumulative return would be incomplete. A single holiday is fine: returns are
        # computed on each series' own trading days, so the next return covers it.
        if n_ret == 0 or n_ret < 0.7 * len(window):
            out["inst"][name] = None
            continue
        growth = (1 + r.fillna(0)).cumprod()
        cum = float(growth.iloc[-1] - 1)
        pos = positions.get(name, 0.0)
        out["inst"][name] = {"ret": cum, "pnl": pos * cum, "pos": pos}
        if pos:
            # Value of the held position each day; its day-on-day change is that day's P&L.
            value = pos * growth
            value_paths.append(value.diff().fillna(value.iloc[0] - pos))
    held = [k for k in positions if positions[k]]
    missing_held = [k for k in held if out["inst"].get(k) is None]
    out["missing_held"] = missing_held
    if value_paths and not missing_held:
        daily = pd.concat(value_paths, axis=1).sum(axis=1)
        out["book"] = float(daily.sum())
        out["worst_day"], out["worst_pnl"] = daily.idxmin(), float(daily.min())
    return out


st.subheader("Historical Scenario Replay")
st.write("What would the positions above have made or lost if held through three past shocks?")
if not positions:
    st.info("Enter at least one non-zero position above to replay historical scenarios.")
else:
    _scen_start = min(start_date, _SCEN_FROM)
    _scen_end = max(end_date, _SCEN_TO)
    scen_panel = common.returns_panel(_scen_start, _scen_end, CARBON_SHORT, CARBON_TICKER)
    if scen_panel is None:
        st.info("Not enough multi-commodity data to replay the scenarios.")
    else:
        _summary, _detail, _notes = [], [], []
        _brent_rolls = _brent_roll_days(scen_panel['Brent Oil'].dropna().index) \
            if 'Brent Oil' in scen_panel else pd.DatetimeIndex([])
        for sc in SCENARIOS:
            res = replay_scenario(scen_panel, positions, sc["first"], sc["last"])
            if not res["days"] or res["base"] is None:
                _summary.append({"Scenario": sc["name"], "Window": f"{sc['first']} to {sc['last']}",
                                 "Trading days": 0, "Book P&L (€m)": np.nan,
                                 "Worst book day (€m)": np.nan, "Worst day": "no data"})
                continue
            _win = f"close {res['base']:%d %b %Y} to close {res['end']:%d %b %Y}"
            if "book" in res:
                _summary.append({"Scenario": sc["name"], "Window": _win, "Trading days": res["days"],
                                 "Book P&L (€m)": round(res["book"] / 1e6, 2),
                                 "Worst book day (€m)": round(res["worst_pnl"] / 1e6, 2),
                                 "Worst day": f"{res['worst_day']:%d %b %Y}"})
            else:
                _summary.append({"Scenario": sc["name"], "Window": _win, "Trading days": res["days"],
                                 "Book P&L (€m)": np.nan, "Worst book day (€m)": np.nan,
                                 "Worst day": "no data for " + ", ".join(res["missing_held"])})
            for name in scen_panel.columns:
                inst = res["inst"].get(name)
                _detail.append({"Scenario": sc["name"], "Instrument": name,
                                "Position (€m)": positions.get(name, 0.0) / 1e6,
                                "Return over window": f"{inst['ret']:+.1%}" if inst else "no data",
                                "P&L (€m)": round(inst["pnl"] / 1e6, 2) + 0.0 if inst else np.nan})
            _roll_in = [d for d in _brent_rolls if pd.Timestamp(sc["first"]) <= d <= pd.Timestamp(sc["last"])]
            if _roll_in:
                _notes.append(f"{sc['name']}: {', '.join(f'{d:%d %b %Y}' for d in _roll_in)}")

        st.dataframe(pd.DataFrame(_summary).set_index("Scenario"), width="stretch")
        with st.expander("Return and P&L by instrument"):
            st.dataframe(pd.DataFrame(_detail).set_index(["Scenario", "Instrument"]), width="stretch")

        _sum_df = pd.DataFrame(_summary).dropna(subset=["Book P&L (€m)"])
        if len(_sum_df):
            fig_sc, ax_sc = plt.subplots(figsize=(14, 2.2 + 0.4 * len(_sum_df)))
            _vals = _sum_df["Book P&L (€m)"].values
            ax_sc.barh(_sum_df["Scenario"], _vals, color=np.where(_vals < 0, 'firebrick', 'seagreen'), height=0.5)
            for _y, _v in enumerate(_vals):
                ax_sc.annotate(f"€{_v:+,.2f}m", (_v, _y), textcoords='offset points',
                               xytext=(6 if _v >= 0 else -6, 0), ha='left' if _v >= 0 else 'right',
                               va='center', fontsize=8)
            ax_sc.axvline(0, color='black', linewidth=0.5)
            ax_sc.invert_yaxis()
            ax_sc.set_xlabel('Book P&L over the window (€m)')
            ax_sc.set_title('Book P&L if today\'s positions had been held through each window')
            ax_sc.margins(x=0.15)
            plt.tight_layout()
            st.pyplot(fig_sc)

        st.caption(
            "This replays what happened; it is not a forecast or a probability. Today's positions are entered "
            "at the close before each window and held unchanged (no rebalancing, hedging or stop-loss): P&L = "
            "position × the instrument's cumulative return, and the worst day is the largest one-day loss of "
            "the book inside the window. Windows: "
            + "; ".join(f"{sc['name']}, {sc['why']}" for sc in SCENARIOS) + ". "
            "Returns are the same daily returns as above: Yahoo Finance continuous front-month futures for "
            f"TTF (NYMEX), Brent and WTI, and for carbon {CARBON_LABEL}; {CARBON_FX_NOTE}. "
            "Windows are fixed dates and do not depend on the sidebar range."
            + (" Brent roll days inside the windows, when the continuous series switches contract and part "
               "of that day's Brent move is the gap between contracts: "
               + "; ".join(_notes) + "." if _notes else "")
        )
