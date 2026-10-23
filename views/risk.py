"""Risk page: market risk of the selected commodity and of a multi-commodity book.

Designed with a trading-house middle office (risk control) in mind.
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common

def _folded(label, *texts):
    """Method notes and caveats under a section, folded into one expander so the figures and the warnings that
    change how a figure reads stay on top. Empty texts are skipped; nothing is drawn when all are empty."""
    texts = [t.strip() for t in texts if t and t.strip()]
    if texts:
        with st.expander(label):
            for t in texts:
                st.caption(t)


def _ord(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _es_below_var99(n_tail, n):
    """Why ES 97.5% can come out below VaR 99%: written once, for the single-commodity VaR and the stress period."""
    k = int((n - 1) * 0.01) + 1
    return (f"Here ES averages the worst {n_tail} days while VaR 99% lies between the {k}{_ord(k)} and "
            f"{k + 1}{_ord(k + 1)} worst, so ES 97.5% can come out slightly below VaR 99%, as it does now; that is "
            "not a calculation error.")


def _nonpos_why(n_days, where=""):
    """Why a day with a zero or negative price drops out (common.simple_returns): written once, for the
    single-commodity VaR and the positions block."""
    return ("a percentage return is not defined across a zero or negative price, so the returns of "
            + ("that day" if n_days == 1 else "each of those days") + f" and the next are left out{where}")


ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]
CARBON_LABEL = ctx.carbon["label"]
CARBON_SHORT = ctx.carbon["short"]
CARBON_NOTE = ctx.carbon["note"]
ANNUAL = common.ANNUAL
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
      over 30 trading days, shown annualised (daily x √252) with the daily figure in brackets. This is a
      simplification: RiskMetrics uses exponentially weighted (EWMA) log returns.
    - **Rolling Correlation**: Pearson correlation of daily returns with the comparison asset over 30 days.
    - **Cross-Commodity Matrix**: 4×4 correlation heatmap (full period vs the last 30 trading days, at least 20 with all four returns).
    - **Value at Risk (VaR)**: 95% and 99% historical VaR and 97.5% Expected Shortfall
      from the last 250 trading days, with the full-period figures alongside. Quantiles use
      linear interpolation between the two nearest days; ES averages the days at or below
      the 2.5% quantile.
    - **GARCH(1,1) Forecast**: Student-t GARCH; daily volatility forecast for each of the next
      10 days, and the 10-day holding-period volatility.
    - **Volatility references**: two references that do not roll with the market: the multiple of
      the pre-war level (median 30-day volatility of Oct to Dec 2025) and the percentile among every
      30-day volatility since the start of the data (2017; TTF from Oct 2017). No Normal / Elevated /
      Extreme label: there is no industry-standard cut-off for one.
    - **Stress-period VaR and ES**: today's positions on the 250 trading days since 2017 (at most
      400 calendar days) in which they have the largest ES 97.5%, equally weighted, after the Basel
      method (BCBS MAR33.7), shown next to the last 250 days.
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

# ─── Section 1: Volatility (one summary for every page: common.vol_summary) ───
st.subheader("Volatility")
_vol_asof = core['vol_asof']
_past_end = end_date < common.berlin_today()
_vs = common.vol_summary(core, name=selected_commodity, with_roll=True, past_end=_past_end)
# Neutral heading: no traffic-light colour, so one window cannot decide the reading on its own.
st.markdown(f"<div style='font-size:1.9rem; font-weight:600; opacity:0.85; line-height:1.3'>{_vs['headline']}</div>"
            f"<div style='font-size:1.25rem; font-weight:500; opacity:0.85; margin:0.2rem 0 0.6rem'>"
            f"{_vs['headline2']}</div>", unsafe_allow_html=True)
st.caption("This compares volatility with the commodity's own past; it is not a measure of overall risk: low "
           "volatility says nothing about supply, storage or position size.")
_last_px = df_analysis['Price'].dropna().index.max()
_cmp_own = common.load_close(ctx.compare_ticker, start_date, end_date).dropna()
_last_cmp = _cmp_own.index.max() if len(_cmp_own) else None   # its own last price, not the last day both have
# TTF=F follows the NYMEX trading calendar, as BZ=F does: when Brent already has a price for a later day and
# TTF=F does not, Yahoo's TTF row for that day is late, not a holiday. Only Brent rows dated before today's New
# York date count (common.ttf_row_late): after New York midnight Yahoo opens today's BZ=F row for the evening session.
_ttf_late = ""
if not _past_end:
    for _n, _t, _d in ((selected_commodity, commodity['ticker'], _last_px), (compare_label, ctx.compare_ticker, _last_cmp)):
        if _t == "TTF=F" and _d is not None:
            _bz = common.load_close("BZ=F", (_d - pd.Timedelta(days=7)).date(), common.berlin_today())
            _late_day = common.ttf_row_late(_d, _bz)
            if _late_day is not None:
                _ttf_late = (f" TTF figures end a day earlier than Brent: Yahoo Finance has no TTF=F price for "
                             f"{_late_day:%d %b} in this download, although NYMEX Brent (BZ=F, which usually "
                             "follows the same trading calendar) has one.")
# The comparison series' last value gets the same note when it may not be a close (it enters the correlation and
# the book's P&L), phrased for that series.
_cmp_value_note = ""
if _last_cmp is not None:
    _cn = common.last_value_note(ctx.compare_ticker, end_date, _last_cmp)
    if _cn.startswith(" The last value") and ("may" in _cn or "latest price" in _cn):
        _cmp_value_note = f" For {compare_label}, the last value" + _cn[len(" The last value"):]
_last_note = common.last_value_note(commodity['ticker'], end_date, _last_px)
_data_full = (f"Data up to {_last_px:%d %b %Y} for {selected_commodity}"
              + (f", {_last_cmp:%d %b %Y} for {compare_label}" if _last_cmp is not None else "")
              + "." + _ttf_late
              # A futures row of today's New York date is a latest price, one of the day before (Monday to
              # Thursday) may not yet be the settlement (common.futures_row_state, the same rule as the curve).
              + _last_note
              + _cmp_value_note
              + (" The volatility, percentile, pre-war multiple, VaR and the roll check"
                 if commodity['ticker'] in common.CURVE_ROOTS else
                 " The volatility, percentile, pre-war multiple and VaR")
              + " all come from one download of the series, computed together.")
# On the page: the dates, what kind of price the last one is (in a few words; the full sentence is in the details)
# and a late TTF row, which changes what the latest figures cover. Definitions and the roll check are folded.
_kind = ("the last value is a live price, not a settlement" if "latest price" in _last_note else
         "the last value may not yet be the settlement" if "may not yet be the settlement" in _last_note else
         "the last value may be an intraday London price" if "London session" in _last_note else "")
# The comparison series' last value in the same few words when it may not be a close (it enters the 30-day
# correlation and the book's last-day P&L).
_cmp_kind = ("a live price" if "latest price" in _cmp_value_note else
             "may not yet be the settlement" if "may not yet be the settlement" in _cmp_value_note else
             "may be an intraday London price" if "London session" in _cmp_value_note else "")
_kinds = [k for k in (_kind, f"for {ctx.compare_label.split(' (')[0]}, {_cmp_kind}" if _cmp_kind else "") if k]
st.caption(f"Data up to {_last_px:%d %b %Y}" + (f" ({'; '.join(_kinds)})" if _kinds else "") + "." + _ttf_late)
# Roll days stay on the page in one short line (date, the contract switched to, the adjusted 30-day volatility),
# because the heading points to them; the return breakdown and why the heading is unadjusted are in the details.
_rc = core.get('rolls')
if _rc and _rc.get('rolls'):
    _root = common.CURVE_ROOTS[commodity['ticker']][0]
    _bits = []
    for _t, _sym, _r, _nr in _rc['rolls']:
        _live = common.futures_row_state(_t) != 'close'
        _bits.append(f"{pd.Timestamp(_t):%d %b %Y} (to the {common._contract_month(_sym, _root)} contract"
                     + (", read from the latest price" if _live else "") + ")")
    st.caption(("Contract roll days in the 30-day window: " if len(_bits) > 1 else "Contract roll day in the 30-day window: ")
               + " and ".join(_bits)
               + (f". With the new contract's own return on {'that day' if len(_bits) == 1 else 'those days'}, "
                  "the 30-day volatility is "
                  f"{common.fmt_vol(_rc['adjusted_vol'])}." if _rc.get('adjusted_vol') is not None else "."))
_folded("Details: roll days, data dates and the references", _vs['roll'], _vs['detail'], _data_full.replace(_ttf_late, ""))

_vol_series = df_analysis['Volatility'].dropna()
_corr_pairs = df_analysis[['Returns', 'Compare_Returns']].dropna()
# The 30-day correlation of the last day with a price (common.rolling_corr), or n/a with the reason: an older
# figure is not shown in its place.
_corr_now = core['latest_corr']
_partner = ctx.compare_ticker

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Current Volatility (30-day)", _vs['tile'],
           help="Standard deviation of the last 30 daily returns; annualised = daily x √252.")
mc2.metric("Average Volatility (30-day)", common.fmt_vol(avg_vol),
           help=f"Average 30-day volatility over the selected period ({_vol_series.index.min():%d %b %Y} to "
                f"{_vol_series.index.max():%d %b %Y}).")
if len(_corr_pairs):
    mc3.metric(f"Correlation vs {_partner}", f"{overall_corr:.2f}",
               help=f"Pearson correlation of daily returns, {_corr_pairs.index.min():%d %b %Y} to "
                    f"{_corr_pairs.index.max():%d %b %Y} ({len(_corr_pairs):,} common days).")
    _corr_note = f"Since {_corr_pairs.index.min():%b %Y}"
    if _corr_now is not None:
        _corr_note += f"; last 30 days: **{_corr_now:.2f}**"
    else:
        _corr_note += "; last 30 days: n/a. " + common.corr_gap_sentence(core)
    mc3.caption(_corr_note)
else:
    mc3.metric(f"Correlation vs {_partner}", "n/a", help="No common days with returns in the selected range.")
mc4.metric("VaR 95% (1-day, per unit)", f"{var_95:.2f}%",
           help=f"Historical, last {var_days} trading days: the daily loss exceeded on 5% of days, shown as a "
                "negative return. The € figures in Positions & Limits are shown as positive losses.")

# ─── Section 2: Price Chart ───
# Each date checked against a primary or contemporaneous source (project spec dashboard_规格/5_事实与来源.md, section E).
macro_events = {
    "2021-07-14": "EU Fit for 55 package",
    "2022-02-24": "Russia invades Ukraine",
    "2022-06-03": "EU adopts oil embargo (6th package; pipeline exempt)",
    "2022-09-26": "Nord Stream sabotage",
    "2022-12-05": "EU ban on Russian seaborne crude applies",
    "2023-02-15": "EU gas market correction mechanism can apply",
    "2023-04-18": "EP adopts ETS reform incl. ETS 2",
    "2024-01-01": "EU ETS extended to shipping",
    "2026-01-01": "CBAM definitive period starts",
    "2026-02-28": "US-Israel strikes on Iran",
    "2026-03-02": "Qatar halts LNG; IRGC official: Hormuz closed",
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
        # A row of today's New York date is a latest price; one of the day before (Monday to Thursday) may not
        # yet be the settlement; an older one is a close (common.futures_row_state). All the wording comes from
        # common.
        _curve_words = common.futures_curve_words(curve['Last trade'], front['Last trade'], settled)
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

            # Contracts of a season still trading, by the same expiry rule as the curve (common.contract_expiry);
            # a contract on its last trading day counts, as it does on the curve.
            _asof = curve['Last trade'].max()

            def _open_months(key):
                _y = int(key.split()[-1].split('/')[0]) + 2000
                months = (pd.date_range(f"{_y}-10-01", f"{_y + 1}-03-01", freq='MS') if key.startswith("Winter")
                          else pd.date_range(f"{_y}-04-01", f"{_y}-09-01", freq='MS'))
                return sum(not common.contract_expired(curve_root, m, _asof) for m in months)

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
                    col_fc.metric(f"{key}: {_d.min():%b %y} to {_d.max():%b %y} average ({n_season} of the {_open_months(key)} contracts still trading are on Yahoo Finance)",
                                  f"{seasons.loc[key, 'Price']:.2f} {curve_unit}")
                    # 4-6: say when the contracts averaged are not from the same day
                    _lt = curve.loc[curve['Season'] == key, 'Last trade']
                    if _lt.dt.normalize().nunique() > 1:
                        col_fc.caption(f"Prices from {_lt.min():%d %b} to {_lt.max():%d %b}")
                else:
                    _months = ", ".join(f"{d:%b %y}" for d in curve.loc[curve['Season'] == key, 'Delivery'])
                    col_fc.metric(f"{key} average", "n/a",
                                  help=f"Only {n_season} of the {_open_months(key)} monthly contracts still trading {'is' if n_season == 1 else 'are'} on Yahoo "
                                       f"Finance ({_months}); an average needs at least {_MIN_SEASON}.")
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
            _expiry = common.contract_expiry(curve_root, front['Delivery'])
            _left = (common.exchange_busday_count(curve_root, front['Last trade'] + pd.Timedelta(days=1),
                                                  _expiry + pd.Timedelta(days=1))
                     if _expiry is not None else None)
            _soon = _left is not None and _left < 3
            fc2.metric(f"Front month ({front['Delivery']:%b %y}" + (f", expires {_expiry:%d %b}" if _soon else "") + ")",
                       f"{front['Price']:.2f} {curve_unit}",
                       help=(f"Last trading day {_expiry:%d %b %Y} ({common.EXPIRY_RULES[curve_root]}; "
                             f"{common.expiry_cal_note(curve_root)})." if _expiry is not None else None))
            if _soon:
                # The roll check of the volatility window (same download) says whether the continuous series
                # has already moved past the front contract.
                _rc = core['rolls'] if not _past_end else None
                _moved = [(t, m) for t, m, _, _ in (_rc['rolls'] if _rc else [])
                          if common.contract_delivery(m, curve_root) > front['Delivery']]
                _follows = (f"`{commodity['ticker']}` has followed the "
                            f"{common.contract_delivery(_moved[-1][1], curve_root):%b %y} contract since "
                            f"{_moved[-1][0]:%d %b}." if _moved else
                            f"`{commodity['ticker']}` may already follow the next contract.")
                _to_go = (f"{front['Last trade']:%d %b} is its last trading day." if _left == 0 else
                          f"{_left} business day{'s' if _left != 1 else ''} to expiry after the {front['Last trade']:%d %b} close.")
                fc2.caption(f"{_to_go} Close to expiry the front month can be squeezed. {_follows}")
            _mixed = front['Last trade'].normalize() != back['Last trade'].normalize()
            fc3.metric(f"Front − {back_label} ({back['Delivery']:%b %y})", f"{fb_spread:+.2f} {curve_unit}",
                       help=(f"Front month price of {front['Last trade']:%d %b}, {back['Delivery']:%b %y} price of "
                             f"{back['Last trade']:%d %b}: the two ends are from different days." if _mixed
                             else _curve_words['hover']))
            if _mixed:
                fc3.caption(f"Front {front['Last trade']:%d %b}, back {back['Last trade']:%d %b}")
            ax_fc.scatter(labels_fc, curve['Price'], color=commodity['color'], s=30, zorder=2)
            shape_note = (
                "Contango: later deliveries cost more than the front month (spread negative). "
                "Backwardation: the front month costs more (spread positive), usually a sign of tight "
                "prompt supply. The label compares only the front and the last month shown. "
                f"The continuous series in the price chart ({commodity['ticker']}) switches to the next "
                "contract on a day near the front contract's expiry, before or after it, so its last price can "
                "differ from the first contract on this curve. ")

        for x_fc, y_fc in zip(labels_fc, curve['Price']):
            ax_fc.annotate(f"{y_fc:.2f}", (x_fc, y_fc), textcoords='offset points', xytext=(0, 7),
                           ha='center', fontsize=7)
        ax_fc.set_ylabel(curve_unit)
        ax_fc.set_xlabel('Delivery month')
        ax_fc.set_title(f"{selected_commodity} futures curve, {_curve_words['title']}: {curve_shape.lower()}")
        plt.tight_layout()
        st.pyplot(fig_fc)

        with st.expander("Contracts used"):
            st.dataframe(curve.assign(Delivery=curve['Delivery'].map(lambda d: f"{d:%b %Y}"),
                                      **{'Last trade': curve['Last trade'].dt.strftime('%Y-%m-%d')}),
                         width="stretch", hide_index=True)
        if _past_end:
            st.caption(f"The curve is today's, not as of the end date ({end_date:%d %b %Y}).")
        _folded("How the curve is built",
                shape_note
                + f"Monthly NYMEX contracts ({curve_root}<month code><year>.NYM) from Yahoo Finance, delayed. "
                f"A contract is shown if it has a daily close within the last {common.CURVE_FRESH_DAYS} trading "
                "days and has not passed its last trading day (exchange rule and holiday calendar; a contract "
                "on its last trading day is shown), "
                + _curve_words['caption']
                + "Contracts past their last trading day are not shown, even while Yahoo Finance still "
                "lists them; older or missing contracts are left out and nothing is interpolated. "
                f"{n_curve} of 12 contracts available.")

# ─── Section 3: Volatility & Correlation ───
vcol1, vcol2 = st.columns(2)
with vcol1:
    st.subheader("30-Day Rolling Volatility")
    _vol_line = common.break_gaps(df_analysis['Volatility'])
    fig_v, ax_v = plt.subplots(figsize=(7, 3.5))
    ax_v.plot(_vol_line.index, _vol_line * ANNUAL, color=commodity['color'], linewidth=1)
    ax_v.set_ylabel('Volatility, annualised (%)')
    ax_v.set_title(f'{selected_commodity}: 30-day rolling volatility, annualised (daily x √252)', fontsize=9)
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
    corr_full = _returns_df.corr().round(3)
    # Recent matrix: the same window rule as the 30-day correlation line (common.rolling_corr): the last 30 trading
    # days of the panel, given only when at least 20 of them have a return in all four series, so the window never
    # reaches back across a suspension (the carbon ETC in 2020-21) to fill itself up.
    _recent_rows = _panel.dropna(how='all').tail(common.VOL_DAYS)
    _recent_df = _recent_rows.dropna()
    corr_30d = _recent_df.corr().round(3) if len(_recent_df) >= common.VOL_MIN else None
else:
    corr_full, corr_30d = None, None


def _heatmap(corr, title):
    fig_h, ax_h = plt.subplots(figsize=(5, 5))
    im = ax_h.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
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
    _win_range = f"{_recent_rows.index.min():%d %b %Y} to {_recent_rows.index.max():%d %b %Y}"
    with cm1:
        st.write(f"**Full period: {_full_range}** ({len(_returns_df):,} common days)")
        st.pyplot(_heatmap(corr_full, f"Full period, {_full_range}"))
    with cm2:
        st.write(f"**Last {common.VOL_DAYS} trading days: {_win_range}** ({len(_recent_df)} with a return in all "
                 "four series)")
        if corr_30d is not None:
            st.pyplot(_heatmap(corr_30d, f"Last {common.VOL_DAYS} trading days, {_win_range}"))
        else:
            st.caption(f"No recent matrix: only {len(_recent_df)} of the last {common.VOL_DAYS} trading days have a "
                       f"return in all four series, and at least {common.VOL_MIN} are needed (the same thresholds as the "
                       f"30-day correlation line: {common.VOL_DAYS} days, at least {common.VOL_MIN}), so the window does not reach back across a gap such as the "
                       "carbon ETC's stale prices.")

    st.caption("Blue = positive correlation (move together). Red = negative (move inversely); for a hedged book a "
               "negative correlation is not in itself bad.")
    _folded("How the matrix is built",
               "Only days on which all four series have a return are used; the recent matrix uses the last 30 "
               "trading days on which any of the four has a return, and needs at least 20 of them with all four. Compare full "
               "period with the last 30 days to see whether relationships have shifted. The series do not close at the same "
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
    _why = _nonpos_why(len(_nonpos))
    # 3-2: say which figures the left-out days fall in: the last var_days days (headline) and/or the full period.
    _in_recent = len(_recent_ret) and any(_recent_ret.index[0] <= d <= _recent_ret.index[-1] for d in _nonpos.index)
    _which = (f"the headline (last {var_days} days) and full-period" if _in_recent else "the full-period")
    st.caption(f"⚠️ {selected_commodity} settled at or below zero on "
               + ", ".join(f"{d:%d %b %Y} ({v:.2f})" for d, v in _nonpos.items())
               + f". {_why[0].upper() + _why[1:]}, and {_which} max daily loss, VaR and ES do not include "
                 "that move.")

_worst_recent_day = _recent_ret.idxmin()
_worst_full_day = returns_clean.idxmin()
vc4.metric("Max daily loss", f"{_recent_ret.min() * 100:.2f}%",
           help=f"Worst daily return in the last {var_days} trading days, on {_worst_recent_day:%d %b %Y}.")
vc4.caption(f"on {_worst_recent_day:%d %b %Y}; {full_label}: **{returns_clean.min() * 100:.2f}%** "
            f"({_worst_full_day:%d %b %Y})")
_var_notes = []
_var_notes.append(f"Expected Shortfall (ES) 97.5% is the average return on the days at or below the 2.5% quantile "
           f"({_n_tail} of the last {var_days} days): the average loss in the worst 2.5% of days. The Basel Committee "
           "chose ES 97.5% for FRTB because it gives \"a broadly similar level of risk capture as the existing 99th "
           "percentile VaR threshold\" (BCBS, Fundamental review of the trading book, October 2013, p. 18)"
           + (". " + _es_below_var99(_n_tail, len(_recent_ret)) if abs(es_975) < abs(var_99) else "."))


_brent_roll_days = common.brent_roll_days


if commodity['ticker'] == "BZ=F":
    # Rule days corrected by the roll check of the latest 30-day window (Yahoo can switch on another day).
    _rolls = common.brent_roll_set(df_analysis['Price'].dropna().index, core['rolls'])
    _ex_roll = _recent_ret[~_recent_ret.index.isin(_rolls)]
    if _worst_recent_day in _rolls:
        st.caption(f"⚠️ {_worst_recent_day:%d %b %Y}, the worst day, is a Brent roll day: on this day the "
                   "continuous series switches to the next contract (the first trading day after an ICE Brent "
                   "contract's last trading day, or the day the roll check found the switch), and part or all of "
                   "the move can be the price gap between the two contracts, not a real loss. Excluding roll "
                   f"days, the worst day was {_ex_roll.min() * 100:.2f}% on {_ex_roll.idxmin():%d %b %Y}.")
    else:
        _var_notes.append(f"The worst day ({_worst_recent_day:%d %b %Y}) is not a Brent roll day (the first trading "
                   "day after an ICE Brent contract's last trading day: normally the first trading day of the "
                   "month, the last trading day of December for the February contract; or the day the roll check "
                   "found the switch).")
if commodity['ticker'] == "TTF=F":
    _ttf_exp = common.ttf_expiry_days(df_analysis['Price'].dropna().index)
    _flag = [d for d in (_worst_recent_day, _worst_full_day) if d in _ttf_exp]
    if _flag:
        st.caption("⚠️ " + " and ".join(sorted({f"{d:%d %b %Y}" for d in _flag})) + " is on or just after a TTF "
                   "front-month expiry (NYMEX Dutch TTF, rule 1159102.E: trading stops on the second London business "
                   "day before the delivery month, normally the same day as ICE Endex TTF), "
                   "so part of that day's move may be the gap between two contracts if the continuous series "
                   "switched on it. Which contract Yahoo's series followed that day is not verified.")
if commodity['ticker'].endswith("=F"):
    _var_notes.append(f"`{commodity['ticker']}` is Yahoo Finance's continuous front-month future. When the front "
               "contract rolls to the next month, the series switches contract and that day's return is "
               "not a real price move: when the next contract trades below the expiring one, "
               "the series shows a drop that no contract actually had. Roll days are not removed, so they can "
               "distort the volatility, the max daily loss, the tails, the VaR and the backtest exceptions. When "
               "the latest 30-day window contains a roll day, the line under the volatility heading gives the "
               "30-day volatility with the new contract's own return on that day.")

_folded("How VaR and ES are calculated", *_var_notes)

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
    _folded("How the backtest works", f"Each day's VaR is estimated only from the {var_days} trading days before it (no look-ahead) "
               "and compared with that day's return. Kupiec also rejects too few exceptions. Basel zones: "
               "green 0 to 4, yellow 5 to 9, red 10 or more exceptions at 99% over 250 days.")
else:
    st.caption(f"VaR backtest needs at least {common.VAR_WINDOW + 100} trading days of history; widen the date range.")


# ─── Section 4b: GARCH ───
st.subheader("GARCH Volatility Forecast")
st.write(f"Forward-looking volatility prediction for {selected_commodity} using GARCH(1,1) "
         "with Student-t shocks (fat tails)")

garch = common.fit_garch(returns_clean, avg_vol)
if garch is not None:
    forecast_vol = pd.Series(garch['forecast_vol'])
    current_cond_vol = garch['current_cond_vol']
    params = garch['params']
    cum_10d = garch['forecast_cum_10d']

    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Current GARCH Vol", common.fmt_vol(current_cond_vol),
               help="Fitted conditional volatility on the last day.")
    gc2.metric("Daily vol forecast, day 5", common.fmt_vol(forecast_vol.iloc[4]),
               help="Forecast volatility of the single daily return 5 trading days ahead, not of a 5-day period.")
    gc3.metric("Daily vol forecast, day 10", common.fmt_vol(forecast_vol.iloc[9]),
               help="Forecast volatility of the single daily return 10 trading days ahead, not of a 10-day period.")
    gc4.metric("10-day holding-period vol", f"{cum_10d * np.sqrt(252 / 10):.0f}% ann. ({cum_10d:.2f}% over 10 days)",
               help="Volatility of the cumulative return over the next 10 trading days: square root of the sum "
                    "of the 10 daily forecast variances; annualised x √(252/10).")

    fig_garch, (ax_cv, ax_fc) = plt.subplots(1, 2, figsize=(14, 4))

    cond_vol = common.break_gaps(garch['conditional_volatility'])
    ax_cv.plot(cond_vol.index, cond_vol * ANNUAL, color='purple', linewidth=0.8, alpha=0.8)
    ax_cv.set_ylabel('Conditional volatility, annualised (%)')
    ax_cv.set_title('GARCH(1,1) conditional volatility, annualised')
    ax_cv.fill_between(cond_vol.index, cond_vol * ANNUAL, 0, alpha=0.1, color='purple')

    _ci_lo, _ci_hi = garch['ci_lo'], garch['ci_hi']

    forecast_days = list(range(1, 11))
    ax_fc.plot(forecast_days, forecast_vol.values * ANNUAL, color='purple', linewidth=2, marker='o', markersize=5)
    ax_fc.fill_between(forecast_days, np.asarray(_ci_lo) * ANNUAL, np.asarray(_ci_hi) * ANNUAL,
                       alpha=0.18, color='purple', label='Simulated 90% range of future volatility (parameters fixed)')
    ax_fc.set_xlabel('Days Ahead')
    ax_fc.set_ylabel('Forecast volatility, annualised (%)')
    ax_fc.set_title('Daily volatility forecast, days 1 to 10, annualised')
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
        st.write(f"**Reference level: sample average 30-day volatility = {common.fmt_vol(avg_vol)}**, "
                 "over the selected range. (The Market page's GARCH check uses the 3-year average instead.)")
        if garch['garch_long_run_ok']:
            st.write(f"GARCH long-run volatility √(ω / (1 − α − β)) = {common.fmt_vol(garch['garch_long_run'])}, "
                     "the level the forecast reverts to.")
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
        pl1.metric("Book VaR 95% (1-day)", f"€{book_var95 / 1e6:,.2f}m",
                   help="Daily book loss exceeded on 5% of the last 250 days.")
        pl2.metric("Book VaR 99% (1-day)", f"€{book_var99 / 1e6:,.2f}m",
                   help="Daily book loss exceeded on 1% of the last 250 days.")
        pl3.metric("Expected Shortfall 97.5%", f"€{book_es975 / 1e6:,.2f}m",
                   help=f"Average loss on the {_n_tail_book} days (of {len(recent)}) at or below the 2.5% quantile, "
                        "1-day horizon. 97.5% is the confidence level Basel FRTB uses. FRTB also scales the ES by "
                        "liquidity horizon and calibrates it to a stress period; this figure is not scaled, and "
                        "the stress-period figures are below.")
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
                   f"€{(sum(standalone.values()) - book_var95) / 1e6:,.2f}m (sum of standalone VaRs minus book VaR). "
                   "EUR/USD is not included: returns are in each instrument's own currency.")

        # ── Stress period (after Basel MAR33.7) ──
        st.write(f"**Stress period: the {common.STRESS_WINDOW} trading days (about 12 months: normally 360 to 380 "
                 f"calendar days, at most {common.STRESS_MAX_SPAN}) in which today's positions have the largest "
                 "ES 97.5%**")
        _stress_panel = common.returns_panel(common.STRESS_FROM, end_date, CARBON_SHORT, CARBON_TICKER)
        _sp = (common.stress_period(_stress_panel[list(positions)], positions)
               if _stress_panel is not None and all(k in _stress_panel for k in positions) else None)
        if _sp is None:
            st.caption(f"No stress period: since {common.STRESS_FROM:%b %Y} there is no run of {common.STRESS_WINDOW} "
                       "days on which every instrument held has a return within "
                       f"{common.STRESS_MAX_SPAN} calendar days.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            for _c, _lab, _s, _now in ((sc1, "Stress VaR 95% (1-day)", _sp['var95'], book_var95),
                                        (sc2, "Stress VaR 99% (1-day)", _sp['var99'], book_var99),
                                        (sc3, "Stress ES 97.5% (1-day)", _sp['es975'], book_es975)):
                _c.metric(_lab, f"€{_s / 1e6:,.2f}m",
                          help=f"Same measure as above, on the stress period's {common.STRESS_WINDOW} days instead "
                               f"of the last {len(recent)}.")
                _c.caption(f"{_s / _now:.2f}x the last {len(recent)} days (€{_now / 1e6:,.2f}m)" if _now > 0 else "")
            _tie_txt = (f" The ES is the same in {_sp['ties']} windows ending {_sp['first_tie_end']:%d %b %Y} to "
                        f"{_sp['end']:%d %b %Y}, because ES 97.5% over 250 days depends only on the worst 7 days "
                        "and these windows all contain them; the latest is shown." if _sp['ties'] > 1 else "")
            # 3-1: VaR 95% depends on the 13th and 14th worst days, so it can differ across the tied windows.
            _v95_lo, _v95_hi = (f"€{_v / 1e6:,.2f}m" for _v in _sp['var95_ties'])
            if _sp['ties'] > 1 and _v95_lo != _v95_hi:
                _tie_txt += f" VaR 95% ranges {_v95_lo} to {_v95_hi} across these windows."
            # 4-3: the same explanation as the single-commodity VaR when ES 97.5% is below VaR 99%.
            _es_txt = (" " + _es_below_var99(_sp['n_tail'], common.STRESS_WINDOW)
                       if _sp['es975'] < _sp['var99'] else "")
            # 1-1: a held instrument with a zero or negative close anywhere in the search range (WTI on 20 Apr
            # 2020): those days and the next have no percentage return, so they are not in the book P&L and the
            # search never sees them. Their P&L is given as the price change on a fixed quantity (the quantity
            # bought at the last positive close, as in the scenario replay below) and compared with the stress
            # ES. Found from the prices, nothing hard-coded; the returns themselves are not changed.
            _nonpos_txt = ""
            _tickers = common.panel_tickers(CARBON_SHORT, CARBON_TICKER)
            for _k, _v in positions.items():
                _px = common.load_close(_tickers[_k], common.STRESS_FROM, end_date).dropna()
                _vals = _px.values
                _i = 1
                while _i < len(_vals):
                    if _vals[_i] > 0 or _vals[_i - 1] <= 0:
                        _i += 1
                        continue
                    # A run of zero or negative closes from _i; the day after the run has no return either.
                    _j = _i
                    while _j < len(_vals) and _vals[_j] <= 0:
                        _j += 1
                    if _px.index[_i] < _sp['search_from']:
                        _i = _j + 1
                        continue
                    _days = list(range(_i, min(_j + 1, len(_vals))))
                    _qty = _v / _vals[_i - 1]
                    _pnl = [(_px.index[d], _qty * (_vals[d] - _vals[d - 1])) for d in _days]
                    _n_run = _j - _i
                    _nonpos_txt += (f" {_k} settled at " + ", ".join(f"{_vals[d]:.2f} on {_px.index[d]:%d %b %Y}"
                                                                    for d in range(_i, _j))
                                    + f", from {_vals[_i - 1]:.2f} the day before: "
                                    + _nonpos_why(_n_run, " of the book P&L and of the stress-period search")
                                    + f". Holding the quantity bought at the {_px.index[_i - 1]:%d %b %Y} close of "
                                    f"{_vals[_i - 1]:.2f}, as the scenario replay below does, the {_k} position "
                                    "would have "
                                    + " and ".join(f"{'made' if _p >= 0 else 'lost'} €{abs(_p) / 1e6:,.2f}m on "
                                                   f"{_d:%d %b %Y}" for _d, _p in _pnl) + ".")
                    _worse = [(_d, -_p) for _d, _p in _pnl if -_p > _sp['es975']]
                    if _worse:
                        _nonpos_txt += ((" The loss on " if len(_worse) == 1 else " The losses on ")
                                        + " and ".join(f"{_d:%d %b %Y}" for _d, _ in _worse)
                                        + (" is" if len(_worse) == 1 else " are each")
                                        + f" larger than the stress ES 97.5% shown (€{_sp['es975'] / 1e6:,.2f}m): "
                                        "counted this way, a window containing these days could have a larger ES than the stress "
                                        "period shown.")
                    _i = _j + 1
            st.caption(f"Stress period: **{_sp['start']:%d %b %Y} to {_sp['end']:%d %b %Y}**"
                       + (f" (the latest of {_sp['ties']} windows with the same ES)" if _sp['ties'] > 1 else "") + ".")
            if _nonpos_txt.strip():
                st.caption("⚠️ " + _nonpos_txt.strip())
            _folded("How the stress period is chosen",
                f"Stress period: {_sp['start']:%d %b %Y} to {_sp['end']:%d %b %Y}, searched from "
                f"{_sp['search_from']:%b %Y} to {_sp['search_to']:%d %b %Y}: the {common.STRESS_WINDOW} trading "
                f"days in which today's positions would have had the largest ES 97.5%, all days equally weighted. "
                "As above, only days on which every instrument held has a return count"
                + (f"; the {common.STRESS_MAX_SPAN}-day limit leaves out the carbon ETC's suspension (Jun 2020 to "
                   "Aug 2021) and its sparse prices before 2018" if CARBON_SHORT in positions else "")
                + f".{_tie_txt}{_es_txt} The stress period changes when the "
                "positions change. Stress-period VaR and ES borrow the Basel method (bank capital rules): Basel "
                "picks the 12 months of largest loss for the current portfolio, with a 10-day horizon and liquidity "
                "scaling; here the horizon is 1 day, as above. The search starts in 2017: TTF=F starts on 23 Oct "
                "2017, so a book with TTF cannot reach back to 2007 as Basel requires. Brent and WTI were checked "
                "back to 2007 and 2000 (on 22 Oct 2026): no earlier 12 months was more volatile or had a larger "
                "loss for one unit long or short.")

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

        _folded("How the book VaR and its backtest work",
            f"Historical simulation on the last {_LOOKBACK} trading days, 1-day horizon. Only days on which "
            "every instrument held has a return count: a move of one instrument on the other's holiday is "
            "left out, and the other's next return spans both days. "
            "Backtest is hypothetical: today's positions applied to past returns, each day's VaR "
            f"estimated only from the {_LOOKBACK} days before it. Positions are held static. Returns are "
            "taken in each instrument's quote currency and applied to the € notional, so EUR/USD is ignored "
            f"for Brent and WTI; {CARBON_FX_NOTE}. Kupiec also rejects too few exceptions (an "
            "over-conservative model). Basel zones: green 0 to 4, yellow 5 to 9, red 10 or more exceptions at 99% over "
            "250 days."
        )


# ─── Section 5b: Historical scenario replay (uses the positions above) ───
# Fixed past windows. Each starts at the close before the first day listed and ends at the
# close of the last day. Dates are checked against prices and news sources (project spec dashboard_规格/5_事实与来源.md, section E).
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

        st.caption("This replays what happened; it is not a forecast or a probability.")
        if _notes:
            st.caption("⚠️ Brent roll days inside the windows (part of that day's Brent move is the gap between two "
                       "contracts): " + "; ".join(_notes) + ".")
        _folded("How the replay works",
            "Today's positions are entered "
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
