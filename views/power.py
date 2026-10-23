"""Power page: German day-ahead power and the clean spark spread of a gas plant.

Designed with power trading desks and utilities in mind.
"""
import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common
import power_data

# Efficiencies on a higher heating value (HHV) basis, as S&P Global Platts publishes its German
# clean spark spreads: 45% and 50% for all locations, plus 60% for the UK, Germany and Italy
# (Platts European Electricity specifications: June 2026 edition p.14, October 2026 edition p.15, same text).
# 50% is the central case.
EFFICIENCY = 0.50
EFFICIENCIES = (0.45, 0.50, 0.60)
# Emission intensity of natural gas on the same HHV basis as TTF prices, which are quoted per MWh
# of gross calorific value (EU Regulation 2015/703, Article 13: energy in kWh on a gross calorific value
# basis for gas data exchange; the ICE Endex contract specifications do not state it): Platts uses 0.053942 tCO2e/MMBtu (HHV) = 0.053942 x 3.412142
# MMBtu/MWh = 0.18406 tCO2 per MWh of gas. (The often-quoted 0.202 is per MWh on a net, lower
# heating value basis and would overstate the carbon cost of TTF-priced gas by about 10%.)
GAS_EF = 0.18406
CARBON_PER_MWH = GAS_EF / EFFICIENCY   # tCO2 per MWh of power at the central efficiency
PLANT_MW = 400
_LOOKBACK = 250            # day-on-day changes behind the 1-in-20 figures
_TTF_MAX_AGE = 5           # calendar days: a TTF settlement older than this is not used
_EUA_MAX_AGE = 14          # calendar days: an auction price is carried forward at most two weeks
_PEAK = f"weekdays {power_data.PEAK_START:02d}:00 to {power_data.PEAK_END:02d}:00"


def _eff(e):
    return f"{e:.0%}"


ctx = common.context()
start_date, end_date = ctx.start_date, ctx.end_date

with st.sidebar.expander("Methodology: Power page"):
    st.markdown(f"""
    - **Power**: Germany/Luxembourg (DE-LU) day-ahead prices from the ENTSO-E Transparency
      Platform (the 12:00 SDAC auction; ENTSO-E also lists EXAA's 10:15 auction, which is not
      used), with Energy-Charts as a fallback carrying the same prices. Hourly, every 15
      minutes since 1 Oct 2025. **Peak** is the time-weighted average over {_PEAK} (Berlin
      time), **baseload** over all 24 hours.
    - **Gas**: TTF front-month future (`TTF=F`), €/MWh, gross calorific value (HHV) basis.
    - **Carbon**: EUA primary-auction clearing price from EEX, €/tCO2.
    - **Timing**: a day's power is sold in the day-ahead auction held at 12:00 the day before
      delivery, every day of the year (Monday's power on Sunday). Gas and carbon do not trade at
      weekends, so each delivery day is matched with the last TTF close on a weekday **before**
      it (Friday's for Monday) and the last EUA auction before it. For Tuesday to Friday the gas
      price is the close of the power-auction day itself, which comes after the 12:00 auction, so
      it is not the price a bidder saw at 12:00. EUA auctions are not held every weekday; the page
      uses the last auction before the delivery day and prints its date: the power-auction day's
      when that day has one (result at about 11:00, before the power auction), otherwise an
      earlier one, and the headline says which.
    - **Clean spark spread** = power − gas / efficiency − ({GAS_EF:.5f} / efficiency) × EUA,
      efficiencies 45%, 50% and 60% (HHV), as S&P Global Platts publishes for Germany. A gross
      margin per MWh before start-up, other variable and fixed costs.
    - **Flexible plant**: runs in any period of a business day (all 24 hours) where the price is
      above its fuel and carbon cost at 50% efficiency. Start-up costs, minimum run times and
      part-load operation are ignored.
    - **Gas-output-weighted price**: the day-ahead price weighted by the actual output of German
      public-supply gas plants (Energy-Charts public net generation; industrial self-generation
      is not included). A day-ahead capture price, not what the plants sold at: output sold
      forward earns the forward price instead.
    - **Risk**: day-on-day changes between consecutive delivery business days (Monday compares
      with Friday), in €/MWh for the spread (it can be negative) and in € for the flexible
      plant's daily margin.
    """)

st.title("Power & Clean Spark Spread")
st.markdown("Does a gas-fired plant in Germany cover its fuel and carbon costs over the peak block, "
            "and in the hours it would choose to run? How much can that margin move?")

# ─── Data ───
with st.spinner("Fetching German day-ahead power prices (ENTSO-E)..."):
    # Tomorrow's day-ahead prices are published around 13:00; include them when the range ends today.
    _power_end = (end_date + pd.Timedelta(days=1) if end_date >= common.berlin_today() else end_date)
    prices, power_sources, power_errors = power_data.load_power(start_date, _power_end)
if prices.empty:
    st.warning("German power prices are unavailable right now (neither ENTSO-E nor Energy-Charts could "
               "be reached). The rest of the dashboard is unaffected; try again later.")
    st.stop()
if power_errors:
    st.warning("Some years of power data could not be loaded, so the charts show only the years "
               "that did: " + "; ".join(power_errors))

ttf = common.load_close("TTF=F", start_date - pd.Timedelta(days=10), end_date).dropna()
if ttf.empty:
    st.warning("TTF gas prices are unavailable right now, so the spread cannot be computed.")
    st.stop()

with st.spinner("Fetching EUA auction prices (EEX)..."):
    eua_auctions, eua_errors = power_data.load_eua_auctions(start_date - pd.Timedelta(days=30), end_date)
if eua_auctions.empty:
    st.warning("EEX auction prices could not be loaded, so the spread cannot be computed. "
               "No built-in or approximate carbon price is used. Try again later.")
    st.stop()
if eua_errors:
    st.warning("Some years of EEX auction prices could not be loaded: " + "; ".join(eua_errors))


def _last_before(series, days, max_age_days):
    """For each day, the last value of `series` dated strictly before it, and that value's date.

    Values older than `max_age_days` calendar days are not used (left missing)."""
    s = series.dropna().sort_index()
    pos = s.index.searchsorted(days, side="left") - 1
    ok = pos >= 0
    vals = pd.Series(np.nan, index=days)
    dates = pd.Series(pd.NaT, index=days, dtype="datetime64[ns]")
    vals[ok] = s.values[pos[ok]]
    dates[ok] = s.index[pos[ok]]
    too_old = (days.to_series() - dates) > pd.Timedelta(days=max_age_days)
    return vals.mask(too_old), dates.mask(too_old)


def _half(x):
    """Round to the nearest 0.5."""
    return round(x * 2) / 2


def _single_period_runs(prices, cost_per_day, days):
    """Of the flexible plant's runs on `days`, the share that last a single delivery period, and the number of
    runs. A run is a stretch of consecutive periods priced above the day's cost; as in flexible_run, a run that
    continues past midnight into the next delivery day is one run, and a gap (a weekend) ends it."""
    day = prices.index.tz_localize(None).normalize()
    keep = day.isin(days)
    p, d = prices[keep], day[keep]
    cost = pd.Series(cost_per_day.reindex(d).values, index=p.index)
    run = ((p - cost) > 0) & cost.notna()
    idx = p.index.to_series()
    contiguous = ((idx - idx.shift(1)) <= pd.Timedelta(hours=1)).values
    start = run & ~(run.shift(1, fill_value=False).astype(bool) & contiguous)
    lengths = run[run].groupby(start.cumsum()[run]).size()
    return ((lengths == 1).mean() if len(lengths) else np.nan), len(lengths)


def _build_frame(prices, ttf, eua):
    """Per delivery business day: power, the TTF and EUA prices matched to it, costs, spreads at each
    efficiency, the flexible plant's run and margin, and the day-on-day changes behind the risk figures."""
    daily = power_data.daily_base_peak(prices)
    days = daily.index[daily.index.dayofweek < 5]
    df = pd.DataFrame(index=days)
    df["Power peak"] = daily["Peak"].reindex(days)
    df["Power base"] = daily["Base"].reindex(days)
    df["TTF"], df["TTF date"] = _last_before(ttf, days, _TTF_MAX_AGE)
    df["EUA"], df["EUA date"] = _last_before(eua, days, _EUA_MAX_AGE)
    df = df.dropna(subset=["Power peak", "Power base", "TTF", "EUA"])
    if df.empty:
        return df, pd.Series(dtype=float)
    df["Fuel cost"] = df["TTF"] / EFFICIENCY
    df["Carbon cost"] = CARBON_PER_MWH * df["EUA"]
    cost = df["Fuel cost"] + df["Carbon cost"]
    df["CSS peak"] = df["Power peak"] - cost
    df["CSS base"] = df["Power base"] - cost
    for e in EFFICIENCIES:
        df[f"CSS peak {_eff(e)}"] = df["Power peak"] - _cost(e, df)
        df[f"CSS base {_eff(e)}"] = df["Power base"] - _cost(e, df)
    flex = power_data.flexible_run(prices, cost)
    df["Run hours"] = flex["Run hours"].reindex(df.index)
    df["Run spread"] = flex["Run spread"].reindex(df.index)
    df["Starts"] = flex["Starts"].reindex(df.index)
    # Daily gross margin of the flexible plant, € (0 on days it stays off).
    df["Flex margin"] = (df["Run hours"] * df["Run spread"]).fillna(0.0) * PLANT_MW
    # Day-on-day changes between consecutive delivery business days; a change across a data gap of
    # more than a week is not a one-day move, so it is left out.
    gap = df.index.to_series().diff() > pd.Timedelta(days=7)
    df["dCSS"] = df["CSS peak"].diff().mask(gap)
    df["dCSS vol"] = df["dCSS"].rolling(30, min_periods=20).std()
    df["dFlex"] = df["Flex margin"].diff().mask(gap)
    return df, cost


def _resolution_effect(prices, cost_per_day, quarters):
    """How much more the flexible plant earns on 15-minute prices than on the same prices averaged to
    hours, in %: lowest and highest over the shown quarters that have 15-minute prices."""
    p = prices[prices.index >= pd.Timestamp("2025-10-01", tz=prices.index.tz)]
    if p.empty:
        return np.nan, np.nan
    out = []
    for name, s in (("q", p), ("h", p.resample("h").mean().dropna())):
        r = power_data.flexible_run(s, cost_per_day)
        m = (r["Run hours"] * r["Run spread"]).fillna(0.0)
        out.append(m.groupby(m.index.to_period("Q")).sum())
    q, h = out
    pct = (100 * (q / h - 1)).reindex([x for x in quarters if x >= pd.Period("2025Q4")]).dropna()
    return (pct.min(), pct.max()) if len(pct) else (np.nan, np.nan)


def _cost(eff, frame):
    return frame["TTF"] / eff + GAS_EF / eff * frame["EUA"]


# Delivery business days (the peak block exists on weekdays only).
df, cost = _build_frame(prices, ttf, eua_auctions)
if len(df) < _LOOKBACK // 5:
    st.warning("Not enough overlapping power, gas and carbon data for this date range; widen it.")
    st.stop()

# ─── Headline: latest delivery day ───
latest = df.iloc[-1]
day = df.index[-1]
st.markdown(f"**Clean spark spread for delivery on {day:%a %d %b %Y}**")
h1, h2, h3 = st.columns(3)
h1.metric(f"Peak block ({power_data.PEAK_START:02d}:00 to {power_data.PEAK_END:02d}:00), 50% efficiency (HHV)",
          f"{latest['CSS peak']:+.2f} €/MWh",
          help="Average spread over this day's peak delivery periods: what a plant running the whole "
               "peak block earns per MWh, before start-up, other variable and fixed costs.")
h2.metric("Baseload (all 24 h), 50% efficiency (HHV)", f"{latest['CSS base']:+.2f} €/MWh")
if latest["Run hours"] > 0:
    h3.metric("Flexible plant, 50% efficiency (HHV)",
              f"{latest['Run hours']:.2f} h at {latest['Run spread']:+.2f} €/MWh",
              help=f"Hours of the day (any of the 24) where the price covered the fuel and carbon cost, and "
                   f"the average spread in those hours. It would start {int(latest['Starts'])} time(s) that "
                   "day; start-up costs are ignored.")
else:
    h3.metric("Flexible plant, 50% efficiency (HHV)", "Stays off",
              help="No period of the day covered the fuel and carbon cost.")
_today = common.berlin_today()
_hl = common.front_month_contract("TTF=F", latest["TTF date"])
_hl_contract = (f"TTF {_hl['delivery']:%b %y} future (front month)" if _hl["matched"] else
                f"TTF {_hl['delivery']:%b %y} future (front month by the exchange expiry rule; "
                "this price could not be matched to a single monthly contract on Yahoo Finance)")
# The day-ahead auction for this delivery day runs the calendar day before; gas and carbon trade on weekdays,
# so the last trading day before the auction is the last weekday before the delivery day.
_trade_day = pd.Timestamp(np.busday_offset(day.date(), -1, roll="backward"))


# ─── Timing sentence (tests/test_power_timing.py runs this block on its own) ───
# Yahoo's TTF=F follows the NYMEX trading calendar, so a weekday can be an exchange holiday (e.g. US
# Thanksgiving). Gas counts as late only if NYMEX Brent (BZ=F, normally the same calendar) has a price for a day
# up to the trade day that TTF=F lacks, by the Risk page's rule (common.ttf_row_late: only BZ=F rows dated before
# today's New York date count, because after New York midnight Yahoo opens a BZ=F row for the evening session).
def _gas_late(ttf_day, trade_day, bz_close, now=None):
    """(day, recent): the latest day up to `trade_day` on which NYMEX Brent has a price and TTF=F, whose last price
    before the delivery day is of `ttf_day`, does not (common.ttf_row_late, the Risk page's rule), else None; and
    whether that price can still arrive, i.e. the day is the day before today's New York date, Monday to Thursday
    (common.futures_live_from). For an older day, or a Friday from Saturday on, the missing price will not come."""
    trade_day = pd.Timestamp(trade_day).normalize()
    if pd.Timestamp(ttf_day).normalize() >= trade_day:
        return None, False
    bz = bz_close.dropna()
    late = common.ttf_row_late(ttf_day, bz[bz.index.normalize() <= trade_day], now)
    return late, late is not None and late >= common.futures_live_from(now)


# EEX does not auction EUAs on every weekday, and TTF=F has no price on NYMEX holidays. When a price is not from the
# last weekday before the delivery day, say so and name the weekdays without one (from the data).
def _days_txt(first_after, last):
    d = pd.bdate_range(first_after.normalize() + pd.Timedelta(days=1), last)
    return (f"{d[0]:%a %d %b}" if len(d) == 1 else f"{d[0]:%a %d %b} or {d[1]:%a %d %b}" if len(d) == 2 else
            f"any weekday from {d[0]:%a %d %b} to {d[-1]:%a %d %b}")


def _timing_sentence(trade_day, ttf_day, eua_day, late_day=None, late_recent=False):
    """Which day the gas (TTF) and carbon (EUA) prices of a delivery day are from, against `trade_day`, the last
    weekday before the delivery day. `late_day` is the day NYMEX Brent has a price and TTF=F does not
    (common.ttf_row_late), `late_recent` whether that price can still arrive. The reason for using the auction day's
    gas close (Monday's power auctioned on Sunday; the close comes after the 12:00 auction) is given only when the
    gas price is from the trade day; otherwise the sentence says which weekdays have no price and that earlier ones
    are used."""
    trade_day, ttf_day, eua_day = (pd.Timestamp(x).normalize() for x in (trade_day, ttf_day, eua_day))
    gas_ok, eua_ok = ttf_day >= trade_day, eua_day >= trade_day
    if gas_ok and eua_ok:
        return (f"Both are taken from the last weekday before this delivery day, {trade_day:%a %d %b}. Monday's "
                "power is auctioned on Sunday, so Friday's prices are used; for Tuesday to Friday it is the day of the "
                "12:00 power auction itself, when the EUA auction result (about 11:00) is already out but the TTF "
                "close comes later, so the gas price is not the one a bidder saw at 12:00.")
    if gas_ok:
        return (f"Gas is taken from the last weekday before this delivery day, {trade_day:%a %d %b}, and carbon "
                f"from the last auction before it (no auction on {_days_txt(eua_day, trade_day)}). Monday's power is "
                "auctioned on Sunday, so Friday's prices are used; for Tuesday to Friday the gas price is from the day "
                "of the 12:00 power auction itself, and the TTF close comes after the auction, so it is not the price a "
                "bidder saw at 12:00.")
    no_gas = f"no TTF close on {_days_txt(ttf_day, trade_day)}"
    if late_day is not None:
        no_gas += (f"{' yet' if late_recent else ''}, although NYMEX Brent (`BZ=F`, normally the same trading days) "
                   f"has one for {pd.Timestamp(late_day):%a %d %b}")
    after = " This spread will change when that TTF price arrives." if late_day is not None and late_recent else ""
    if eua_ok:
        return (f"Carbon is taken from the last weekday before this delivery day, {trade_day:%a %d %b}, but gas is "
                f"not: this page's data has {no_gas}, so an earlier gas price is used." + after)
    return (f"Neither is from the last weekday before this delivery day, {trade_day:%a %d %b}: this page's data has "
            f"no EUA auction on {_days_txt(eua_day, trade_day)} and {no_gas}, so the last earlier ones are used."
            + after)
# ─── End of timing sentence ───


# How a TTF price is labelled by its row's date in New York (common.futures_row_state, the Risk page's rule):
# today's New York date or later is the latest price, the day before (Monday to Thursday) may not yet be the
# settlement, older rows and Friday's from Saturday on are closes. The headline, the CSV column and the gas-purchase label all use these words.
_PRICE_TYPE = {"live": "latest price, not the settlement", "unsettled": "may not yet be the settlement",
               "close": "close"}


def _price_of(d, value_txt):
    state = common.futures_row_state(d)
    return {"live": f"latest price of {d:%a %d %b} ({value_txt}), not the settlement",
            "unsettled": f"price of {d:%a %d %b} ({value_txt}), which may not yet be the settlement",
            "close": f"close of {d:%a %d %b} ({value_txt})"}[state]


_bz = (common.load_close("BZ=F", _trade_day.date() - pd.Timedelta(days=7), _today)
       if latest["TTF date"].normalize() < _trade_day else pd.Series(dtype=float))
_from_txt = _timing_sentence(_trade_day, latest["TTF date"], latest["EUA date"],
                             *_gas_late(latest["TTF date"], _trade_day, _bz))
_ttf_txt = f"{latest['TTF']:.2f} €/MWh"
st.caption(f"Gas: {_hl_contract}, {_price_of(latest['TTF date'], _ttf_txt)}; not day-ahead gas, see Caveats. "
           f"Carbon: EEX auction of "
           f"{latest['EUA date']:%a %d %b} ({latest['EUA']:.2f} €/t). " + _from_txt)

last_year = df[df.index > day - pd.DateOffset(years=1)]
has_year = df.index[0] <= day - pd.DateOffset(years=1)
period_label = "last 12 months" if has_year else "selected range"
eff_table = pd.DataFrame({
    _eff(e): [f"{latest[f'CSS peak {_eff(e)}']:+.2f}", f"{latest[f'CSS base {_eff(e)}']:+.2f}",
              f"{(last_year[f'CSS peak {_eff(e)}'] > 0).mean():.0%}"]
    for e in EFFICIENCIES
}, index=[f"Peak spread, {day:%d %b} (€/MWh)", f"Baseload spread, {day:%d %b} (€/MWh)",
          f"Days with a positive peak spread, {period_label}"])
eff_table.columns.name = "Efficiency (HHV)"
st.markdown("**By plant efficiency, higher heating value (HHV) basis** (S&P Global Platts publishes German "
            "spreads at 45%, 50% and 60% HHV; 50% HHV is roughly 55% on a lower heating value basis)")
st.table(eff_table)
st.caption(f"{period_label.capitalize()}: {last_year.index[0]:%d %b %Y} to {day:%d %b %Y}, "
           f"{len(last_year)} delivery business days (days with no EUA auction in the previous two weeks, "
           "around the year-end auction break, are left out)."
           + (f" Over the whole selected range ({df.index[0]:%d %b %Y} to {day:%d %b %Y}), the peak spread "
              f"at 50% was positive on {(df['CSS peak'] > 0).mean():.0%} of days." if has_year else ""))

k1, k2, k3 = st.columns(3)
k1.metric("Power (DE-LU peak)", f"{latest['Power peak']:.2f} €/MWh",
          help=f"Average over {_PEAK}. Baseload (24 h): {latest['Power base']:.2f} €/MWh.")
k2.metric("Fuel cost at 50% (TTF / 0.5)", f"{latest['Fuel cost']:.2f} €/MWh")
k3.metric(f"Carbon cost at 50% ({CARBON_PER_MWH:.3f} × EUA)", f"{latest['Carbon cost']:.2f} €/MWh")

# ─── Chart 1: spread over time ───
st.subheader("Clean Spark Spread")
fig1, ax1 = plt.subplots(figsize=(14, 4))
# Lines break where the data has gaps longer than a week, instead of joining across them.
_pk = common.break_gaps(df["CSS peak"])
_bs = common.break_gaps(df["CSS base"])
ax1.plot(_pk.index, _pk.values, color="black", linewidth=0.8, label=f"Peak ({_PEAK})")
ax1.fill_between(_pk.index, _pk.values, 0, where=(_pk > 0).values, color="green", alpha=0.25,
                 interpolate=True)
ax1.fill_between(_pk.index, _pk.values, 0, where=(_pk <= 0).values, color="red", alpha=0.25,
                 interpolate=True)
ax1.plot(_bs.index, _bs.values, color="gray", linewidth=0.6, alpha=0.8, label="Baseload (24 h)")
ax1.axhline(0, color="black", linewidth=1)
ax1.set_ylabel("€/MWh")
ax1.set_title("Clean spark spread at 50% efficiency, DE-LU: peak (green above zero, red below) and baseload")
ax1.legend(fontsize=8, loc="upper left")
plt.tight_layout()
st.pyplot(fig1)
_cheap_peak = (last_year["Power peak"] < last_year["Power base"]).mean()
st.caption(f"On {_cheap_peak:.0%} of delivery business days in the {period_label}, the peak block averaged "
           "below baseload: solar output pushes prices down around midday, so the 08:00 to 20:00 block "
           "is not always the expensive part of the day. That is why the flexible plant below can earn "
           "money when the peak spread is negative.")

# ─── Chart 2: decomposition ───
st.subheader("What Drives the Spread")
fig2, ax2 = plt.subplots(figsize=(14, 4))
for _col, _kw in [("Power peak", dict(color="goldenrod", linewidth=1, label="Power price, peak")),
                  ("Power base", dict(color="goldenrod", linewidth=0.6, alpha=0.5, label="Power price, baseload")),
                  ("Fuel cost", dict(color="steelblue", linewidth=1, label="Fuel cost at 50% (TTF / 0.5)")),
                  ("Carbon cost", dict(color="seagreen", linewidth=1,
                                       label=f"Carbon cost at 50% ({CARBON_PER_MWH:.3f} × EUA)"))]:
    _s = common.break_gaps(df[_col])
    ax2.plot(_s.index, _s.values, **_kw)
ax2.set_ylabel("€/MWh")
ax2.set_title("Spread = power price − fuel cost − carbon cost")
ax2.legend(fontsize=8, loc="upper left")
plt.tight_layout()
st.pyplot(fig2)

# ─── What a flexible plant earns, and what gas plants actually captured ───
st.subheader("What a Flexible Plant Earns")
q = df[["Flex margin", "CSS peak"]].copy()
q["Quarter"] = q.index.to_period("Q")
by_q = q.groupby("Quarter").agg(margin=("Flex margin", "sum"), peak=("CSS peak", "mean"), n=("Flex margin", "size"))
by_q = by_q.tail(8)
fig_q, (axq1, axq2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
_labels = [str(p) + (" (to date)" if p == by_q.index[-1] and day < p.end_time.normalize() else "")
           for p in by_q.index]
_x = np.arange(len(by_q))
axq1.bar(_x, by_q["margin"] / 1e6, color="seagreen")
axq1.set_ylabel("€m")
axq1.set_title(f"{PLANT_MW} MW flexible plant at 50% HHV: gross margin per quarter, an upper bound "
               "(no start-up costs or minimum run times; delivery business days with an EUA price only)")
axq2.bar(_x, by_q["peak"], color=np.where(by_q["peak"] > 0, "green", "red"), alpha=0.6)
axq2.axhline(0, color="black", linewidth=0.8)
axq2.set_ylabel("€/MWh")
axq2.set_title("Average peak spread at 50% in the same quarter")
axq2.set_xticks(_x)
axq2.set_xticklabels(_labels)
plt.tight_layout()
st.pyplot(fig_q)
_starts_day = last_year["Starts"].mean()
_q_days = 63   # delivery business days in a quarter, roughly
_start_lo = _starts_day * _q_days * PLANT_MW * 35 / 1e6
_start_hi = _starts_day * _q_days * PLANT_MW * 55 / 1e6
_res_lo, _res_hi = _resolution_effect(prices, cost, by_q.index)
_single, _n_runs = _single_period_runs(prices, cost, last_year.index)
_qh_from = pd.Timestamp("2025-10-01")
if last_year.index[0] >= _qh_from:
    _period_word = "15-minute period"
elif last_year.index[-1] < _qh_from:
    _period_word = "hour"
else:
    _period_word = "delivery period (an hour before 1 Oct 2025, 15 minutes from then)"
_full_q = by_q[[not (p == by_q.index[-1] and day < p.end_time.normalize()) for p in by_q.index]]
_bars_txt = (f"; the full quarters shown range from €{_full_q['margin'].min() / 1e6:.1f}m to "
             f"€{_full_q['margin'].max() / 1e6:.1f}m" if len(_full_q) else "")
st.caption("By construction this margin cannot be negative: the plant runs in exactly the periods whose "
           "price covers its fuel and carbon cost (as a unit offering at that cost would be dispatched in the "
           "auction) and pays nothing to start, stop or stay on, so the bars are an upper bound, not a finding "
           "about what gas plants earn. Days with no EUA auction in the previous two weeks (around the year-end "
           "auction break) are left out, so first quarters have fewer days than the others."
           + (" From 2025Q4 prices are per 15 minutes instead of hourly; the finer periods alone raise the margin "
              f"by {_res_lo:.0f} to {_res_hi:.0f}% (the same prices averaged to hours, 2025Q4 onwards), so quarters "
              "before and after are not fully comparable." if pd.notna(_res_lo) else "")
           + " What it does show is that the average peak spread hides profitable hours. It ignores start-up "
           f"costs: the plant would start {_starts_day:.1f} times a day in the {period_label}"
           + (f", and {_single:.0%} of its {_n_runs:,} runs last a single {_period_word}" if _n_runs else "")
           + ". At NREL's lower-bound median wear-and-tear cost for a gas combined-cycle start (35 dollars per MW "
           "for a hot start, 55 for a warm start, 2011 dollars, excluding start fuel; Kumar et al., Power "
           "Plant Cycling Costs, NREL 2012, Table 1-1), that many starts would cost roughly "
           f"{_start_lo:.1f} to {_start_hi:.1f} million dollars a quarter for {PLANT_MW} MW (2011 dollars, not "
           f"converted to euros or to today's prices{_bars_txt}). Real plants avoid this by running through some "
           "loss-making hours or keeping a minimum output.")

with st.spinner("Fetching German gas-fired generation (Energy-Charts)..."):
    gas_mw, gen_errors = power_data.load_gas_generation(day - pd.Timedelta(days=120), day)
captured = power_data.gas_weighted_price(prices, gas_mw)
if captured.empty:
    st.caption("German gas-fired generation could not be loaded, so the gas-weighted price is not shown."
               + (" " + "; ".join(gen_errors) if gen_errors else ""))
else:
    cmp = pd.DataFrame({"Gas-weighted": captured, "Peak": df["Power peak"], "Base": df["Power base"]}).dropna()
    cmp = cmp[cmp.index > day - pd.Timedelta(days=90)]
    df["Gas-weighted price"] = captured.reindex(df.index)
    if not cmp.empty:
        last30 = cmp[cmp.index > cmp.index[-1] - pd.Timedelta(days=30)]
        _to = f"30 days to {cmp.index[-1]:%d %b}"
        g1, g2 = st.columns(2)
        g1.metric(f"Gas-output-weighted price minus peak price, {_to}",
                  f"{(last30['Gas-weighted'] - last30['Peak']).mean():+.2f} €/MWh",
                  help="Average over delivery business days with complete generation data (it lags the "
                       "price data by a few days). Positive: gas plants produced more in the hours when "
                       "the day-ahead price was above the peak-block average.")
        g2.metric(f"Gas-output-weighted price minus baseload price, {_to}",
                  f"{(last30['Gas-weighted'] - last30['Base']).mean():+.2f} €/MWh")
        fig_g, ax_g = plt.subplots(figsize=(14, 3.5))
        ax_g.plot(cmp.index, cmp["Gas-weighted"], color="firebrick", linewidth=1.2,
                  label="Weighted by actual gas-fired output (capture price)")
        ax_g.plot(cmp.index, cmp["Peak"], color="black", linewidth=0.8, label=f"Peak ({_PEAK})")
        ax_g.plot(cmp.index, cmp["Base"], color="gray", linewidth=0.6, label="Baseload (24 h)")
        ax_g.set_ylabel("€/MWh")
        ax_g.set_title("Day-ahead price weighted by German gas-fired output, against the peak and baseload "
                       "averages (last 90 days, delivery business days)")
        ax_g.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig_g)
        _gm = gas_mw[gas_mw.index.tz_localize(None).normalize() > day - pd.Timedelta(days=90)].dropna()
        _gm_txt = (f"German gas plants produced in every period of the last 90 days (lowest output "
                   f"{_gm.min():,.0f} MW), and more when prices were high. "
                   if len(_gm) and _gm.min() > 0 else "German gas plants produce more when prices are high. ")
        st.caption("A reality check on the peak block: " + _gm_txt +
                   "It is a day-ahead capture price, not what the plants sold at: output sold forward "
                   "earns the forward price instead. Generation: Energy-Charts (Fraunhofer ISE) public net generation, fossil gas; "
                   "power plants for the public supply only, industrial self-generation not included.")

export = pd.DataFrame({
    "Delivery date": df.index.strftime("%Y-%m-%d"),
    "DE power price, peak (EUR/MWh)": df["Power peak"].round(2).values,
    "DE power price, baseload (EUR/MWh)": df["Power base"].round(2).values,
    "TTF front month (EUR/MWh, HHV)": df["TTF"].round(3).values,
    "TTF price date": df["TTF date"].dt.strftime("%Y-%m-%d").values,
    "TTF price type": [_PRICE_TYPE[common.futures_row_state(d)] for d in df["TTF date"]],
    "EUA auction price (EUR/tCO2)": df["EUA"].round(2).values,
    "EUA auction date": df["EUA date"].dt.strftime("%Y-%m-%d").values,
    "Fuel cost at 50% (EUR/MWh)": df["Fuel cost"].round(3).values,
    "Carbon cost at 50% (EUR/MWh)": df["Carbon cost"].round(3).values,
    **{f"Clean spark spread, peak, {_eff(e)} (EUR/MWh)": df[f"CSS peak {_eff(e)}"].round(3).values
       for e in EFFICIENCIES},
    **{f"Clean spark spread, baseload, {_eff(e)} (EUR/MWh)": df[f"CSS base {_eff(e)}"].round(3).values
       for e in EFFICIENCIES},
    "Flexible plant, hours run": df["Run hours"].round(2).values,
    "Flexible plant, starts": df["Starts"].values,
    "Flexible plant, average spread when running (EUR/MWh)": df["Run spread"].round(3).values,
    "Flexible plant, daily gross margin (EUR)": df["Flex margin"].round(0).values,
    "Day-on-day change in peak spread at 50% (EUR/MWh)": df["dCSS"].round(3).values,
    "Day-on-day change in flexible plant margin (EUR)": df["dFlex"].round(0).values,
    "Gas-weighted day-ahead price (EUR/MWh)": (df["Gas-weighted price"].round(2).values
                                               if "Gas-weighted price" in df else np.nan),
})
st.download_button("Download CSV", data=export.to_csv(index=False), file_name="clean_spark_spread_de.csv",
                   mime="text/csv",
                   help=f"Daily data behind the page ({len(export):,} delivery business days), including the dates "
                        "of the TTF price and EUA auction used for each day, so every figure can be recomputed. "
                        "The TTF price type follows the row's date in New York: a price of today's New York date is "
                        "the latest price, not the settlement; one of the day before, Monday to Thursday, may not yet "
                        f"be the settlement ({common._UNSETTLED_WHY}); older ones, and Friday's from Saturday on, "
                        "are closes.")

# ─── Risk: day-on-day changes ───
st.subheader("Spread Risk")
recent = df["dCSS"].dropna().tail(_LOOKBACK)
css_vol_now = df["dCSS vol"].dropna().iloc[-1]
css_fall95 = -np.percentile(recent, 5)          # 1-in-20 day-on-day fall, €/MWh (positive number)
flex_recent = df["dFlex"].dropna().tail(_LOOKBACK)
flex_fall95 = max(0.0, -np.percentile(flex_recent, 5))

# Stress period: the same 1-in-20 falls on the 12 months since the start of the power data in which they were
# largest (common.stress_period_fall, the Risk page's window rule). Always searched from the first year of
# power data, whatever start date is chosen, and only up to this page's last delivery day.
_stress_from = pd.Timestamp(power_data.FIRST_YEAR, 1, 1).date()
if start_date <= _stress_from:
    _full = df
else:
    with st.spinner("Loading the power history for the stress period..."):
        _p_all, _, _ = power_data.load_power(_stress_from, _power_end)
        _ttf_all = common.load_close("TTF=F", _stress_from - pd.Timedelta(days=10), end_date).dropna()
        _eua_all, _ = power_data.load_eua_auctions(_stress_from - pd.Timedelta(days=30), end_date)
        _full, _ = (_build_frame(_p_all, _ttf_all, _eua_all) if not (_p_all.empty or _eua_all.empty)
                    else (pd.DataFrame(), None))
_full = _full[_full.index <= day] if len(_full) else _full
sp_css = common.stress_period_fall(_full["dCSS"]) if len(_full) else None
sp_flex = common.stress_period_fall(_full["dFlex"]) if len(_full) else None

r1, r2, r3 = st.columns(3)
r1.metric("30-day volatility of the day-on-day change in peak spread", f"{css_vol_now:.2f} €/MWh",
          help="Standard deviation of the last 30 day-on-day changes (consecutive delivery business days).")
r2.metric("1-in-20 day-on-day fall in the peak spread", f"{css_fall95:.2f} €/MWh",
          help=f"5th percentile of the last {len(recent)} day-on-day changes in the peak spread at 50%, shown as "
               "a positive number (95% historical VaR of the daily change). Monday is compared with Friday.")
r3.metric(f"1-in-20 day-on-day fall in the {PLANT_MW} MW flexible plant's daily margin", f"€{flex_fall95:,.0f}",
          help=f"5th percentile of the last {len(flex_recent)} day-on-day changes in the flexible plant's "
               f"daily gross margin ({PLANT_MW} MW × hours run × average spread when running).")


def _peak_mean(changes_index):
    return _full.loc[changes_index, "Power peak"].mean()


if sp_css is not None:
    _sp_flex_fall = max(0.0, sp_flex["fall"]) if sp_flex is not None else None
    s0, s1, s2 = st.columns(3)
    s0.markdown(f"**Stress period**: the {common.STRESS_WINDOW} consecutive day-on-day changes (about 12 months) "
                f"since {_full.index[0]:%b %Y} in which each 1-in-20 fall was largest")
    s1.metric(f"Stress period: 1-in-20 fall in the peak spread, {sp_css['start']:%d %b %Y} to {sp_css['end']:%d %b %Y}",
              f"{sp_css['fall']:.2f} €/MWh",
              help="Same measure as the figure above, on the stress period's changes instead of the last "
                   f"{len(recent)}. In €/MWh, not divided by the price level.")
    s1.caption(f"{sp_css['fall'] / css_fall95:.2f}x the last {len(recent)} changes ({css_fall95:.2f} €/MWh)"
               if css_fall95 > 0 else "")
    if sp_flex is not None:
        s2.metric(f"Stress period: 1-in-20 fall in the flexible plant's margin, {sp_flex['start']:%d %b %Y} to "
                  f"{sp_flex['end']:%d %b %Y}", f"€{_sp_flex_fall:,.0f}",
                  help="Same measure as the figure above, on the stress period's changes instead of the last "
                       f"{len(flex_recent)}. The two stress periods are searched separately, so their dates can differ.")
        s2.caption(f"{_sp_flex_fall / flex_fall95:.2f}x the last {len(flex_recent)} changes (€{flex_fall95:,.0f})"
                   if flex_fall95 > 0 else "")

    def _stress_text(sp, name, fmt):
        _w = _full["dCSS" if name == "peak spread" else "dFlex"].dropna()
        _w = _w[(_w.index >= sp["start"]) & (_w.index <= sp["end"])]
        out = (f"Peak spread: {sp['start']:%d %b %Y} to {sp['end']:%d %b %Y}" if name == "peak spread" else
               f"Flexible plant: {sp['start']:%d %b %Y} to {sp['end']:%d %b %Y}")
        out += f", 1-in-20 fall {fmt(sp['fall'])}"
        if sp["ties"] > 1:
            out += (f"; {sp['ties']} windows ending {sp['first_tie_end']:%d %b %Y} to {sp['end']:%d %b %Y} have "
                    "exactly this value, because the 5th percentile of 250 changes depends only on the 13th and "
                    "14th largest falls and these windows all contain them; the latest is shown")
        ru = sp["runner_up"]
        # Name the next-highest window only when it comes within 1%: then the stress period's dates can jump.
        if ru is not None and sp["fall"] - ru["fall"] < 0.01 * abs(sp["fall"]):
            out += (f". The next-highest value, {fmt(ru['fall'])}, is only {fmt(sp['fall'] - ru['fall'])} lower and "
                    f"belongs to {ru['n']} window{'s' if ru['n'] > 1 else ''} ending "
                    + (f"between {ru['first_end']:%d %b %Y} and {ru['last_end']:%d %b %Y}" if ru['n'] > 1
                       else f"{ru['first_end']:%d %b %Y}")
                    + ", so a small change in the data could move the stress period to those dates")
        out += (f". Average peak power price on the window's delivery days: {_peak_mean(_w.index):.2f} €/MWh "
                f"(last {len(recent)} changes: {_peak_mean(recent.index):.2f} €/MWh)")
        return out + "."

    st.caption(
        _stress_text(sp_css, "peak spread", lambda x: f"{x:.2f} €/MWh") + " "
        + (_stress_text(sp_flex, "flexible plant", lambda x: f"€{max(0.0, x):,.0f}") + " " if sp_flex else "")
        + f"Each stress period is the run of {common.STRESS_WINDOW} consecutive day-on-day changes, within "
        f"{common.STRESS_MAX_SPAN} calendar days, with the largest 1-in-20 fall, searched over every change from "
        f"the first in the power data ({sp_css['search_from']:%d %b %Y}) to the last on this page "
        f"({sp_css['search_to']:%d %b %Y}), whatever start date is chosen and never past the end date; when "
        "several windows tie, the latest is taken. The figures are in €/MWh and €, not divided by the price level: "
        "the power price is given only as background. Stress-period figures borrow the Basel method (bank capital "
        "rules): Basel picks the 12 months of largest loss for the current portfolio, equally weighted, searching "
        "back to 2007, with a 10-day horizon and liquidity scaling (BCBS MAR33.4, MAR33.7); here the measure is the 1-day "
        "1-in-20 fall, the search starts with the power data in 2020, and the window rule is the Risk page's."
    )
else:
    st.caption(f"No stress period: the power data since {_stress_from:%b %Y} has no run of "
               f"{common.STRESS_WINDOW} day-on-day changes within {common.STRESS_MAX_SPAN} calendar days.")

fig3, ax3 = plt.subplots(figsize=(14, 3.5))
_vol = common.break_gaps(df["dCSS vol"])
ax3.plot(_vol.index, _vol.values, color="purple", linewidth=1)
ax3.fill_between(_vol.index, _vol.values, 0, color="purple", alpha=0.1)
ax3.set_ylabel("€/MWh")
ax3.set_title("30-day volatility of the day-on-day change in the peak clean spark spread (50%)")
plt.tight_layout()
st.pyplot(fig3)

st.caption(
    f"These figures measure how much the margin moves from one delivery day to the next. They are not a "
    f"loss on a position: each day's margin is fixed once the day-ahead auction clears, and output and fuel "
    f"bought or sold forward are not exposed to it. The flexible plant cannot earn less than zero on a day (it stays "
    f"off), so a 1-in-20 fall of €{flex_fall95:,.0f} means a good day followed by a much weaker one. The changes "
    "include the days on which `TTF=F` switches to the next monthly contract: part of that day's change is the "
    "price gap between two contracts, and the fuel cost (TTF ÷ 0.5) doubles it."
)
st.caption(
    "**Caveats.** Power is the day-ahead price, while gas is the TTF front-month future, so the tenors "
    "don't match. S&P Global Platts calculates its German spark spreads using its own TTF assessments, which are "
    "not freely available; a change to its day-ahead and weekend assessments in March 2026 affected 12 day-ahead "
    "spread codes (S&P Global subscriber note, 31 Mar 2026). Platts prices "
    "carbon with its EUA assessment of the nearest December contract; this page uses the EEX primary "
    "auction (spot) instead. Efficiencies are on a higher heating value basis "
    "(50% HHV is roughly 55% on a lower heating value basis). German plants buy gas at the THE hub, "
    "which trades at a spread to TTF; like Platts, the page uses TTF. Fuel cost excludes gas transport and other "
    "variable costs, as in the Platts spreads. The peak block includes public holidays, as exchange peak "
    "products do. `TTF=F` is a continuous front-month series from Yahoo Finance that jumps when the "
    "contract rolls each month and follows the NYMEX (US) trading calendar."
)

# ─── Gas purchase timing: buy next month's gas now or in a week (gas leg only) ───
st.subheader("Buy Next Month's Gas Now, or in a Week? (gas leg only)")
_WAIT_DAYS = 5
_MIN_FIT = 250   # daily returns a monthly contract needs in the fit window to have its own GARCH fit
_MIN_CMP = 20    # daily returns on common days needed to compare TTF=F's and a contract's volatility
_SAME_VOL = 0.05  # volatilities within 5% of each other are "about the same"
_ROLL_CHECK = 5   # last returns of the front-month fit checked for a contract roll day
ttf_last_date = ttf.index[-1]
# GARCH fit window: it does not shrink with a later start date. It starts 10 days before 1 Jan 2020 (the default
# start date) or before the start date if that is earlier, as the page's TTF data does, and ends at the end date;
# all prices come from the shared download.
_fit_from = min(pd.Timestamp(start_date), pd.Timestamp(common.DEFAULT_START)) - pd.Timedelta(days=10)
_ttf_fit = common.load_close("TTF=F", _fit_from.date(), end_date).dropna()
# The decision day is the end date (the last weekday on or before it), even when its TTF price is not out yet.
_decide = max(pd.Timestamp(np.busday_offset(end_date, 0, roll="backward")), ttf_last_date.normalize())
_wait_day = pd.Timestamp(np.busday_offset(_decide.date(), _WAIT_DAYS))
_front = common.front_month_contract("TTF=F", ttf_last_date)
_front_exp = common.contract_expiry("TTF", _front["delivery"])
# The front contract stops trading before the plant could buy in a week: switch to the next contract that
# is still trading on the wait day (contract_expiry, on the exchange holiday calendars).
_switch = _wait_day.normalize() > _front_exp.normalize()
_batch = common.shared_download("TTF=F", _fit_from)
_c = pd.Series(dtype=float)
if _switch:
    _next = pd.Timestamp(_front["delivery"]) + pd.DateOffset(months=1)
    while common.contract_expired("TTF", _next, _wait_day):
        _next += pd.DateOffset(months=1)
    delivery_month = _next.date()
    contract = common.contract_symbol("TTF", delivery_month)
    if contract in _batch:
        _c = _batch[contract].dropna()
        _c = _c[(_c.index >= _fit_from) & (_c.index <= ttf_last_date)]
        _c = _c.mask(common._stale_mask(_c)).dropna()
    if _c.empty or (ttf_last_date - _c.index[-1]).days > _TTF_MAX_AGE:
        _mode = "none"      # no recent price for the next contract
    elif len(_c) > _MIN_FIT:
        _mode = "own"       # the next contract's own price and its own GARCH fit
    else:
        _mode = "approx"    # its own price, with the range fitted on TTF=F
else:
    delivery_month = _front["delivery"]
    contract = _front["symbol"]
    _mode = "front"
month_label = f"{delivery_month:%b %Y} delivery"
run_h = last_year["Run hours"].tail(30).mean()
_, n_days = calendar.monthrange(delivery_month.year, delivery_month.month)
n_bdays = int(np.is_busday(np.arange(np.datetime64(delivery_month),
                                     np.datetime64(delivery_month) + np.timedelta64(n_days, "D"))).sum())
gas_mwh = PLANT_MW * run_h * n_bdays / EFFICIENCY
# A TTF month contract delivers equally in every hour from 06:00 on the 1st to 06:00 on the 1st of the next
# month, local time, so the month has 743, 744 or 745 hours when the clocks change (ICE Endex contract specs).
_m0 = pd.Timestamp(delivery_month).tz_localize(None) + pd.Timedelta(hours=6)
_hours_in_month = int(round((_m0 + pd.DateOffset(months=1)).tz_localize("Europe/Amsterdam").value / 3.6e12
                            - _m0.tz_localize("Europe/Amsterdam").value / 3.6e12))
contract_mw = gas_mwh / _hours_in_month   # TTF month futures deliver the same MW in every hour

# Power and gas: correlation of month-on-month changes, complete months only, at least 24 changes. If the
# selected range is too short, the same figure from the whole power history up to the end date is used.
def _monthly_changes(frame, first):
    m = frame[["Power peak", "TTF"]].resample("MS").mean().dropna()
    m = m[(m.index + pd.offsets.MonthEnd(0) <= day) & (m.index >= pd.Timestamp(first))]
    return m.diff().dropna().tail(36)


_dm, _dm_note = _monthly_changes(df, start_date), ""
if len(_dm) < 24 and len(_full):
    _dm_note = (f"; the selected range has only {len(_dm)} such changes, so this uses the whole power history "
                "up to the end date")
    _dm = _monthly_changes(_full, _full.index[0].replace(day=1))
if len(_dm) >= 24:
    _corr_txt = (f"power and gas prices tend to move together (correlation of the month-on-month changes in average "
                 f"day-ahead peak power and TTF, {len(_dm)} complete months to {_dm.index[-1]:%b %Y}: "
                 f"{_dm['Power peak'].corr(_dm['TTF']):.2f}{_dm_note}; that is day-ahead power, not the forward "
                 "power a hedging plant would sell)")
else:
    _corr_txt = (f"power and gas prices may move together (there are only {len(_dm)} month-on-month changes of "
                 "complete months, fewer than the 24 needed to show a correlation)")

if _switch:
    st.warning(f"The front-month contract, TTF {_front['delivery']:%b %y}, "
               f"{'stopped' if _front_exp.normalize() < _decide else 'stops'} trading on "
               f"{_front_exp:%a %d %b %Y} (its last trading day by the exchange rule and holiday calendar), "
               f"before {_wait_day:%a %d %b}, {_WAIT_DAYS} business days after {_decide:%a %d %b}: the plant cannot "
               f"wait a week to buy {_front['delivery']:%B} gas on it. Once that contract has stopped trading, "
               f"{_front['delivery']:%B} gas can still be bought for shorter delivery periods, such as the TTF day, "
               "weekend and within-day spot contracts EEX lists (EEX contract specifications 0105a, section 1.2.4.1), "
               "but this page has no prices for them. This section therefore uses the next contract still trading "
               f"then, TTF {delivery_month:%b %y} (`{contract}`): its price, delivery month, days and hours.")
st.write(f"The plant buys the gas for {month_label} on TTF {delivery_month:%b %y}"
         + (", the front month" if not _switch else "") + " (price from the NYMEX Dutch TTF contract)"
         + f". If it waits {_WAIT_DAYS} business days instead of "
         f"buying on {_decide:%a %d %b}, how much could that gas cost? This is the gas leg alone: a plant that "
         f"hedges its spread also sells its power forward at the same time, and {_corr_txt}, so the plant's real "
         "exposure is smaller.")

if _mode == "none":
    wait = None
    st.caption(f"No price for `{contract}` within {_TTF_MAX_AGE} days of {ttf_last_date:%d %b %Y} in this page's "
               "data, so no range is shown: the page downloads the monthly contracts from two months before today to "
               "five months after, and Yahoo Finance no longer lists expired ones.")
else:
    fit_series = _c if _mode == "own" else _ttf_fit
    price_now = float(_c.iloc[-1]) if _mode in ("own", "approx") else float(ttf.iloc[-1])
    price_date = _c.index[-1] if _mode in ("own", "approx") else ttf_last_date
    _horizon = max(_WAIT_DAYS, int(np.busday_count(price_date.date(), _wait_day.date())))
    wait = common.garch_price_range(fit_series, horizon=_horizon)
    if wait is None:
        st.caption("TTF price range unavailable: the GARCH model could not be fitted on this date range.")
if wait is not None:
    # The range is fitted on fit_series and applied to price_now; scale is 1 except in the approximation,
    # where TTF=F's relative range is applied to the next contract's price.
    scale = price_now / wait["price_now"]
    lo, hi = wait["lo"][-1] * scale, wait["hi"][-1] * scale
    bill_now = gas_mwh * price_now
    bill_lo, bill_hi = gas_mwh * lo, gas_mwh * hi   # unrounded quantiles; rounded only for display
    fit_start = fit_series.dropna().index[0]

    w1, w2, w3 = st.columns(3)
    _w_state = common.futures_row_state(price_date)
    if _switch:
        _price_help = f"Yahoo Finance `{contract}`, the monthly contract itself, from the same download as `TTF=F`."
    elif _front["matched"]:
        _price_help = f"Yahoo Finance `TTF=F`, matched to the monthly contract `{contract}` on this date."
    else:
        _price_help = ("Yahoo Finance `TTF=F`. Its price could not be matched to a single monthly contract on Yahoo "
                       "Finance for this date, so the delivery month is the front month by the exchange expiry rule "
                       f"(`{contract}`, {common.expiry_cal_note("TTF")}).")
    _w_label = {"live": f"latest price {price_date:%d %b %Y}",
                "unsettled": f"price of {price_date:%d %b %Y} (may not yet be the settlement)",
                "close": f"close {price_date:%d %b %Y}"}[_w_state]
    w1.metric(f"TTF {delivery_month:%b %y} ({month_label}), {_w_label}", f"{price_now:.2f} €/MWh",
              help=_price_help + {"live": " This is the latest price, not the settlement.",
                                  "unsettled": f" It may not yet be the settlement: {common._UNSETTLED_WHY}.",
                                  "close": ""}[_w_state])
    w2.metric(f"On {_wait_day:%a %d %b}, in {_WAIT_DAYS} business days: 90% range (€/MWh)",
              f"{_half(lo):.1f} to {_half(hi):.1f}",
              help="Rounded to 0.5 €/MWh: the range comes from 10,000 simulated paths with a fixed random seed "
                   "(the same prices give the same range; another seed moves the ends by a few tenths).")
    w3.metric(f"Std. dev. of the {_horizon}-day return", f"{wait['horizon_vol']:.1f}%",
              help=f"Standard deviation of the simulated log return from the last price ({price_date:%d %b}) to "
                   f"{_wait_day:%d %b}, {_horizon} business days.")

    b1, b2, b3 = st.columns(3)
    b1.metric("The month's gas, bought at the last price", f"€{bill_now / 1e6:,.2f}m",
              help=f"{gas_mwh:,.0f} MWh of gas: {PLANT_MW} MW × {run_h:.1f} h a day (the flexible plant's "
                   f"average over the last 30 delivery business days) × {n_bdays} business days in "
                   f"{delivery_month:%B %Y} ÷ {EFFICIENCY:.0%} efficiency. A TTF month future delivers the same "
                   f"amount in every hour of the month ({_hours_in_month} hours from 06:00 on the 1st, local time; "
                   f"trading unit 1 MW), so this volume is about {contract_mw:,.0f} MW of the contract; the plant's "
                   "actual use varies by hour and is only on business days, so the contract matches it only on "
                   "average.")
    b2.metric("Bought in a week, 90% range", f"€{bill_lo / 1e6:,.1f}m to €{bill_hi / 1e6:,.1f}m",
              help=f"{gas_mwh:,.0f} MWh times the 90% range, from the simulated percentiles before rounding "
                   f"({lo:.2f} to {hi:.2f} €/MWh), so it can differ by €0.1m from the rounded range times the volume.")
    b3.metric("Extra cost if TTF ends at the 95th percentile", f"+€{(bill_hi - bill_now) / 1e6:,.1f}m",
              help=f"Top of the 90% range ({hi:.2f} €/MWh before rounding) minus the last price, times the volume: a "
                   "1-in-20 threshold, not the average of the worst 5%.")

    fig_w, ax_w = plt.subplots(figsize=(14, 3.5))
    hist = (_c if _switch else ttf).tail(40)
    days_ahead = pd.bdate_range(price_date, periods=_horizon + 1)
    ax_w.plot(hist.index, hist.values, color="steelblue", linewidth=1.2,
              label=f"TTF {delivery_month:%b %y} ({contract})" if _switch else "TTF front month (TTF=F)")
    ax_w.fill_between(days_ahead, np.r_[price_now, wait["lo"] * scale], np.r_[price_now, wait["hi"] * scale],
                      color="steelblue", alpha=0.2, label="90% range if you wait")
    ax_w.plot(days_ahead, np.r_[price_now, wait["median"] * scale], color="steelblue",
              linewidth=1, linestyle="--", label="Median (≈ the last price)")
    ax_w.set_ylabel("€/MWh")
    ax_w.set_title(f"TTF {delivery_month:%b %y}: last 40 business days and the range to {_wait_day:%d %b %Y}")
    ax_w.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_w)

    # Volatility of TTF=F and of the contract on the same days of the fit window, from the same download (data,
    # not a rule): log returns on the days on which both have a price.
    _both = (np.log(pd.DataFrame({"TTF=F": _ttf_fit, contract: _c}).dropna()).diff().dropna()
             if _switch and len(_c) else pd.DataFrame())
    _fit_note = (" The fit window does not shrink with a later start date: it starts 10 days before 1 Jan 2020, or "
                 "before the start date if that is earlier (as the page's TTF data does), and ends at the end date.")
    if _mode == "own":
        _recent = _both.tail(_LOOKBACK)
        _cmp = (f" Over the last {len(_recent)} trading days on which both have a price, `TTF=F` moved "
                f"{_recent['TTF=F'].std() * common.ANNUAL * 100:.0f}% annualised and `{contract}` "
                f"{_recent[contract].std() * common.ANNUAL * 100:.0f}%." if len(_recent) >= _MIN_CMP else "")
        _method = (f"GARCH(1,1) with Student-t shocks fitted on the daily log returns of `{contract}` itself from "
                   f"{fit_start:%d %b %Y} to {price_date:%d %b %Y}, not on `TTF=F`, which still follows the "
                   "expiring contract and jumps when it rolls." + _cmp + " In its earlier years the contract was "
                   "further from delivery than it is now." + _fit_note)
    elif _mode == "approx":
        # Compared on the same days of the same fit window, so the direction is the one this sample shows.
        if len(_both) >= _MIN_CMP:
            _v_f, _v_c = (_both[c].std() * common.ANNUAL * 100 for c in ("TTF=F", contract))
            _verdict = ("about the same" if abs(_v_f / _v_c - 1) < _SAME_VOL else
                        "so with `TTF=F` the more volatile, the range is likely too wide" if _v_f > _v_c else
                        "so with `TTF=F` the less volatile, the range is likely too narrow")
            _cmp = (f" On the {len(_both)} days of the fit window on which both have a price ({_both.index[0]:%d %b %Y} "
                    f"to {_both.index[-1]:%d %b %Y}), `TTF=F` moved {_v_f:.0f}% annualised and `{contract}` "
                    f"{_v_c:.0f}%, " + _verdict + ".")
        else:
            _cmp = (f" With only {len(_both)} daily returns of both on the same days, fewer than {_MIN_CMP}, the page "
                    "cannot say whether the range is too wide or too narrow.")
        _method = (f"Approximation: `{contract}` has {max(len(_c) - 1, 0)} daily returns in the fit window "
                   f"({_c.index[0]:%d %b %Y} to {_c.index[-1]:%d %b %Y}), fewer than the {_MIN_FIT} needed for its own "
                   "fit, so the range is the one fitted on `TTF=F` (the expiring front month, roll jumps included, "
                   f"{fit_start:%d %b %Y} to {wait['date']:%d %b %Y}) applied to this contract's price. It is not this "
                   "contract's own volatility." + _cmp + _fit_note)
    else:
        # A roll day among the last returns of the fit weighs most on the GARCH variance at the end of the sample.
        # Only days the roll check identifies (common.window_rolls: TTF=F's close equals a new monthly contract's)
        # are named with a range without them; when Yahoo no longer lists the contracts, a day on the exchange rule's
        # expiry or the day after is named as a possible roll (not verified) and no range is given.
        _roll_txt = ""
        _last_r = _ttf_fit.pct_change(fill_method=None).dropna().tail(_ROLL_CHECK)
        _rc = common.window_rolls("TTF=F", _batch, _last_r)
        if _rc and _rc["rolls"]:
            _rdays = tuple(t for t, _, _, _ in _rc["rolls"])
            _alt = common.garch_price_range(fit_series, horizon=_horizon, drop=_rdays)
            _one = len(_rdays) == 1
            _roll_txt = (f" The last {_ROLL_CHECK} returns of the fit include "
                         + ("a contract roll day, " if _one else "contract roll days, ")
                         + " and ".join(f"{t:%a %d %b %Y} (`TTF=F` {cr * 100:+.2f}%, the "
                                        f"{common.contract_delivery(m, 'TTF'):%b %y} contract it switched to "
                                        f"{nr * 100:+.2f}%)" for t, m, cr, nr in _rc["rolls"])
                         + ": part of " + ("that day's return" if _one else "those days' returns")
                         + " is the gap between two contracts.")
            if _alt is not None:
                _s_alt = price_now / _alt["price_now"]
                _roll_txt += (f" Fitted again on the same series, the range with {'that day' if _one else 'those days'} "
                              f"left out would be {_half(_alt['lo'][-1] * _s_alt):.1f} to "
                              f"{_half(_alt['hi'][-1] * _s_alt):.1f} €/MWh (same seed and number of paths).")
        elif _rc and _rc["unknown"]:
            # Not matched to a listed contract: name a possible roll on the exchange-rule days, without a range.
            _cand = common.rule_roll_days("TTF", _rc["unknown"])
            if _cand:
                _roll_txt = (" " + common.rule_roll_sentence(_cand) + f" These days are among the last {_ROLL_CHECK} "
                             "returns of the fit, which weigh most on the fitted volatility; no range without them "
                             "is given, because the roll is not identified.")
        _method = (f"GARCH(1,1) with Student-t shocks fitted on daily log returns of `TTF=F` from "
                   f"{fit_start:%d %b %Y} to {price_date:%d %b %Y}. Monthly roll jumps in `TTF=F` feed into the "
                   "fitted volatility." + _roll_txt + _fit_note)
    st.caption(
        "**This is the risk of waiting, not a forecast of direction.** The model has no view on whether "
        "TTF goes up or down: its mean return is set to zero, so the range is centred on the last price. "
        f"Method: {_method} 10,000 simulated paths over {_horizon} business days"
        + (f" (from the last price, {price_date:%a %d %b}, to {_wait_day:%a %d %b})" if _horizon != _WAIT_DAYS
           else "")
        + ", 5th to 95th percentile. The front-month contract's last trading day comes from the exchange rule "
        "(NYMEX Dutch TTF, CME rule 1159102.E, normally the same day as ICE Endex TTF: two UK business days "
        "before the delivery month), with " + common.expiry_cal_note("TTF") + ". The volume "
        "assumes the plant runs as the flexible plant has recently; a spot (day-ahead) gas comparison is "
        "not shown because no freely licensed daily TTF spot series is available."
    )

_by_source = {}
for _y, _s in sorted(power_sources.items()):
    _by_source.setdefault(_s, []).append(_y)
_src = "; ".join(f"{s} ({ys[0]}{' to ' + str(ys[-1]) if len(ys) > 1 else ''})" for s, ys in _by_source.items())
st.caption(f"Power, DE-LU day-ahead: {_src}.")
st.caption(f"EUA: EEX primary-auction clearing prices, €/tCO2 ({eua_auctions.index.min():%d %b %Y} to "
           f"{eua_auctions.index.max():%d %b %Y}), carried forward at most two weeks between auctions. The "
           "auction is for spot allowances, so it can differ from the December future the market quotes.")
