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
# (Platts European Electricity specifications, p.15). 50% is the central case.
EFFICIENCY = 0.50
EFFICIENCIES = (0.45, 0.50, 0.60)
# Emission intensity of natural gas on the same HHV basis as TTF prices, which are quoted per MWh
# of gross calorific value: Platts uses 0.053942 tCO2e/MMBtu (HHV) = 0.053942 x 3.412142
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
      weekends, so each delivery day is matched with the last TTF settlement and EUA auction
      **before** it: Friday's prices for Monday, the latest available when Sunday's auction runs.
    - **Clean spark spread** = power − gas / efficiency − ({GAS_EF:.4f} / efficiency) × EUA,
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
    _power_end = (end_date + pd.Timedelta(days=1)
                  if end_date >= pd.Timestamp.now(tz="Europe/Berlin").date() else end_date)
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


# Delivery business days (the peak block exists on weekdays only).
daily = power_data.daily_base_peak(prices)
days = daily.index[daily.index.dayofweek < 5]
df = pd.DataFrame(index=days)
df["Power peak"] = daily["Peak"].reindex(days)
df["Power base"] = daily["Base"].reindex(days)
df["TTF"], df["TTF date"] = _last_before(ttf, days, _TTF_MAX_AGE)
df["EUA"], df["EUA date"] = _last_before(eua_auctions, days, _EUA_MAX_AGE)
df = df.dropna(subset=["Power peak", "Power base", "TTF", "EUA"])
if len(df) < _LOOKBACK // 5:
    st.warning("Not enough overlapping power, gas and carbon data for this date range; widen it.")
    st.stop()


def _cost(eff, frame):
    return frame["TTF"] / eff + GAS_EF / eff * frame["EUA"]


df["Fuel cost"] = df["TTF"] / EFFICIENCY
df["Carbon cost"] = CARBON_PER_MWH * df["EUA"]
cost = df["Fuel cost"] + df["Carbon cost"]
df["CSS peak"] = df["Power peak"] - cost
df["CSS base"] = df["Power base"] - cost
for _e in EFFICIENCIES:
    df[f"CSS peak {_eff(_e)}"] = df["Power peak"] - _cost(_e, df)
    df[f"CSS base {_eff(_e)}"] = df["Power base"] - _cost(_e, df)
flex = power_data.flexible_run(prices, cost)
df["Run hours"] = flex["Run hours"].reindex(df.index)
df["Run spread"] = flex["Run spread"].reindex(df.index)
df["Starts"] = flex["Starts"].reindex(df.index)
# Daily gross margin of the flexible plant, € (0 on days it stays off).
df["Flex margin"] = (df["Run hours"] * df["Run spread"]).fillna(0.0) * PLANT_MW
# Day-on-day changes between consecutive delivery business days; a change across a data gap of
# more than a week is not a one-day move, so it is left out.
_gap = df.index.to_series().diff() > pd.Timedelta(days=7)
df["dCSS"] = df["CSS peak"].diff().mask(_gap)
df["dCSS vol"] = df["dCSS"].rolling(30, min_periods=20).std()
df["dFlex"] = df["Flex margin"].diff().mask(_gap)

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
_hl_month, _ = common.front_month_contract("TTF=F", "TTF", latest["TTF date"], latest["TTF"])
_hl_contract = f"TTF {_hl_month:%b %y} future (front month)" if _hl_month else "TTF front-month future"
_is_today = latest["TTF date"].date() == pd.Timestamp.now(tz="Europe/Berlin").date()
# The day-ahead auction for this delivery day runs the calendar day before; gas and carbon trade on weekdays,
# so the last trading day before the auction is the last weekday before the delivery day.
_trade_day = pd.Timestamp(np.busday_offset(day.date(), -1, roll="backward"))
_gas_lags = latest["TTF date"].normalize() < _trade_day
st.caption(f"Gas: {_hl_contract}, {'latest price' if _is_today else 'close'} of {latest['TTF date']:%a %d %b} "
           f"({latest['TTF']:.2f} €/MWh); not day-ahead gas, see Caveats. Carbon: EEX auction of "
           f"{latest['EUA date']:%a %d %b} ({latest['EUA']:.2f} €/t). Each is the latest price published before "
           f"delivery; the last trading day before this delivery day is {_trade_day:%a %d %b} (Monday's power is "
           "auctioned on Sunday, so Friday's prices are the latest)."
           + (f" **Gas is from an earlier day than that:** Yahoo Finance has no TTF price for "
              f"{_trade_day:%a %d %b} yet, so this spread will change once it does." if _gas_lags else ""))

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
               "(no start-up costs or minimum run times; delivery business days only)")
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
st.caption("By construction this margin cannot be negative: the plant runs in exactly the periods whose "
           "price covers its fuel and carbon cost (as a unit offering at that cost would be dispatched in the "
           "auction) and pays nothing to start, stop or stay on, so the bars are an upper bound, not a finding "
           "about what gas plants earn. From 2025Q4 prices are per 15 minutes instead of hourly; the finer "
           f"periods alone raise the margin by {_res_lo:.0f} to {_res_hi:.0f}% (the same prices averaged to hours, "
           "2025Q4 onwards), so quarters before and after are not fully comparable. What it does show is that the "
           "average peak spread hides profitable hours. It ignores start-up costs: the plant would start "
           f"{_starts_day:.1f} times a day in the {period_label}, many of them for single 15-minute periods. "
           "At NREL's lower-bound median wear-and-tear cost for a gas combined-cycle start (35 dollars per MW "
           "for a hot start, 55 for a warm start, 2011 dollars, excluding start fuel; Kumar et al., Power "
           "Plant Cycling Costs, NREL 2012, Table 1-1), that many starts would cost roughly "
           f"{_start_lo:.1f} to {_start_hi:.1f} million dollars a quarter for {PLANT_MW} MW, a large share of "
           "the bars. Real plants avoid this by running through some loss-making hours or keeping a minimum "
           "output.")

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
        st.caption("A reality check on the peak block: German gas plants run in every period of the day (combined "
                   "heat and power and grid services keep some online), but they produce more when prices are "
                   "high. It is a day-ahead capture price, not what the plants sold at: output sold forward "
                   "earns the forward price instead. Generation: Energy-Charts (Fraunhofer ISE) public net generation, fossil gas; "
                   "power plants for the public supply only, industrial self-generation not included.")

export = pd.DataFrame({
    "Delivery date": df.index.strftime("%Y-%m-%d"),
    "DE power price, peak (EUR/MWh)": df["Power peak"].round(2).values,
    "DE power price, baseload (EUR/MWh)": df["Power base"].round(2).values,
    "TTF front month (EUR/MWh, HHV)": df["TTF"].round(3).values,
    "TTF settlement date": df["TTF date"].dt.strftime("%Y-%m-%d").values,
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
                        "of the TTF settlement and EUA auction used for each day, so every figure can be "
                        "recomputed.")

# ─── Risk: day-on-day changes ───
st.subheader("Spread Risk")
recent = df["dCSS"].dropna().tail(_LOOKBACK)
css_vol_now = df["dCSS vol"].dropna().iloc[-1]
css_fall95 = -np.percentile(recent, 5)          # 1-in-20 day-on-day fall, €/MWh (positive number)
flex_recent = df["dFlex"].dropna().tail(_LOOKBACK)
flex_fall95 = max(0.0, -np.percentile(flex_recent, 5))

r1, r2, r3 = st.columns(3)
r1.metric("30-day volatility of the day-on-day change in peak spread", f"{css_vol_now:.2f} €/MWh",
          help="Standard deviation of the last 30 day-on-day changes (consecutive delivery business days).")
r2.metric("1-in-20 day-on-day fall in the peak spread", f"{css_fall95:.2f} €/MWh",
          help=f"5th percentile of the last {len(recent)} day-on-day changes in the peak spread at 50%, shown as "
               "a positive number (95% historical VaR of the daily change). Monday is compared with Friday.")
r3.metric(f"1-in-20 day-on-day fall in the {PLANT_MW} MW flexible plant's daily margin", f"€{flex_fall95:,.0f}",
          help=f"5th percentile of the last {len(flex_recent)} day-on-day changes in the flexible plant's "
               f"daily gross margin ({PLANT_MW} MW × hours run × average spread when running).")

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
    f"off), so a 1-in-20 fall of €{flex_fall95:,.0f} means a good day followed by a much weaker one."
)
st.caption(
    "**Caveats.** Power is the day-ahead price, while gas is the TTF front-month future, so the tenors "
    "don't match; S&P Global Platts computes its German day-ahead spreads from its own TTF day-ahead "
    "assessments (S&P Global subscriber note, 31 Mar 2026), which are not freely available. Platts prices "
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
ttf_last_date, ttf_last = ttf.index[-1], float(ttf.iloc[-1])
delivery_month, contract = common.front_month_contract("TTF=F", "TTF", ttf_last_date, ttf_last)
month_label = f"{delivery_month:%b %Y} delivery" if delivery_month else "front-month delivery"
run_h = last_year["Run hours"].tail(30).mean()
if delivery_month:
    _, n_days = calendar.monthrange(delivery_month.year, delivery_month.month)
    n_bdays = int(np.is_busday(np.arange(np.datetime64(delivery_month),
                                         np.datetime64(delivery_month) + np.timedelta64(n_days, "D"))).sum())
else:
    n_bdays = 21
gas_mwh = PLANT_MW * run_h * n_bdays / EFFICIENCY
if delivery_month:
    _hours_in_month = n_days * 24
else:
    _hours_in_month = 30 * 24
contract_mw = gas_mwh / _hours_in_month   # TTF month futures deliver the same MW in every hour
_monthly = df[["Power peak", "TTF"]].resample("MS").mean().dropna().tail(36)
_corr, _corr_start, _corr_end = _monthly["Power peak"].corr(_monthly["TTF"]), _monthly.index[0], _monthly.index[-1]
st.write(f"The plant buys the gas for {month_label} on the TTF front-month contract. If it waits "
         f"{_WAIT_DAYS} business days instead of buying today, how much could that gas cost? This is the "
         "gas leg alone: a plant that hedges its spread also sells its power forward at the same time, and "
         f"power and gas prices move together (correlation of monthly average peak power and TTF, "
         f"{_corr_start:%b %Y} to {_corr_end:%b %Y}: {_corr:.2f}), so the plant's real exposure is smaller.")

wait = common.garch_price_range(ttf, horizon=_WAIT_DAYS)
if wait is None:
    st.caption("TTF price range unavailable: the GARCH model could not be fitted on this date range.")
else:
    lo, hi = wait["lo"][-1], wait["hi"][-1]
    bill_now = gas_mwh * wait["price_now"]
    bill_lo, bill_hi = gas_mwh * _half(lo), gas_mwh * _half(hi)
    fit_start = ttf.dropna().index[0]

    w1, w2, w3 = st.columns(3)
    _w_today = pd.Timestamp(wait["date"]).date() == pd.Timestamp.now(tz="Europe/Berlin").date()
    w1.metric(f"TTF front month ({month_label}), {'latest price' if _w_today else 'close'} "
              f"{wait['date']:%d %b %Y}", f"{wait['price_now']:.2f} €/MWh",
              help=(f"Yahoo Finance `TTF=F`, matched to the monthly contract `{contract}` on this date."
                    if contract else "Yahoo Finance `TTF=F`; the monthly contract could not be identified.")
                   + (" During the trading day this is the latest price, not yet a settlement." if _w_today else ""))
    w2.metric(f"In {_WAIT_DAYS} business days, 90% range (€/MWh)", f"{_half(lo):.1f} to {_half(hi):.1f}",
              help="Rounded to 0.5 €/MWh: the range comes from 10,000 simulated paths and moves by a few tenths "
                   "between runs.")
    w3.metric(f"{_WAIT_DAYS}-day volatility", f"{wait['horizon_vol']:.1f}%",
              help="Standard deviation of the simulated 5-day log return.")

    b1, b2, b3 = st.columns(3)
    b1.metric("The month's gas, bought today", f"€{bill_now / 1e6:,.2f}m",
              help=f"{gas_mwh:,.0f} MWh of gas: {PLANT_MW} MW × {run_h:.1f} h a day (the flexible plant's "
                   f"average over the last 30 delivery business days) × {n_bdays} business days ÷ "
                   f"{EFFICIENCY:.0%} efficiency. A TTF month future delivers the same amount in every hour of "
                   f"the month (trading unit 1 MW), so this volume is about {contract_mw:,.0f} MW of the "
                   "contract; the plant's actual use varies by hour and is only on business days, so the "
                   "contract matches it only on average.")
    b2.metric("Bought in a week, 90% range", f"€{bill_lo / 1e6:,.1f}m to €{bill_hi / 1e6:,.1f}m")
    b3.metric("Extra cost if TTF ends at the 95th percentile", f"+€{(bill_hi - bill_now) / 1e6:,.1f}m",
              help="Top of the 90% range minus today's cost: a 1-in-20 threshold, not the average of the worst 5%.")

    fig_w, ax_w = plt.subplots(figsize=(14, 3.5))
    hist = ttf.tail(40)
    days_ahead = pd.bdate_range(wait["date"], periods=_WAIT_DAYS + 1)
    ax_w.plot(hist.index, hist.values, color="steelblue", linewidth=1.2, label="TTF front month")
    ax_w.fill_between(days_ahead, np.r_[wait["price_now"], wait["lo"]], np.r_[wait["price_now"], wait["hi"]],
                      color="steelblue", alpha=0.2, label="90% range if you wait")
    ax_w.plot(days_ahead, np.r_[wait["price_now"], wait["median"]], color="steelblue",
              linewidth=1, linestyle="--", label="Median (≈ today's price)")
    ax_w.set_ylabel("€/MWh")
    ax_w.set_title(f"TTF: last 40 business days and the range {_WAIT_DAYS} business days ahead")
    ax_w.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_w)

    st.caption(
        "**This is the risk of waiting, not a forecast of direction.** The model has no view on whether "
        "TTF goes up or down: its mean return is set to zero, so the range is centred on today's price. "
        "Method: GARCH(1,1) with Student-t shocks fitted on daily log returns of `TTF=F` from "
        f"{fit_start:%d %b %Y} to {wait['date']:%d %b %Y}, 10,000 simulated {_WAIT_DAYS}-day paths, "
        "5th to 95th percentile. Monthly roll jumps in `TTF=F` feed into the fitted volatility. The volume "
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
