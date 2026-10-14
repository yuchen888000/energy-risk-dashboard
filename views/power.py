"""Power page: German day-ahead power and the clean spark spread of a gas plant.

Designed with power trading desks and utilities in mind.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common
import power_data

EFFICIENCY = 0.5         # gas plant efficiency (electric MWh per thermal MWh of gas)
GAS_EF = 0.202           # natural gas emission factor, tCO2 per MWh thermal
CARBON_PER_MWH = GAS_EF / EFFICIENCY   # 0.404 tCO2 per MWh of power
PLANT_MW = 400
PLANT_HOURS = power_data.PEAK_END - power_data.PEAK_START   # runs the 12 peak hours
PLANT_MWH = PLANT_MW * PLANT_HOURS     # 4,800 MWh per weekday
_LOOKBACK = 250          # days behind the VaR estimate
_PEAK = f"weekdays {power_data.PEAK_START:02d}:00 to {power_data.PEAK_END:02d}:00"

ctx = common.context()
start_date, end_date = ctx.start_date, ctx.end_date

with st.sidebar.expander("Methodology: Power page"):
    st.markdown(f"""
    - **Power**: Germany/Luxembourg (DE-LU) day-ahead prices from Energy-Charts, per
      hour (per 15 minutes since 1 Oct 2025). **Peak** is the average over {_PEAK}
      (Berlin time); **baseload** is the average over all 24 hours.
    - **Gas**: TTF front-month future (`TTF=F`), €/MWh.
    - **Carbon**: EUA primary-auction clearing price from EEX, €/tCO2 (fallback: see the
      note under the charts).
    - **Clean spark spread** = power − gas / {EFFICIENCY} − ({GAS_EF} / {EFFICIENCY}) × EUA:
      the gross margin per MWh before start-up, other variable and fixed costs.
    - **Flexible plant**: on business days, runs in any period of the day (all 24 hours, not
      only the peak block) where the day-ahead price is above its fuel and carbon cost.
      Day-ahead prices are known the day before delivery; start-up costs, minimum run times
      and part-load operation are ignored.
    - **Risk**: day-on-day change in the peak spread, in €/MWh, not %, because power
      prices and the spread can be negative, and in the flexible plant's daily margin (€).
      Changes across data gaps of more than a week are left out.
    """)

st.title("Power & Clean Spark Spread")
st.markdown("Does a gas-fired plant in Germany cover its fuel and carbon costs over the peak block, "
            "and in the hours it would choose to run? How much can that margin move?")

# ─── Data ───
with st.spinner("Fetching German day-ahead power prices (Energy-Charts)..."):
    prices, power_licence, power_errors = power_data.load_power(start_date, end_date)
if prices.empty:
    st.warning("German power prices are unavailable right now (Energy-Charts could not be reached). "
               "The rest of the dashboard is unaffected; try again later.")
    st.stop()
if power_errors:
    st.warning("Some years of power data could not be loaded, so the charts show only the years "
               "that did: " + "; ".join(power_errors))

ttf = common.load_close("TTF=F", start_date, end_date)
if ttf.empty:
    st.warning("TTF gas prices are unavailable right now, so the spread cannot be computed.")
    st.stop()

with st.spinner("Fetching EUA auction prices (EEX)..."):
    eua_auctions, eua_errors = power_data.load_eua_auctions(start_date, end_date)

if eua_errors and not eua_auctions.empty:
    st.warning("Some years of EEX auction prices could not be loaded: " + "; ".join(eua_errors))
if not eua_auctions.empty:
    eua = eua_auctions
    eua_row_source = None   # per day, set below: auction day or carried forward
    eua_source = (f"EUA: EEX primary-auction clearing prices, €/tCO2 "
                  f"({eua.index.min():%d %b %Y} to {eua.index.max():%d %b %Y}); "
                  "carried forward for up to 10 business days without an auction. The auction is for spot "
                  "allowances, so it can differ from the December future that the market quotes.")
else:
    # Fallback: rescale the carbon ETC so its latest price equals a user-entered EUA price.
    # No default value: a stale built-in price could flip the sign of the spread.
    eua_now = st.sidebar.number_input("Current EUA price (€/t)", min_value=1.0, value=None, step=0.5,
                                      placeholder="Enter today's EUA price",
                                      help="EEX auction data could not be loaded. The carbon ETC series "
                                           "is rescaled so that its latest value equals this price.")
    if eua_now is None:
        st.warning("EEX auction prices could not be loaded. Enter the current EUA price in the sidebar "
                   "to compute the spread; no built-in price is used.")
        st.stop()
    carb = common.load_close("CARB.L", start_date, end_date).dropna()
    if not carb.empty:
        eua = carb * (eua_now / carb.iloc[-1])
        eua_row_source = f"Approximation: CARB.L rescaled to latest EUR {eua_now:.2f}/t"
        eua_source = ("EUA: **approximation**, CARB.L (USD ETC tracking EUA futures) rescaled so its "
                      f"latest close equals €{eua_now:.2f}/t. The ratio is held fixed over time, so "
                      "EUR/USD, roll yield and fees make earlier values drift from the true EUA price.")
    else:
        eua = pd.Series(eua_now, index=ttf.index)
        eua_row_source = f"Flat assumption: EUR {eua_now:.2f}/t"
        eua_source = (f"EUA: **flat assumption** of €{eua_now:.2f}/t, as neither EEX auction data nor "
                      "the carbon ETC could be loaded.")
    st.warning("EEX auction prices could not be loaded; EUA is approximated (see note under the charts).")

# Business days only: align everything on TTF trading days.
daily = power_data.daily_base_peak(prices)
df = pd.DataFrame({"TTF": ttf})
df["Power peak"] = daily["Peak"].reindex(df.index)
df["Power base"] = daily["Base"].reindex(df.index)
eua_ext = eua.reindex(eua.index.union(df.index)).sort_index().ffill(limit=10)
df["EUA"] = eua_ext.reindex(df.index)
df = df.dropna()
if len(df) < _LOOKBACK // 5:
    st.warning("Not enough overlapping power, gas and carbon data for this date range; widen it.")
    st.stop()

df["Fuel cost"] = df["TTF"] / EFFICIENCY
df["Carbon cost"] = CARBON_PER_MWH * df["EUA"]
cost = df["Fuel cost"] + df["Carbon cost"]
df["CSS peak"] = df["Power peak"] - cost
df["CSS base"] = df["Power base"] - cost
flex = power_data.flexible_run(prices, cost)
df["Run hours"] = flex["Run hours"].reindex(df.index)
df["Run spread"] = flex["Run spread"].reindex(df.index)
# Day-on-day changes; a change across a data gap of more than a week (no EUA price within 10
# business days, e.g. early January) is not a one-day move, so it is left out.
_gap = df.index.to_series().diff() > pd.Timedelta(days=7)
df["dCSS"] = df["CSS peak"].diff().mask(_gap)
df["dCSS vol"] = df["dCSS"].rolling(30, min_periods=20).std()
# Daily gross margin of the flexible plant, € (0 on days it stays off), and its daily change.
df["Flex margin"] = (df["Run hours"] * df["Run spread"]).fillna(0.0) * PLANT_MW
df["dFlex"] = df["Flex margin"].diff().mask(_gap)

# Where each day's EUA price comes from, for the headline and the CSV export.
if eua_row_source is None:
    last_auction = (pd.Series(eua.index, index=eua.index).reindex(eua_ext.index)
                    .ffill(limit=10).reindex(df.index))
    df["EUA source"] = [f"EEX auction {a:%Y-%m-%d}" if a == d
                        else f"EEX auction {a:%Y-%m-%d}, carried forward"
                        for d, a in zip(df.index, last_auction)]
else:
    df["EUA source"] = eua_row_source

# ─── Headline: latest peak spread, with baseload and the flexible plant alongside ───
latest = df.iloc[-1]
day = df.index[-1]
peak_now = latest["CSS peak"]
if latest["Run hours"] > 0:
    flex_value = f"{latest['Run hours']:.2f} h at {latest['Run spread']:+.2f} €/MWh"
    flex_help = ("Hours of the day (any of the 24) where the day-ahead price covered the fuel and carbon "
                 "cost, and the average spread in those hours.")
else:
    flex_value = "Stays off"
    flex_help = "No period of the day covered the fuel and carbon cost."
st.markdown(f"**Clean spark spread, {day:%a %d %b %Y}** (gas plant at {EFFICIENCY:.0%} efficiency)")
h1, h2, h3 = st.columns(3)
h1.metric(f"Peak block ({power_data.PEAK_START:02d}:00 to {power_data.PEAK_END:02d}:00)",
          f"{peak_now:+.2f} €/MWh",
          help=f"Average spread over the {power_data.PEAK_START:02d}:00 to {power_data.PEAK_END:02d}:00 "
               "delivery periods of this day: what a plant "
               "running the whole peak block earns per MWh.")
h2.metric("Baseload (all 24 h)", f"{latest['CSS base']:+.2f} €/MWh")
h3.metric("Flexible plant", flex_value, help=flex_help)
st.caption(f"Gross margin per MWh at {EFFICIENCY:.0%} efficiency, before start-up, other variable and "
           f"fixed costs. EUA price: {latest['EUA source']}.")

last_year = df[df.index > day - pd.DateOffset(years=1)]
share_label = (f"Days with positive peak spread at {EFFICIENCY:.0%} efficiency, last 12 months"
               if df.index[0] <= day - pd.DateOffset(years=1)
               else f"Days with positive peak spread at {EFFICIENCY:.0%} efficiency, selected range")


def _share_at(eff, frame):
    """Share of days with a positive peak spread if the plant had efficiency `eff`."""
    spread = frame["Power peak"] - frame["TTF"] / eff - GAS_EF / eff * frame["EUA"]
    return (spread > 0).mean()


_sens = ", ".join(f"{_share_at(e, last_year):.0%} of days at {e:.0%}" for e in (0.45, 0.55, 0.60))
k1, k2, k3, k4 = st.columns(4)
k1.metric("Power (DE-LU peak)", f"{latest['Power peak']:.2f} €/MWh",
          help=f"Average over {_PEAK}. Baseload (24 h): {latest['Power base']:.2f} €/MWh.")
k2.metric(f"Fuel cost (TTF / {EFFICIENCY})", f"{latest['Fuel cost']:.2f} €/MWh")
k3.metric(f"Carbon cost ({CARBON_PER_MWH:.3f} × EUA)", f"{latest['Carbon cost']:.2f} €/MWh")
k4.metric(share_label, f"{(last_year['CSS peak'] > 0).mean():.0%}",
          help=f"Share of business days with a positive peak spread, {last_year.index[0]:%d %b %Y} to "
               f"{day:%d %b %Y}. Over the whole selected range ({df.index[0]:%d %b %Y} to {day:%d %b %Y}): "
               f"{(df['CSS peak'] > 0).mean():.0%}. The share depends strongly on the assumed efficiency; "
               f"same period at other efficiencies: {_sens}.")

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
ax1.set_title("Clean spark spread, DE-LU: peak (green above zero, red below) and baseload")
ax1.legend(fontsize=8, loc="upper left")
plt.tight_layout()
st.pyplot(fig1)

# ─── Chart 2: decomposition ───
st.subheader("What Drives the Spread")
fig2, ax2 = plt.subplots(figsize=(14, 4))
for _col, _kw in [("Power peak", dict(color="goldenrod", linewidth=1, label="Power price, peak")),
                  ("Power base", dict(color="goldenrod", linewidth=0.6, alpha=0.5, label="Power price, baseload")),
                  ("Fuel cost", dict(color="steelblue", linewidth=1, label=f"Fuel cost (TTF / {EFFICIENCY})")),
                  ("Carbon cost", dict(color="seagreen", linewidth=1,
                                       label=f"Carbon cost ({CARBON_PER_MWH:.3f} × EUA)"))]:
    _s = common.break_gaps(df[_col])
    ax2.plot(_s.index, _s.values, **_kw)
ax2.set_ylabel("€/MWh")
ax2.set_title("Spread = power price − fuel cost − carbon cost")
ax2.legend(fontsize=8, loc="upper left")
plt.tight_layout()
st.pyplot(fig2)

export = pd.DataFrame({
    "Date": df.index.strftime("%Y-%m-%d"),
    "DE power price, peak (EUR/MWh)": df["Power peak"].round(2).values,
    "DE power price, baseload (EUR/MWh)": df["Power base"].round(2).values,
    "TTF (EUR/MWh)": df["TTF"].round(3).values,
    "EUA price (EUR/tCO2)": df["EUA"].round(2).values,
    "EUA source": df["EUA source"].values,
    "Fuel cost (EUR/MWh)": df["Fuel cost"].round(3).values,
    "Carbon cost (EUR/MWh)": df["Carbon cost"].round(3).values,
    "Clean spark spread, peak (EUR/MWh)": df["CSS peak"].round(3).values,
    "Clean spark spread, baseload (EUR/MWh)": df["CSS base"].round(3).values,
    "Flexible plant, hours run": df["Run hours"].round(2).values,
    "Flexible plant, average spread when running (EUR/MWh)": df["Run spread"].round(3).values,
})
st.download_button("Download CSV", data=export.to_csv(index=False), file_name="clean_spark_spread_de.csv",
                   mime="text/csv",
                   help=f"Daily data behind the charts ({len(export):,} business days): DE-LU peak and "
                        f"baseload power, TTF, EUA and its source, fuel cost (TTF / {EFFICIENCY}), carbon cost "
                        f"({CARBON_PER_MWH:.3f} × EUA), peak and baseload clean spark spreads, and the hours "
                        "and average spread of a plant that runs only when the spread is positive.")

# ─── Risk: day-on-day changes in the peak spread, €/MWh ───
st.subheader("Spread Risk")
recent = df["dCSS"].dropna().tail(_LOOKBACK)
css_vol_now = df["dCSS vol"].dropna().iloc[-1]
css_var95 = -np.percentile(recent, 5)          # adverse daily move, €/MWh (positive number)
plant_var95 = css_var95 * PLANT_MWH
flex_recent = df["dFlex"].dropna().tail(_LOOKBACK)
flex_var95 = max(0.0, -np.percentile(flex_recent, 5))

r1, r2, r3, r4 = st.columns(4)
r1.metric("30-day volatility of the daily change in peak spread", f"{css_vol_now:.2f} €/MWh")
r2.metric("VaR 95%, 1-day change in peak spread", f"{css_var95:.2f} €/MWh",
          help=f"5th percentile of day-on-day changes in the peak spread over the last {len(recent)} "
               "available business days, shown as a positive number. Changes across data gaps of more "
               "than a week are left out.")
r3.metric(f"VaR 95%: {PLANT_MW} MW unit running all {PLANT_HOURS} peak hours", f"€{plant_var95:,.0f}",
          help=f"Spread VaR × {PLANT_MWH:,} MWh ({PLANT_MW} MW × {PLANT_HOURS} h, {_PEAK}).")
r4.metric(f"VaR 95%: {PLANT_MW} MW flexible plant", f"€{flex_var95:,.0f}",
          help=f"5th percentile of day-on-day changes in the flexible plant's daily gross margin "
               f"({PLANT_MW} MW × hours run × average spread when running) over the last "
               f"{len(flex_recent)} available business days.")

fig3, ax3 = plt.subplots(figsize=(14, 3.5))
_vol = common.break_gaps(df["dCSS vol"])
ax3.plot(_vol.index, _vol.values, color="purple", linewidth=1)
ax3.fill_between(_vol.index, _vol.values, 0, color="purple", alpha=0.1)
ax3.set_ylabel("€/MWh")
ax3.set_title("30-day volatility of the daily change in the peak clean spark spread")
plt.tight_layout()
st.pyplot(fig3)

st.caption(
    f"For a {PLANT_MW} MW gas unit running the {PLANT_HOURS} peak hours ({PLANT_MWH:,} MWh a day), a "
    f"1-in-20 adverse day-on-day move in the peak spread cuts that day's gross margin by at least "
    f"€{plant_var95:,.0f}. That unit runs all peak hours whatever the spread. The flexible plant "
    "switches off in periods with a negative spread, so its daily margin cannot fall below zero; its "
    f"1-in-20 daily fall is €{flex_var95:,.0f}. Both measure how much the margin moves from one day to "
    "the next, not the VaR of a hedged position."
)
st.caption(
    "**Caveats.** Power is the day-ahead price, while TTF is the front-month future, so the tenors "
    "don't match; S&P Platts computes its German spreads from its own TTF assessments, including "
    f"day-ahead. Efficiency is assumed at {EFFICIENCY:.0%}; Platts publishes German spreads at 45%, 50% "
    "and 60%. Fuel cost excludes gas transport and other variable costs. The peak block includes "
    "public holidays, as exchange peak products do. `TTF=F` is a continuous front-month series from "
    "Yahoo Finance: it jumps when the contract rolls each month, and it follows the NYMEX (US) trading "
    "calendar, so US holidays are missing. Days are aligned on its trading days, and an EUA price is "
    "carried forward for at most 10 business days, so the series has gaps (for example in early "
    "January); charts break at gaps longer than a week."
)
# ─── Wait a week? Risk of delaying the gas purchase ───
_WAIT_DAYS = 5
gas_mwh_week = PLANT_MWH * _WAIT_DAYS / EFFICIENCY   # thermal MWh of gas for one week of running
st.subheader("Wait a Week?")
st.write(f"If the plant buys its gas {_WAIT_DAYS} business days from now instead of today, "
         "how much could one week of fuel cost change?")

wait = common.garch_price_range(ttf, horizon=_WAIT_DAYS)
if wait is None:
    st.caption("TTF price range unavailable: the GARCH model could not be fitted on this date range.")
else:
    lo, hi = wait["lo"][-1], wait["hi"][-1]
    bill_now = gas_mwh_week * wait["price_now"]
    bill_lo, bill_hi = gas_mwh_week * lo, gas_mwh_week * hi
    fit_start = ttf.dropna().index[0]

    w1, w2, w3 = st.columns(3)
    w1.metric(f"TTF latest ({wait['date']:%d %b %Y})", f"{wait['price_now']:.2f} €/MWh")
    w2.metric(f"TTF in {_WAIT_DAYS} business days, 90% range (€/MWh)", f"{lo:.2f} to {hi:.2f}")
    w3.metric(f"{_WAIT_DAYS}-day volatility", f"{wait['horizon_vol']:.1f}%",
              help="Standard deviation of the simulated 5-day log return.")

    b1, b2, b3 = st.columns(3)
    b1.metric("One week of fuel, bought today", f"€{bill_now:,.0f}",
              help=f"{gas_mwh_week:,.0f} MWh of gas: {PLANT_MW} MW × {PLANT_HOURS} peak hours × "
                   f"{_WAIT_DAYS} days ÷ {EFFICIENCY} efficiency.")
    b2.metric("Bought in a week, 90% range", f"€{bill_lo / 1e3:,.0f}k to €{bill_hi / 1e3:,.0f}k",
              help=f"€{bill_lo:,.0f} to €{bill_hi:,.0f}")
    b3.metric("Cost of waiting, worst 5%", f"+€{bill_hi - bill_now:,.0f}",
              help="How much more the week of gas costs if TTF ends at the top of the range.")

    fig_w, ax_w = plt.subplots(figsize=(14, 3.5))
    hist = ttf.tail(40)
    days_ahead = pd.bdate_range(wait["date"], periods=_WAIT_DAYS + 1)
    ax_w.plot(hist.index, hist.values, color="steelblue", linewidth=1.2, label="TTF (front month)")
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
        "**This is the risk of waiting, not a forecast of direction.** The model has no view on "
        "whether TTF goes up or down: its mean return is set to zero, so the range is centred on "
        "today's price (slightly wider on the upside, because prices move in percentage terms). "
        "Method: GARCH(1,1) with Student-t shocks fitted on daily log returns of `TTF=F` from "
        f"{fit_start:%d %b %Y} to {wait['date']:%d %b %Y} (the first and last TTF price used), 10,000 "
        f"simulated {_WAIT_DAYS}-day paths, 5th to 95th percentile. Monthly roll jumps in `TTF=F` feed "
        "into the fitted volatility, and a real purchase would be priced on a specific contract."
    )

st.caption(eua_source)
st.caption(f"Power: Energy-Charts API (Fraunhofer ISE), bidding zone DE-LU. {power_licence}")
