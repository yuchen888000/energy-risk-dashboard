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
PLANT_HOURS = 16
PLANT_MWH = PLANT_MW * PLANT_HOURS     # 6,400 MWh per day
_LOOKBACK = 250          # days behind the VaR estimate

ctx = common.context()
start_date, end_date = ctx.start_date, ctx.end_date

with st.sidebar.expander("Methodology — Power page"):
    st.markdown(f"""
    - **Power**: Germany/Luxembourg (DE-LU) day-ahead price from Energy-Charts, averaged
      over each Berlin calendar day (baseload).
    - **Gas**: TTF front-month future (`TTF=F`), €/MWh.
    - **Carbon**: EUA primary-auction clearing price from EEX, €/tCO2 (fallback below).
    - **Clean spark spread** = power − gas / {EFFICIENCY} − ({GAS_EF} / {EFFICIENCY}) × EUA.
    - **Risk**: day-on-day change in the spread, in €/MWh — not %, because power
      prices and the spread can be negative.
    """)

st.title("Power & Clean Spark Spread")
st.markdown("Does a gas-fired plant in Germany make money running today, and how much can that margin move?")

# ─── Data ───
with st.spinner("Fetching German day-ahead power prices (Energy-Charts)..."):
    power, power_licence, power_errors = power_data.load_power(start_date, end_date)
if power.empty:
    st.warning("German power prices are unavailable right now (Energy-Charts could not be reached). "
               "The rest of the dashboard is unaffected — try again later.")
    st.stop()
if power_errors:
    st.warning("Some years of power data could not be loaded: " + "; ".join(power_errors))

ttf = common.load_close("TTF=F", start_date, end_date)
if ttf.empty:
    st.warning("TTF gas prices are unavailable right now, so the spread cannot be computed.")
    st.stop()

with st.spinner("Fetching EUA auction prices (EEX)..."):
    eua_auctions, eua_errors = power_data.load_eua_auctions(start_date, end_date)

if not eua_auctions.empty:
    eua = eua_auctions
    eua_source = (f"EUA: EEX primary-auction clearing prices, €/tCO2 "
                  f"({eua.index.min():%d %b %Y} – {eua.index.max():%d %b %Y}); "
                  "carried forward on business days without an auction.")
else:
    # Fallback: rescale the carbon ETC so its latest price equals a user-entered EUA price.
    eua_now = st.sidebar.number_input("Current EUA price (€/t)", min_value=1.0, value=70.0, step=0.5,
                                      help="EEX auction data could not be loaded. The carbon ETC series "
                                           "is rescaled so that its latest value equals this price.")
    carb = common.load_close("CARB.L", start_date, end_date).dropna()
    if not carb.empty:
        eua = carb * (eua_now / carb.iloc[-1])
        eua_source = ("EUA: **approximation** — CARB.L (USD ETC tracking EUA futures) rescaled so its "
                      f"latest close equals €{eua_now:.2f}/t. The ratio is held fixed over time, so "
                      "EUR/USD, roll yield and fees make earlier values drift from the true EUA price.")
    else:
        eua = pd.Series(eua_now, index=ttf.index)
        eua_source = (f"EUA: **flat assumption** of €{eua_now:.2f}/t — neither EEX auction data nor "
                      "the carbon ETC could be loaded.")
    st.warning("EEX auction prices could not be loaded; EUA is approximated (see note under the charts).")

# Business days only: align everything on TTF trading days.
df = pd.DataFrame({"TTF": ttf})
df["Power"] = power.reindex(df.index)
eua_ext = eua.reindex(eua.index.union(df.index)).sort_index().ffill(limit=10)
df["EUA"] = eua_ext.reindex(df.index)
df = df.dropna()
if len(df) < _LOOKBACK // 5:
    st.warning("Not enough overlapping power, gas and carbon data for this date range — widen it.")
    st.stop()

df["Fuel cost"] = df["TTF"] / EFFICIENCY
df["Carbon cost"] = CARBON_PER_MWH * df["EUA"]
df["CSS"] = df["Power"] - df["Fuel cost"] - df["Carbon cost"]
df["dCSS"] = df["CSS"].diff()
df["dCSS vol"] = df["dCSS"].rolling(30).std()

# ─── Headline: current spread ───
latest = df.iloc[-1]
css_now = latest["CSS"]
if css_now >= 0:
    css_color, css_text = "green", "Running a gas plant is profitable"
else:
    css_color, css_text = "red", "Running a gas plant loses money"
st.markdown(f"<h2 style='color:{css_color}'>Clean spark spread: {css_now:+.2f} €/MWh</h2>"
            f"<p style='color:{css_color}; font-weight:bold; margin-top:-10px'>{css_text} "
            f"({df.index[-1]:%d %b %Y})</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Power (DE-LU baseload)", f"{latest['Power']:.2f} €/MWh")
k2.metric(f"Fuel cost (TTF / {EFFICIENCY})", f"{latest['Fuel cost']:.2f} €/MWh")
k3.metric(f"Carbon cost ({CARBON_PER_MWH:.3f} × EUA)", f"{latest['Carbon cost']:.2f} €/MWh")
k4.metric("Days with positive spread", f"{(df['CSS'] > 0).mean():.0%}")

# ─── Chart 1: spread over time ───
st.subheader("Clean Spark Spread")
fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(df.index, df["CSS"], color="black", linewidth=0.8)
ax1.fill_between(df.index, df["CSS"], 0, where=df["CSS"] >= 0, color="green", alpha=0.25, interpolate=True)
ax1.fill_between(df.index, df["CSS"], 0, where=df["CSS"] < 0, color="red", alpha=0.25, interpolate=True)
ax1.axhline(0, color="black", linewidth=1)
ax1.set_ylabel("€/MWh")
ax1.set_title("Clean spark spread, DE-LU baseload (green = plant earns, red = plant loses)")
plt.tight_layout()
st.pyplot(fig1)

# ─── Chart 2: decomposition ───
st.subheader("What Drives the Spread")
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(df.index, df["Power"], color="goldenrod", linewidth=1, label="Power price")
ax2.plot(df.index, df["Fuel cost"], color="steelblue", linewidth=1, label=f"Fuel cost (TTF / {EFFICIENCY})")
ax2.plot(df.index, df["Carbon cost"], color="seagreen", linewidth=1,
         label=f"Carbon cost ({CARBON_PER_MWH:.3f} × EUA)")
ax2.set_ylabel("€/MWh")
ax2.set_title("Spread = power price − fuel cost − carbon cost")
ax2.legend(fontsize=8, loc="upper left")
plt.tight_layout()
st.pyplot(fig2)

# ─── Risk: day-on-day changes in €/MWh ───
st.subheader("Spread Risk")
recent = df["dCSS"].dropna().tail(_LOOKBACK)
css_vol_now = df["dCSS vol"].dropna().iloc[-1]
css_var95 = -np.percentile(recent, 5)          # adverse daily move, €/MWh (positive number)
plant_var95 = css_var95 * PLANT_MWH

r1, r2, r3 = st.columns(3)
r1.metric("30-day volatility of daily change", f"{css_vol_now:.2f} €/MWh")
r2.metric("VaR 95% (1-day, spread)", f"{css_var95:.2f} €/MWh",
          help=f"5th percentile of day-on-day spread changes over the last {len(recent)} business days, "
               "shown as a positive loss.")
r3.metric(f"VaR 95% — {PLANT_MW} MW unit, {PLANT_HOURS} h/day", f"€{plant_var95:,.0f}",
          help=f"Spread VaR × {PLANT_MWH:,} MWh ({PLANT_MW} MW × {PLANT_HOURS} h).")

fig3, ax3 = plt.subplots(figsize=(14, 3.5))
ax3.plot(df.index, df["dCSS vol"], color="purple", linewidth=1)
ax3.fill_between(df.index, df["dCSS vol"], 0, color="purple", alpha=0.1)
ax3.set_ylabel("€/MWh")
ax3.set_title("30-day volatility of the daily change in the clean spark spread")
plt.tight_layout()
st.pyplot(fig3)

st.caption(
    f"For a {PLANT_MW} MW gas unit running {PLANT_HOURS} hours a day ({PLANT_MWH:,} MWh), a 1-in-20 "
    f"adverse day-on-day move in the spread cuts that day's gross margin by at least €{plant_var95:,.0f}. "
    "It assumes the unit runs whatever the spread, so it ignores the option to switch off when the "
    "spread is negative."
)
st.caption(
    "**Caveats.** Power is the day-ahead spot price, while TTF is the front-month future: the "
    "tenors don't match. Trading desks compute spreads from contracts with the same delivery period "
    f"(e.g. month-ahead power against month-ahead gas). Efficiency is assumed at {EFFICIENCY:.0%} "
    "(modern CCGTs reach 55–60%). `TTF=F` is a continuous front-month series from Yahoo Finance, so "
    "it jumps when the contract rolls each month; those jumps show up as day-on-day spread moves and "
    "inflate the volatility and VaR above. Only business days (TTF trading days) are used."
)
st.caption(eua_source)
st.caption(f"Power: Energy-Charts API (Fraunhofer ISE), bidding zone DE-LU. {power_licence}")
