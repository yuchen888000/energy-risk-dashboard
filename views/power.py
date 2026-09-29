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
    st.warning("Some years of power data could not be loaded, so the charts show only the years "
               "that did: " + "; ".join(power_errors))

ttf = common.load_close("TTF=F", start_date, end_date)
if ttf.empty:
    st.warning("TTF gas prices are unavailable right now, so the spread cannot be computed.")
    st.stop()

with st.spinner("Fetching EUA auction prices (EEX)..."):
    eua_auctions, eua_errors = power_data.load_eua_auctions(start_date, end_date)

if not eua_auctions.empty:
    eua = eua_auctions
    eua_row_source = None   # per day, set below: auction day or carried forward
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
        eua_row_source = f"Approximation: CARB.L rescaled to latest EUR {eua_now:.2f}/t"
        eua_source = ("EUA: **approximation** — CARB.L (USD ETC tracking EUA futures) rescaled so its "
                      f"latest close equals €{eua_now:.2f}/t. The ratio is held fixed over time, so "
                      "EUR/USD, roll yield and fees make earlier values drift from the true EUA price.")
    else:
        eua = pd.Series(eua_now, index=ttf.index)
        eua_row_source = f"Flat assumption: EUR {eua_now:.2f}/t"
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

# Where each day's EUA price comes from, for the CSV export.
if eua_row_source is None:
    last_auction = (pd.Series(eua.index, index=eua.index).reindex(eua_ext.index)
                    .ffill(limit=10).reindex(df.index))
    df["EUA source"] = [f"EEX auction {a:%Y-%m-%d}" if a == d
                        else f"EEX auction {a:%Y-%m-%d}, carried forward"
                        for d, a in zip(df.index, last_auction)]
else:
    df["EUA source"] = eua_row_source

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

export = pd.DataFrame({
    "Date": df.index.strftime("%Y-%m-%d"),
    "DE power price (EUR/MWh)": df["Power"].round(2).values,
    "TTF (EUR/MWh)": df["TTF"].round(3).values,
    "EUA price (EUR/tCO2)": df["EUA"].round(2).values,
    "EUA source": df["EUA source"].values,
    "Fuel cost (EUR/MWh)": df["Fuel cost"].round(3).values,
    "Carbon cost (EUR/MWh)": df["Carbon cost"].round(3).values,
    "Clean spark spread (EUR/MWh)": df["CSS"].round(3).values,
})
st.download_button("Download CSV", data=export.to_csv(index=False), file_name="clean_spark_spread_de.csv",
                   mime="text/csv",
                   help=f"Daily data behind the charts ({len(export):,} business days): DE-LU baseload power, TTF, "
                        f"EUA and its source, fuel cost (TTF / {EFFICIENCY}), carbon cost ({CARBON_PER_MWH:.3f} × EUA) "
                        "and the clean spark spread.")

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
# ─── Wait a week? Risk of delaying the gas purchase ───
_WAIT_DAYS = 5
gas_mwh_week = PLANT_MWH * _WAIT_DAYS / EFFICIENCY   # thermal MWh of gas for one week of running
st.subheader("Wait a Week?")
st.write(f"If the plant buys its gas {_WAIT_DAYS} business days from now instead of today, "
         "how much could one week of fuel cost change?")

wait = common.garch_price_range(ttf, horizon=_WAIT_DAYS)
if wait is None:
    st.caption("TTF price range unavailable — the GARCH model could not be fitted on this date range.")
else:
    lo, hi = wait["lo"][-1], wait["hi"][-1]
    bill_now = gas_mwh_week * wait["price_now"]
    bill_lo, bill_hi = gas_mwh_week * lo, gas_mwh_week * hi

    w1, w2, w3 = st.columns(3)
    w1.metric(f"TTF latest ({wait['date']:%d %b %Y})", f"{wait['price_now']:.2f} €/MWh")
    w2.metric(f"TTF in {_WAIT_DAYS} business days — 90% range (€/MWh)", f"{lo:.2f} – {hi:.2f}")
    w3.metric(f"{_WAIT_DAYS}-day volatility", f"{wait['horizon_vol']:.1f}%",
              help="Standard deviation of the simulated 5-day log return.")

    b1, b2, b3 = st.columns(3)
    b1.metric("One week of fuel, bought today", f"€{bill_now:,.0f}",
              help=f"{gas_mwh_week:,.0f} MWh of gas: {PLANT_MW} MW × {PLANT_HOURS} h × "
                   f"{_WAIT_DAYS} days ÷ {EFFICIENCY} efficiency.")
    b2.metric("Bought in a week — 90% range", f"€{bill_lo / 1e3:,.0f}k – €{bill_hi / 1e3:,.0f}k",
              help=f"€{bill_lo:,.0f} – €{bill_hi:,.0f}")
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
        "Method: GARCH(1,1) with Student-t shocks fitted on daily log returns of `TTF=F`, 10,000 "
        f"simulated {_WAIT_DAYS}-day paths, 5th–95th percentile. Monthly roll jumps in `TTF=F` feed "
        "into the fitted volatility, and a real purchase would be priced on a specific contract."
    )

st.caption(eua_source)
st.caption(f"Power: Energy-Charts API (Fraunhofer ISE), bidding zone DE-LU. {power_licence}")
