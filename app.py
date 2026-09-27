"""Entry point: shared sidebar settings and navigation between the three pages.

Streamlit Cloud runs this file. The commodity selector and date range are created
here, before pg.run(), so the selection carries across pages.
"""
import datetime as dt

import streamlit as st

import common

st.set_page_config(page_title="European Energy & Commodity Risk Dashboard", layout="wide")

carbon = common.resolve_carbon_benchmark()
commodities = common.build_commodities(carbon)

with st.sidebar:
    st.title("Settings")
    st.selectbox("Select Commodity", list(commodities.keys()), key="commodity",
                 help="Used by the Risk and Market pages. The Power page always shows German power, TTF and EUA.")
    st.date_input("Start Date", value=common.DEFAULT_START, key="start_date")
    st.date_input("End Date", value=dt.date.today(), key="end_date")
    st.caption(f"Carbon benchmark: **{carbon['label']}** — resolved at runtime from a cascade "
               "(`CARB.L` → `KRBN` → `ICLN`); KEUA, the original EUA proxy, was liquidated in March 2026.")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Yuchen Xia · IHEID MSc International Economics  \n"
                   "Python · Streamlit · yfinance · scikit-learn · arch (GARCH) · FinBERT · FinVADER · "
                   "Anthropic Claude API · Energy-Charts")

risk_page = st.Page("views/risk.py", title="Risk", icon="📉", url_path="risk")
power_page = st.Page("views/power.py", title="Power", icon="⚡", url_path="power")
market_page = st.Page("views/market.py", title="Market", icon="📰", url_path="market")


def _open_risk():
    st.switch_page(risk_page)


# Streamlit serves the default page at "/" and ignores its url_path, so the root is a
# hidden page that forwards to /risk. That keeps /risk, /power and /market all linkable.
pg = st.navigation([
    st.Page(_open_risk, title="Home", default=True, visibility="hidden"),
    risk_page, power_page, market_page,
])

pg.run()
