"""Power and EUA data for the Power page.

- German/Luxembourg day-ahead prices from the Energy-Charts API (Fraunhofer ISE).
- EUA prices in €/t from EEX's yearly primary-auction reports.

Each fetch is cached for an hour. Loaders return what they could get plus a list of
problems, so the page can warn instead of failing.
"""
import io

import pandas as pd
import requests
import streamlit as st

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
POWER_URL = "https://api.energy-charts.info/price"
EEX_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
           "emission-spot-primary-market-auction-report-{year}-data.xlsx")
FIRST_YEAR = 2020


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_power_year(year, bzn="DE-LU"):
    """Daily baseload price (€/MWh) for one calendar year, as (Series, licence text).

    The request brackets the year by one day on each side, then keeps the Berlin
    calendar days of `year`, so no hour is lost to UTC/CET boundaries. The daily
    value is the plain mean of all delivery periods (hourly, or 15-minute since
    the day-ahead market moved to 15-minute products on 1 Oct 2025).
    """
    params = {"bzn": bzn, "start": f"{year - 1}-12-31", "end": f"{year + 1}-01-01"}
    r = requests.get(POWER_URL, params=params, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()
    prices = js.get("price", js.get("data"))
    if not js.get("unix_seconds") or prices is None:
        raise ValueError(f"unexpected response fields: {sorted(js)}")
    s = pd.Series(pd.to_numeric(pd.Series(prices), errors="coerce").values,
                  index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin"))
    s = s[s.index.year == year]
    daily = s.groupby(s.index.date).mean().dropna()
    daily.index = pd.to_datetime(daily.index)
    return daily, js.get("license_info", "")


def load_power(start, end):
    """Daily DE-LU day-ahead baseload between start and end, fetched year by year."""
    parts, errors, licence = [], [], ""
    for year in range(max(start.year, FIRST_YEAR), end.year + 1):
        try:
            daily, licence = fetch_power_year(year)
            parts.append(daily)
        except Exception as e:
            errors.append(f"{year}: {e}")
    if not parts:
        return pd.Series(dtype=float), licence, errors
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))], licence, errors


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eex_year(year):
    """EUA primary-auction clearing prices (€/tCO2) for one year, one value per auction day."""
    r = requests.get(EEX_URL.format(year=year), headers=_HEADERS, timeout=60)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), header=None, engine="openpyxl")

    # The report opens with title rows; the header row has a "Date" cell and an "Auction Price" cell.
    header_row = None
    for i in range(min(len(raw), 30)):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if "date" in cells and any("auction price" in c for c in cells):
            header_row = i
            break
    if header_row is None:
        raise ValueError("header row with 'Date' and 'Auction Price' not found")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    date_col = next(c for c in df.columns if c.lower() == "date")
    price_col = next(c for c in df.columns if "auction price" in c.lower())

    # Keep general EUA auctions; aviation (EUAA) auctions clear at slightly different prices.
    contract_col = next((c for c in df.columns if c.lower() == "contract"), None)
    if contract_col is not None:
        contracts = df[contract_col].astype(str).str.strip().str.upper()
        if (contracts == "EUA").any():
            df = df[contracts == "EUA"]

    dates = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    prices = pd.to_numeric(df[price_col], errors="coerce")
    s = pd.Series(prices.values, index=dates.values).dropna()
    s = s[s.index.notna()]
    if s.empty:
        raise ValueError("no auction prices parsed")
    # Several auctions can clear on the same day (EU, Germany, Poland): average them.
    return s.groupby(level=0).mean()


def load_eua_auctions(start, end):
    """Daily EUA auction prices between start and end, fetched year by year."""
    parts, errors = [], []
    for year in range(max(start.year, FIRST_YEAR), end.year + 1):
        try:
            parts.append(fetch_eex_year(year))
        except Exception as e:
            errors.append(f"{year}: {e}")
    if not parts:
        return pd.Series(dtype=float), errors
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))], errors
