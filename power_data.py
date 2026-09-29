"""Power and EUA data for the Power page.

- German/Luxembourg day-ahead prices from the Energy-Charts API (Fraunhofer ISE).
- EUA prices in €/t from EEX's yearly primary-auction reports.

Energy-Charts rate-limits (HTTP 429), so power requests are spaced out and retried
with backoff. Completed past years never change and are cached for 30 days; only
the current year is refetched every hour. EEX fetches are cached for an hour.
Loaders return what they could get plus a list of problems, so the page can warn
instead of failing.
"""
import datetime as dt
import io
import time

import pandas as pd
import requests
import streamlit as st

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
POWER_URL = "https://api.energy-charts.info/price"
EEX_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
           "emission-spot-primary-market-auction-report-{year}-data.xlsx")
FIRST_YEAR = 2020

_MIN_GAP = 1.5           # seconds between Energy-Charts requests
_RETRIES = 4             # retries after a 429, waiting 2, 4, 8, 16 s (or Retry-After)
_MAX_WAIT = 30
_last_request = 0.0


def _get_power(params):
    """GET the Energy-Charts price endpoint, spacing requests and backing off on 429."""
    global _last_request
    for attempt in range(_RETRIES + 1):
        gap = _MIN_GAP - (time.monotonic() - _last_request)
        if gap > 0:
            time.sleep(gap)
        r = requests.get(POWER_URL, params=params, headers=_HEADERS, timeout=60)
        _last_request = time.monotonic()
        if r.status_code != 429 or attempt == _RETRIES:
            break
        try:
            wait = float(r.headers.get("Retry-After", ""))
        except ValueError:
            wait = 2 ** (attempt + 1)
        time.sleep(min(max(wait, 1), _MAX_WAIT))
    if r.status_code == 429:
        raise RuntimeError(f"Energy-Charts rate limit (HTTP 429), still refused after {_RETRIES} retries")
    r.raise_for_status()
    return r.json()


def _fetch_power_year(year, bzn):
    """Daily baseload price (€/MWh) for one calendar year, as (Series, licence text).

    The request brackets the year by one day on each side, then keeps the Berlin
    calendar days of `year`, so no hour is lost to UTC/CET boundaries. The daily
    value is the plain mean of all delivery periods (hourly, or 15-minute since
    the day-ahead market moved to 15-minute products on 1 Oct 2025).
    """
    params = {"bzn": bzn, "start": f"{year - 1}-12-31", "end": f"{year + 1}-01-01"}
    js = _get_power(params)
    prices = js.get("price", js.get("data"))
    if not js.get("unix_seconds") or prices is None:
        raise ValueError(f"unexpected response fields: {sorted(js)}")
    s = pd.Series(pd.to_numeric(pd.Series(prices), errors="coerce").values,
                  index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin"))
    s = s[s.index.year == year]
    daily = s.groupby(s.index.date).mean().dropna()
    daily.index = pd.to_datetime(daily.index)
    return daily, js.get("license_info", "")


# Errors are not cached, so a year that failed is retried on the next run.
@st.cache_data(ttl=dt.timedelta(days=30), show_spinner=False)
def _fetch_past_year(year, bzn):
    return _fetch_power_year(year, bzn)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_current_year(year, bzn):
    return _fetch_power_year(year, bzn)


def fetch_power_year(year, bzn="DE-LU"):
    """One year of daily prices: long cache for completed years, hourly for the current one."""
    this_year = pd.Timestamp.now(tz="Europe/Berlin").year
    return (_fetch_past_year if year < this_year else _fetch_current_year)(year, bzn)


def load_power(start, end):
    """Daily DE-LU day-ahead baseload between start and end, fetched year by year.

    A year that still fails after retries is reported in the error list; the years
    that loaded are returned.
    """
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

    # Keep general EUA auctions (contract code T3PA). Aviation allowances (EUAA, code EAA3)
    # are auctioned separately and clear at different prices, so they are dropped.
    contract_col = next((c for c in df.columns if c.lower() == "contract"), None)
    if contract_col is not None:
        contracts = df[contract_col].astype(str).str.strip().str.upper()
        df = df[~contracts.str.contains("EAA", regex=False)]

    dates = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    prices = pd.to_numeric(df[price_col], errors="coerce")
    s = pd.Series(prices.values, index=dates.values).dropna()
    s = s[s.index.notna()]
    if s.empty:
        raise ValueError("no auction prices parsed")
    # Guard in case two general auctions ever clear on the same day: average them.
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
