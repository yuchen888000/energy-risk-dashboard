"""Power and EUA data for the Power page.

- German/Luxembourg day-ahead prices from the ENTSO-E Transparency Platform (document
  A44), kept at their original resolution (hourly, 15-minute since 1 Oct 2025) so the
  page can compute baseload, peak and hour-by-hour spreads. ENTSO-E publishes two price
  series for DE-LU: sequence 1 is the 12:00 Single Day-Ahead Coupling auction (the German
  day-ahead price), sequence 2 is EXAA's separate 10:15 auction. Only sequence 1 is used.
  If ENTSO-E cannot be reached, or no API token is configured, the same prices are taken
  from the Energy-Charts API (Fraunhofer ISE, data from Bundesnetzagentur / SMARD).
- EUA prices in €/t from EEX's yearly primary-auction reports.

Energy-Charts rate-limits (HTTP 429), so power requests are spaced out and retried
with backoff. Completed past years never change and are cached for 30 days; only
the current year is refetched every hour. EEX fetches are cached for an hour.
Loaders return what they could get plus a list of problems, so the page can warn
instead of failing.
"""
import datetime as dt
import io
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
POWER_URL = "https://api.energy-charts.info/price"
PUBLIC_POWER_URL = "https://api.energy-charts.info/public_power"
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_DE_LU = "10Y1001A1001A82H"      # EIC code of the DE-LU bidding zone
_ENTSOE_NS = {"n": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
_RESOLUTION = {"PT15M": "15min", "PT30M": "30min", "PT60M": "60min"}
EEX_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
           "emission-spot-primary-market-auction-report-{year}-data.xlsx")
FIRST_YEAR = 2020

_MIN_GAP = 1.5           # seconds between Energy-Charts requests
_RETRIES = 4             # retries after a 429, waiting 2, 4, 8, 16 s (or Retry-After)
_TIMEOUT_RETRIES = 2     # retries after a timeout or dropped connection
_MAX_WAIT = 30
_last_request = 0.0


def _get_power(params, url=POWER_URL):
    """GET an Energy-Charts endpoint, spacing requests and backing off on 429 and timeouts."""
    global _last_request
    for attempt in range(_RETRIES + 1):
        gap = _MIN_GAP - (time.monotonic() - _last_request)
        if gap > 0:
            time.sleep(gap)
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            # A slow or dropped response is retried up to twice (so a page load waits at most a
            # few minutes), then the year is reported as missing, as before.
            _last_request = time.monotonic()
            if attempt >= _TIMEOUT_RETRIES:
                raise
            time.sleep(min(2 ** (attempt + 1), _MAX_WAIT))
            continue
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


def _fetch_energy_charts_year(year, bzn):
    """Day-ahead prices (€/MWh) for one calendar year from Energy-Charts, as (Series, source).

    The Series is indexed by the start of each delivery period in Berlin time: hourly,
    or 15-minute since the day-ahead market moved to 15-minute products on 1 Oct 2025.
    The request brackets the year by one day on each side, then keeps the Berlin
    calendar days of `year`, so no period is lost to UTC/CET boundaries.
    """
    params = {"bzn": bzn, "start": f"{year - 1}-12-31", "end": f"{year + 1}-01-01"}
    js = _get_power(params)
    prices = js.get("price", js.get("data"))
    if not js.get("unix_seconds") or prices is None:
        raise ValueError(f"unexpected response fields: {sorted(js)}")
    s = pd.Series(pd.to_numeric(pd.Series(prices), errors="coerce").values,
                  index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin"))
    s = s[s.index.year == year].dropna()
    s = s[~s.index.duplicated()].sort_index()
    if s.empty:
        raise ValueError("no prices returned")
    licence = js.get("license_info", "")
    return s, f"Energy-Charts API (Fraunhofer ISE). {licence}".strip()


def _entsoe_token():
    """The ENTSO-E API token from Streamlit secrets or the environment, or None."""
    try:
        token = st.secrets.get("ENTSOE_TOKEN")
    except Exception:          # no secrets file at all
        token = None
    return token or os.environ.get("ENTSOE_TOKEN") or None


def _parse_entsoe_a44(xml_bytes):
    """Sequence-1 day-ahead prices from an ENTSO-E A44 document, indexed in UTC.

    Each TimeSeries carries a classification sequence: 1 is the SDAC auction, 2 is EXAA's
    10:15 auction (both published for DE-LU). A series without a sequence tag is taken as
    the only one. Curve type A03 omits a point when the price repeats the previous one, so
    missing positions are filled forward within each period.
    """
    root = ET.fromstring(xml_bytes)
    if not root.tag.endswith("Publication_MarketDocument"):
        reason = root.find(".//{*}Reason/{*}text")
        raise ValueError(reason.text if reason is not None else f"unexpected document {root.tag}")
    parts = []
    for ts in root.findall("n:TimeSeries", _ENTSOE_NS):
        seq = ts.find("n:classificationSequence_AttributeInstanceComponent.position", _ENTSOE_NS)
        if seq is not None and seq.text.strip() != "1":
            continue
        for period in ts.findall("n:Period", _ENTSOE_NS):
            res = period.find("n:resolution", _ENTSOE_NS).text.strip()
            if res not in _RESOLUTION:
                raise ValueError(f"unexpected resolution {res}")
            start = pd.Timestamp(period.find("n:timeInterval/n:start", _ENTSOE_NS).text)
            end = pd.Timestamp(period.find("n:timeInterval/n:end", _ENTSOE_NS).text)
            idx = pd.date_range(start, end, freq=_RESOLUTION[res], inclusive="left")
            values = pd.Series(float("nan"), index=range(1, len(idx) + 1))
            for pt in period.findall("n:Point", _ENTSOE_NS):
                values[int(pt.find("n:position", _ENTSOE_NS).text)] = float(
                    pt.find("n:price.amount", _ENTSOE_NS).text)
            parts.append(pd.Series(values.ffill().values, index=idx))
    if not parts:
        raise ValueError("no sequence-1 price series in the response")
    return pd.concat(parts).sort_index()


def _fetch_entsoe_year(year, token):
    """Day-ahead prices (€/MWh) for one Berlin calendar year from ENTSO-E, as (Series, source)."""
    # 23:00 UTC on 31 Dec is midnight in Berlin (CET, UTC+1), so this covers the Berlin year.
    params = {"securityToken": token, "documentType": "A44",
              "in_Domain": ENTSOE_DE_LU, "out_Domain": ENTSOE_DE_LU,
              "periodStart": f"{year - 1}12312300", "periodEnd": f"{year}12312300"}
    r = requests.get(ENTSOE_URL, params=params, headers=_HEADERS, timeout=120)
    if r.status_code == 401:
        raise RuntimeError("ENTSO-E rejected the API token (HTTP 401)")
    r.raise_for_status()
    s = _parse_entsoe_a44(r.content).tz_convert("Europe/Berlin")
    s = s[s.index.year == year].dropna()
    s = s[~s.index.duplicated()].sort_index()
    if s.empty:
        raise ValueError("no prices returned")
    return s, "ENTSO-E Transparency Platform, day-ahead prices (SDAC, sequence 1)"


def _fetch_power_year(year, bzn):
    """One year of DE-LU prices: ENTSO-E first, Energy-Charts if that fails.

    Returns (Series, source text, note). The note says why the fallback was used, or is empty.
    """
    token = _entsoe_token()
    note = ""
    if token and bzn == "DE-LU":
        try:
            s, source = _fetch_entsoe_year(year, token)
            return s, source, note
        except Exception as e:
            note = f"ENTSO-E unavailable ({e}); Energy-Charts used instead"
    else:
        note = "no ENTSO-E token configured; Energy-Charts used instead"
    s, source = _fetch_energy_charts_year(year, bzn)
    return s, source, note


# Errors are not cached, so a year that failed is retried on the next run.
# `version` is part of the cache key: bump it when the returned data changes shape, so
# an old cached result is never reused.
_CACHE_VERSION = 3


@st.cache_data(ttl=dt.timedelta(days=30), show_spinner=False)
def _fetch_past_year(year, bzn, version):
    return _fetch_power_year(year, bzn)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_current_year(year, bzn, version):
    return _fetch_power_year(year, bzn)


def fetch_power_year(year, bzn="DE-LU"):
    """One year of day-ahead prices: long cache for completed years, hourly for the current one."""
    this_year = pd.Timestamp.now(tz="Europe/Berlin").year
    return (_fetch_past_year if year < this_year else _fetch_current_year)(year, bzn, _CACHE_VERSION)


def load_power(start, end):
    """DE-LU day-ahead prices per delivery period between start and end (Berlin dates).

    Returns (prices, sources, errors). `sources` maps each loaded year to its source text
    (and the reason if Energy-Charts replaced ENTSO-E). A year that still fails after
    retries is reported in the error list; the years that loaded are returned.
    """
    parts, errors, sources = [], [], {}
    for year in range(max(start.year, FIRST_YEAR), end.year + 1):
        try:
            prices, source, note = fetch_power_year(year)
            parts.append(prices)
            sources[year] = f"{source} ({note})" if note else source
        except Exception as e:
            errors.append(f"{year}: {e}")
    if not parts:
        return pd.Series(dtype=float), sources, errors
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    days = s.index.tz_localize(None).normalize()   # Berlin calendar day
    return s[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))], sources, errors


PEAK_START, PEAK_END = 8, 20   # peak block: Monday to Friday, 08:00 to 20:00 Berlin time


def period_hours(prices):
    """Length of each delivery period in hours (1 before 1 Oct 2025, 0.25 after)."""
    idx = prices.index.to_series()
    step = (idx.shift(-1) - idx).dt.total_seconds() / 3600
    # The last period of each gap (end of data, missing hours) takes the previous length.
    step = step.where((step > 0) & (step <= 1)).ffill().bfill()
    return step.fillna(1.0)


def is_peak(index):
    """True for delivery periods in the peak block (weekdays, 08:00 to 20:00)."""
    return (index.dayofweek < 5) & (index.hour >= PEAK_START) & (index.hour < PEAK_END)


def daily_base_peak(prices):
    """Per Berlin calendar day: baseload (all periods) and peak (weekday 08:00 to 20:00) averages.

    Both are time-weighted, so hourly and 15-minute days are treated alike. Peak is
    missing on weekends.
    """
    h = period_hours(prices)
    day = prices.index.tz_localize(None).normalize()   # Berlin calendar day
    w = prices * h
    base = w.groupby(day).sum() / h.groupby(day).sum()
    pk = is_peak(prices.index)
    peak = w[pk].groupby(day[pk]).sum() / h[pk].groupby(day[pk]).sum()
    out = pd.DataFrame({"Base": base, "Peak": peak})
    out.index = pd.to_datetime(out.index)
    return out


def flexible_run(prices, cost_per_day):
    """A plant that runs only in the periods where the power price is above its cost.

    `cost_per_day` is the fuel plus carbon cost in €/MWh of power for each day (index:
    dates). Returns, per day, the hours it would run, its average clean spark spread
    over those hours (€/MWh), and how many separate times it would start up that day.
    Days with no profitable period get 0 hours and no spread.
    """
    day = prices.index.tz_localize(None).normalize()   # Berlin calendar day
    cost = pd.Series(cost_per_day.reindex(day).values, index=prices.index)
    keep = cost.notna()
    prices, cost, day = prices[keep], cost[keep], day[keep.values]
    h = period_hours(prices)
    spread = prices - cost
    run = spread > 0
    hours = h.where(run, 0.0).groupby(day).sum()
    margin = (spread * h).where(run, 0.0).groupby(day).sum()   # €/MW for the day
    # A start is a period where the plant runs and was off in the period before (same day).
    was_on = run.groupby(day).shift(1, fill_value=False).astype(bool)
    starts = (run & ~was_on).groupby(day).sum()
    out = pd.DataFrame({"Run hours": hours, "Run spread": margin / hours.where(hours > 0),
                        "Starts": starts})
    out.index = pd.to_datetime(out.index)
    return out


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
    # are auctioned separately and clear at different prices, so they are dropped, whichever
    # of the two names the file uses.
    contract_col = next((c for c in df.columns if c.lower() == "contract"), None)
    if contract_col is not None:
        contracts = df[contract_col].astype(str).str.strip().str.upper()
        df = df[~(contracts.str.contains("EAA", regex=False) | contracts.str.contains("EUAA", regex=False))]

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


# ─── Actual gas-fired generation (Energy-Charts public_power, from SMARD) ───
def _fetch_gas_generation_year(year):
    """German fossil-gas generation (MW, net public generation) per period for one Berlin year."""
    params = {"country": "de", "start": f"{year - 1}-12-31", "end": f"{year + 1}-01-01"}
    js = _get_power(params, url=PUBLIC_POWER_URL)
    gas = next((t for t in js.get("production_types", []) if t.get("name") == "Fossil gas"), None)
    if not js.get("unix_seconds") or gas is None:
        raise ValueError("no 'Fossil gas' series in the response")
    s = pd.Series(pd.to_numeric(pd.Series(gas["data"]), errors="coerce").values,
                  index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin"))
    s = s[s.index.year == year].dropna()
    s = s[~s.index.duplicated()].sort_index()
    if s.empty:
        raise ValueError("no generation data returned")
    return s


@st.cache_data(ttl=dt.timedelta(days=30), show_spinner=False)
def _gas_gen_past_year(year, version):
    return _fetch_gas_generation_year(year)


@st.cache_data(ttl=3600, show_spinner=False)
def _gas_gen_current_year(year, version):
    return _fetch_gas_generation_year(year)


def load_gas_generation(start, end):
    """German gas-fired generation per period between start and end (Berlin dates), with errors."""
    this_year = pd.Timestamp.now(tz="Europe/Berlin").year
    parts, errors = [], []
    for year in range(max(start.year, FIRST_YEAR), end.year + 1):
        try:
            f = _gas_gen_past_year if year < this_year else _gas_gen_current_year
            parts.append(f(year, _CACHE_VERSION))
        except Exception as e:
            errors.append(f"{year}: {e}")
    if not parts:
        return pd.Series(dtype=float), errors
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    days = s.index.tz_localize(None).normalize()
    return s[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))], errors


def gas_weighted_price(prices, gas_mw):
    """Per Berlin day, the day-ahead price weighted by German gas-fired output.

    This is the average price at which gas plants as a whole sold their output that day.
    Generation is reported per 15 minutes; before 1 Oct 2025 the price is hourly, so each
    hourly price applies to its four quarter-hours.
    """
    if prices.empty or gas_mw.empty:
        return pd.Series(dtype=float)
    price_at_gen = prices.sort_index().reindex(gas_mw.index, method="ffill", limit=3)
    ok = price_at_gen.notna() & gas_mw.notna() & (gas_mw > 0)
    day = gas_mw.index.tz_localize(None).normalize()
    num = (price_at_gen * gas_mw).where(ok, 0.0).groupby(day).sum()
    den = gas_mw.where(ok, 0.0).groupby(day).sum()
    # Only complete days: today's generation is published with a lag, and a part day would
    # miss the evening peak. A Berlin day has 92, 96 or 100 quarter-hours (clock changes).
    days = pd.DatetimeIndex(den.index)
    expected = pd.Series([(pd.Timestamp(d + pd.Timedelta(days=1), tz="Europe/Berlin")
                           - pd.Timestamp(d, tz="Europe/Berlin")) / pd.Timedelta(minutes=15) for d in days],
                         index=den.index)
    complete = ok.groupby(day).sum() >= 0.98 * expected
    out = (num / den.where(den > 0))[complete]
    out.index = pd.to_datetime(out.index)
    return out.dropna()
