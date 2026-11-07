"""Update data/ttf_saved.csv, the saved copy of TTF prices the app falls back on when Yahoo Finance returns too
little of TTF=F to the hosted app. Run by .github/workflows/refresh-ttf.yml.

A short or failed download never replaces history and never fails the job: see merge. If yfinance is short, TTF=F is
read from Yahoo's chart API instead. When the file is still behind after all sources, the run ends with a warning in
the Actions log, not a failure. The Dashboard patrol reports a saved copy that falls behind.
"""
import json
import sys
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

PATH = "data/ttf_saved.csv"
TICKER = "TTF=F"
FULL_ROWS = 250   # fewer rows than this means Yahoo cut the answer short
MONTH_CODES = "FGHJKMNQUVXZ"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=3mo&interval=1d"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def contracts(today):
    """Monthly TTF contracts from two months back to the app's forward curve's last month (15 months ahead)."""
    months = pd.period_range(today.to_period("M") - 2, today.to_period("M") + 15, freq="M")
    return [f"TTF{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM" for p in months]


def close_table(data):
    """Close prices from a yf.download result as a table with one column per ticker, whatever its shape: empty, one
    row, one ticker or several. Never squeezes, so one row stays a table (7 Nov 2026: squeeze turned a one-row
    answer into a number and the job failed)."""
    if data is None or data.empty or "Close" not in data.columns.get_level_values(0):
        return pd.DataFrame(columns=[TICKER], index=pd.DatetimeIndex([]))
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(TICKER)
    if TICKER not in close.columns:
        close = close.assign(**{TICKER: float("nan")})
    return close


def chart_closes(read=None):
    """TTF=F daily closes for the last 3 months from Yahoo's chart API, a second source when yfinance is short. Dates
    are trading days in the exchange's time zone (New York), as yfinance gives them."""
    if read is None:
        def read(url):
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
    result = read(CHART_URL.format(TICKER))["chart"]["result"][0]
    tz = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    days = [pd.Timestamp.fromtimestamp(t, tz).tz_localize(None).normalize() for t in result["timestamp"]]
    closes = pd.Series(result["indicators"]["quote"][0]["close"], index=pd.DatetimeIndex(days), dtype=float)
    return closes[~closes.index.duplicated(keep="last")].dropna()


def _attempt(name, fetch):
    """Run one download; a network or Yahoo error is printed and treated as no data, so it cannot fail the job."""
    try:
        return fetch()
    except Exception as exc:  # Yahoo answers vary; any failure here just means this source gave nothing
        print(f"{name}: no data ({type(exc).__name__}: {exc})")
        return None


def _use_if_longer(new, series):
    if series is not None and series.notna().sum() > new[TICKER].notna().sum():
        new = new.reindex(new.index.union(series.index))
        new[TICKER] = series.reindex(new.index)
    return new


def download(today, yf_download=None, chart=None):
    """Close prices of TTF=F and its contracts, on weekdays. Yahoo sometimes returns one row of TTF=F (hosted app on
    1 Nov 2026, this job on 5 and 7 Nov 2026), so a short answer is asked again for TTF=F alone, then from the chart
    API. The longest TTF=F answer is used."""
    yf_download = yf_download or yf.download
    chart = chart or chart_closes
    end = today + pd.Timedelta(days=1)
    new = close_table(_attempt("yfinance, all tickers", lambda: yf_download(
        [TICKER] + contracts(today), start="2017-01-01", end=end, progress=False, auto_adjust=False)))
    if new[TICKER].notna().sum() < FULL_ROWS:
        alone = _attempt("yfinance, TTF=F alone", lambda: yf_download(
            TICKER, start="2017-01-01", end=end, progress=False, auto_adjust=False))
        new = _use_if_longer(new, close_table(alone)[TICKER])
    if new[TICKER].notna().sum() < FULL_ROWS:
        new = _use_if_longer(new, _attempt("Yahoo chart API", chart))
    new.index = pd.to_datetime(new.index)
    if new.index.tz is not None:
        new.index = new.index.tz_localize(None)
    new = new[new.index.dayofweek < 5].dropna(how="all")
    return new


def merge(saved, new, today):
    """The file's rows with this download's. A full download wins on the dates it has. A short one only adds rows that
    follow the file with no weekday missing, so it can neither shorten the file nor leave a gap in it. Today's row is
    still a live price and is never saved. A contract that Yahoo no longer lists keeps the prices in the file."""
    new = new[new[TICKER].notna() & (new.index < today)]
    if len(new) < FULL_ROWS:
        last = saved[TICKER].dropna().index.max()
        keep = []
        for d in new.index[new.index > last]:
            if len(pd.bdate_range(last, d)) != 2:
                break
            keep.append(d)
            last = d
        new = new.loc[keep]
    out = new.combine_first(saved).sort_index().dropna(axis=1, how="all")
    out = out[[TICKER] + sorted(c for c in out.columns if c != TICKER)]
    out = out[out[TICKER].notna()]
    out.index.name = "Date"
    return out


SETTLED_FROM_HOUR = 5   # as common.SETTLED_FROM_HOUR: from 05:00 New York the day before's row is the settlement


def settled_before(now=None):
    """First date whose row is not yet a settlement: rows of earlier dates may be saved. Until 05:00 New York the day
    before's row can still be rewritten (Yahoo writes the evening session into it until New York midnight), so it
    waits; this holds whenever the job runs, so the schedule can run at any hour."""
    ny = pd.Timestamp.now(tz="America/New_York") if now is None else pd.Timestamp(now).tz_convert("America/New_York")
    return (ny - pd.Timedelta(hours=SETTLED_FROM_HOUR)).tz_localize(None).normalize()


def main():
    today = settled_before()
    saved = pd.read_csv(PATH, index_col=0, parse_dates=True)
    new = download(today)
    print(f"{TICKER}: {new[TICKER].notna().sum()} rows downloaded to {new[TICKER].dropna().index.max()}, "
          f"{saved[TICKER].notna().sum()} in the file to {saved[TICKER].dropna().index.max()}")
    out = merge(saved, new, today)
    out.to_csv(PATH)
    last = out[TICKER].dropna().index.max()
    print(f"Saved {out.shape[0]} rows to {last}, columns {list(out.columns)}")
    newest = new[TICKER].dropna().index
    newest = newest[newest < today].max() if len(newest) else None
    if newest is not None and newest > last:
        print(f"::warning::Saved copy ends {last.date()}, Yahoo has {TICKER} to {newest.date()}: the days between "
              "were not added (short answer with a gap). The next run tries again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
