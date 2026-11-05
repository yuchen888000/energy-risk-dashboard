"""Update data/ttf_saved.csv, the saved copy of TTF prices the app falls back on when Yahoo Finance returns too
little of TTF=F to the hosted app. Run once a day by .github/workflows/refresh-ttf.yml.

The file changes only if this download has at least as many TTF=F rows as the file. Otherwise the script exits
with an error and the file stays as it is, so a short download can never replace good history.
Prices from this download win; a contract that Yahoo no longer lists keeps the prices already in the file.
"""
import sys

import pandas as pd
import yfinance as yf

PATH = "data/ttf_saved.csv"
TICKER = "TTF=F"
MONTH_CODES = "FGHJKMNQUVXZ"


def contracts(today):
    """Monthly TTF contracts from two months back to the app's forward curve's last month (15 months ahead)."""
    months = pd.period_range(today.to_period("M") - 2, today.to_period("M") + 15, freq="M")
    return [f"TTF{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM" for p in months]


def main():
    today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
    saved = pd.read_csv(PATH, index_col=0, parse_dates=True)
    data = yf.download([TICKER] + contracts(today), start="2017-01-01", end=today + pd.Timedelta(days=1),
                       progress=False, auto_adjust=False)
    if data is None or data.empty or TICKER not in data["Close"]:
        print(f"FAIL: no {TICKER} in the download; file not changed.")
        return 1
    new = data["Close"]
    new = new[new.index.dayofweek < 5].dropna(how="all")
    new.index = pd.to_datetime(new.index).tz_localize(None)
    have, got = saved[TICKER].notna().sum(), new[TICKER].notna().sum()
    print(f"{TICKER}: {got} rows downloaded to {new[TICKER].dropna().index.max()}, {have} in the file to "
          f"{saved[TICKER].dropna().index.max()}")
    if got < have:
        print("FAIL: download shorter than the file; file not changed.")
        return 1
    out = new.combine_first(saved).sort_index().dropna(axis=1, how="all")
    out = out[[TICKER] + sorted(c for c in out.columns if c != TICKER)]
    # Today's row is still a live price, so it is never saved: the app takes it from Yahoo.
    out = out[out[TICKER].notna() & (out.index < today)]
    out.index.name = "Date"
    out.to_csv(PATH)
    print(f"Saved {out.shape[0]} rows, columns {list(out.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
