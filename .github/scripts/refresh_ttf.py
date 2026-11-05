"""Update data/ttf_saved.csv, the saved copy of TTF prices the app falls back on when Yahoo Finance returns too
little of TTF=F to the hosted app. Run once a day by .github/workflows/refresh-ttf.yml.

A short download never replaces history: see merge. The job does not fail on one. The Dashboard patrol reports a
saved copy that falls behind.
"""
import sys

import pandas as pd
import yfinance as yf

PATH = "data/ttf_saved.csv"
TICKER = "TTF=F"
FULL_ROWS = 250   # fewer rows than this means Yahoo cut the answer short
MONTH_CODES = "FGHJKMNQUVXZ"


def contracts(today):
    """Monthly TTF contracts from two months back to the app's forward curve's last month (15 months ahead)."""
    months = pd.period_range(today.to_period("M") - 2, today.to_period("M") + 15, freq="M")
    return [f"TTF{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM" for p in months]


def download(today):
    """Close prices of TTF=F and its contracts, on weekdays. Yahoo sometimes returns one row of TTF=F (hosted app on
    1 Nov 2026, this job on 5 Nov 2026), so a short answer is asked again for TTF=F alone."""
    data = yf.download([TICKER] + contracts(today), start="2017-01-01", end=today + pd.Timedelta(days=1),
                       progress=False, auto_adjust=False)
    if data is None or data.empty or TICKER not in data["Close"]:
        new = pd.DataFrame(columns=[TICKER], index=pd.DatetimeIndex([]))
    else:
        new = data["Close"]
    if new[TICKER].notna().sum() < FULL_ROWS:
        alone = yf.download(TICKER, start="2017-01-01", end=today + pd.Timedelta(days=1), progress=False,
                            auto_adjust=False)
        if alone is not None and not alone.empty:
            alone = alone["Close"].squeeze()
            if alone.notna().sum() > new[TICKER].notna().sum():
                new = new.reindex(new.index.union(alone.index))
                new[TICKER] = alone.reindex(new.index)
    new = new[new.index.dayofweek < 5].dropna(how="all")
    new.index = pd.to_datetime(new.index).tz_localize(None)
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


def main():
    today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
    saved = pd.read_csv(PATH, index_col=0, parse_dates=True)
    new = download(today)
    print(f"{TICKER}: {new[TICKER].notna().sum()} rows downloaded, {saved[TICKER].notna().sum()} in the file to "
          f"{saved[TICKER].dropna().index.max()}")
    out = merge(saved, new, today)
    out.to_csv(PATH)
    print(f"Saved {out.shape[0]} rows to {out[TICKER].dropna().index.max()}, columns {list(out.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
