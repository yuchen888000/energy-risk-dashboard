"""The daily refresh of the saved TTF prices: a short answer from Yahoo must not shorten the file or leave a gap."""
import importlib.util
import os

import pandas as pd

_path = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "refresh_ttf.py")
_spec = importlib.util.spec_from_file_location("refresh_ttf", _path)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
TODAY = pd.Timestamp("2026-11-05")


def _frame(days, start=100.0):
    return pd.DataFrame({rt.TICKER: [start + i for i in range(len(days))]}, index=pd.to_datetime(days))


SAVED = _frame(pd.bdate_range("2025-10-01", "2026-11-03"))


def test_short_download_of_today_changes_nothing():
    out = rt.merge(SAVED, _frame(["2026-11-05"], 88.0), TODAY)
    assert len(out) == len(SAVED) and out.index.max() == pd.Timestamp("2026-11-03")


def test_short_download_adds_the_next_day():
    out = rt.merge(SAVED, _frame(["2026-11-04"], 88.0), TODAY)
    assert out.index.max() == pd.Timestamp("2026-11-04") and out.iloc[-1][rt.TICKER] == 88.0


def test_short_download_leaves_no_gap():
    out = rt.merge(SAVED[SAVED.index <= "2026-10-30"], _frame(["2026-11-04"], 88.0), TODAY)
    assert out.index.max() == pd.Timestamp("2026-10-30")


def _yf(days, start=88.0, tickers=(rt.TICKER,)):
    """A yf.download answer in yfinance 1.x's shape: columns (field, ticker), even for one ticker."""
    idx = pd.to_datetime(days)
    cols = pd.MultiIndex.from_product([["Adj Close", "Close"], list(tickers)], names=["Price", "Ticker"])
    return pd.DataFrame([[start + i] * len(cols) for i in range(len(idx))], index=idx, columns=cols)


def _no_chart():
    raise AssertionError("chart API not needed")


def test_one_row_answer_does_not_fail():
    """7 Nov 2026: Yahoo gave one row of TTF=F alone and squeeze turned it into a number, so the job failed."""
    new = rt.download(pd.Timestamp("2026-11-07"), yf_download=lambda t, **k: _yf(["2026-11-06"], 85.588),
                      chart=lambda: None)
    assert list(new.index) == [pd.Timestamp("2026-11-06")] and new[rt.TICKER].iloc[0] == 85.588
    out = rt.merge(_frame(pd.bdate_range("2025-10-01", "2026-11-05")), new, pd.Timestamp("2026-11-07"))
    assert out.index.max() == pd.Timestamp("2026-11-06")


def test_empty_answers_and_errors_do_not_fail():
    def broken(*a, **k):
        raise RuntimeError("Yahoo down")
    new = rt.download(pd.Timestamp("2026-11-07"), yf_download=broken, chart=broken)
    assert new.empty and rt.TICKER in new.columns
    new = rt.download(pd.Timestamp("2026-11-07"), yf_download=lambda t, **k: pd.DataFrame(), chart=lambda: None)
    out = rt.merge(SAVED, new, TODAY)
    assert len(out) == len(SAVED)


def test_chart_api_fills_a_short_answer_and_closes_a_gap():
    """yfinance gives one row; the chart API's three months add every missing day in order."""
    chart = pd.Series([90.679, 85.588], index=pd.to_datetime(["2026-11-05", "2026-11-06"]))
    new = rt.download(pd.Timestamp("2026-11-07"), yf_download=lambda t, **k: _yf(["2026-11-06"], 85.588),
                      chart=lambda: chart)
    out = rt.merge(_frame(pd.bdate_range("2025-10-01", "2026-11-04")), new, pd.Timestamp("2026-11-07"))
    assert out.loc["2026-11-05", rt.TICKER] == 90.679 and out.index.max() == pd.Timestamp("2026-11-06")


def test_full_answer_with_contracts_keeps_all_columns():
    days = pd.bdate_range("2025-10-01", "2026-11-06")
    new = rt.download(pd.Timestamp("2026-11-07"), chart=_no_chart,
                      yf_download=lambda t, **k: _yf(days, 50.0, (rt.TICKER, "TTFZ26.NYM")))
    assert len(new) == len(days) and "TTFZ26.NYM" in new.columns


def test_chart_closes_reads_new_york_dates():
    ts = [int(pd.Timestamp(d, tz="America/New_York").timestamp()) for d in ("2026-11-05 00:00", "2026-11-06 00:00")]
    payload = {"chart": {"result": [{"meta": {"exchangeTimezoneName": "America/New_York"}, "timestamp": ts,
                                     "indicators": {"quote": [{"close": [90.679, None]}]}}]}}
    s = rt.chart_closes(read=lambda url: payload)
    assert list(s.index) == [pd.Timestamp("2026-11-05")] and s.iloc[0] == 90.679


def test_row_waits_until_0500_new_york():
    assert rt.settled_before(pd.Timestamp("2026-11-06 09:00", tz="UTC")) == pd.Timestamp("2026-11-05")  # 04:00 NY
    assert rt.settled_before(pd.Timestamp("2026-11-06 10:30", tz="UTC")) == pd.Timestamp("2026-11-06")  # 05:30 NY
    assert rt.settled_before(pd.Timestamp("2026-11-07 15:00", tz="UTC")) == pd.Timestamp("2026-11-07")  # Saturday


def test_full_download_corrects_old_rows_and_keeps_today_out():
    full = _frame(pd.bdate_range("2025-10-01", "2026-11-05"), 50.0)
    out = rt.merge(SAVED, full, TODAY)
    assert out.index.max() == pd.Timestamp("2026-11-04")
    assert out.loc["2026-11-03", rt.TICKER] == full.loc["2026-11-03", rt.TICKER]
