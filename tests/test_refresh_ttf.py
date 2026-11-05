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


def test_full_download_corrects_old_rows_and_keeps_today_out():
    full = _frame(pd.bdate_range("2025-10-01", "2026-11-05"), 50.0)
    out = rt.merge(SAVED, full, TODAY)
    assert out.index.max() == pd.Timestamp("2026-11-04")
    assert out.loc["2026-11-03", rt.TICKER] == full.loc["2026-11-03", rt.TICKER]
