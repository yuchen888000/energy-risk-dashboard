"""The saved copy of TTF prices and the live prices after it: a business day missing between them must not become
a one-day return (the hosted app showed -18.20% for 4 Nov 2026, three trading days of moves)."""
import pandas as pd

import common


def _run(monkeypatch, live_days, saved_days, ticker="TTF=F"):
    saved = pd.DataFrame({ticker: [100.0 + i for i in range(len(saved_days))]}, index=pd.to_datetime(saved_days))
    live = pd.DataFrame({ticker: [90.0 + i for i in range(len(live_days))]}, index=pd.to_datetime(live_days))
    live.columns = pd.MultiIndex.from_product([["Close"], live.columns])
    monkeypatch.setattr(common, "_read_saved", lambda t: saved)
    monkeypatch.setattr(common, "_batch_contracts", lambda t, d: [])
    monkeypatch.setattr(common.yf, "download", lambda *a, **k: live)
    return common._download_batch.__wrapped__(ticker, pd.Timestamp("2026-10-01").date(), pd.Timestamp("2026-11-04").date())


def test_gap_after_saved_copy_stops_the_data(monkeypatch):
    out = _run(monkeypatch, ["2026-11-04"], ["2026-10-29", "2026-10-30"])
    assert out.index[-1] == pd.Timestamp("2026-10-30")
    cut = out.attrs["cut"]
    assert cut["skipped"] == [pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03")]
    assert cut["dropped"] == 1


def test_next_trading_day_is_kept(monkeypatch):
    out = _run(monkeypatch, ["2026-11-04"], ["2026-11-02", "2026-11-03"])
    assert out.index[-1] == pd.Timestamp("2026-11-04")
    assert out.attrs["cut"] is None


def test_monday_after_friday_is_not_a_gap(monkeypatch):
    out = _run(monkeypatch, ["2026-11-02"], ["2026-10-29", "2026-10-30"])
    assert out.index[-1] == pd.Timestamp("2026-11-02")
    assert out.attrs["cut"] is None


def test_nymex_holiday_is_not_a_gap():
    # Labor Day 2026: no NYMEX settlement, Friday 4 Sep to Tuesday 8 Sep is consecutive.
    assert common._skipped_days(["2026-09-04", "2026-09-08"], common.NYMEX_BDAYS) == []


def test_skipped_days_in_the_latest_rows():
    days = pd.to_datetime(["2026-11-02", "2026-11-04"])
    assert common._skipped_days(list(days), common.NYMEX_BDAYS) == [pd.Timestamp("2026-11-03")]


def test_curve_flags_one_price_and_old_quote():
    curve = pd.DataFrame({
        "Contract": ["A", "B", "C"], "Delivery": pd.to_datetime(["2026-12-01", "2027-01-01", "2027-02-01"]),
        "Price": [85.0, 87.5, 90.5], "Last trade": pd.to_datetime(["2026-11-04", "2026-11-04", "2026-11-03"]),
        "Last hour": [pd.Timestamp("2026-11-04 19:00", tz="Europe/Berlin"),
                      pd.Timestamp("2026-11-04 11:00", tz="Europe/Berlin"), pd.NaT],
        "Note": ["", "", common.CURVE_SAVED]})
    flagged = dict(common.curve_flagged(curve))
    assert 0 not in flagged
    assert "11:00" in flagged[1]
    assert common.CURVE_SAVED in flagged[2] and "03 Nov" in flagged[2]
