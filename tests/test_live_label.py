"""Offline test of the three wordings of a futures row (common.futures_row_state): 'live' (a row of today's New
York date: the latest price, not the settlement), 'unsettled' (a row of the last weekday before it: may not yet be
the settlement) and 'close' (older). Used by the Risk page's forward curve (common.futures_curve_words), its "Data
up to" line and common.last_value_note (Risk and Market pages); also the check that TTF=F's last row is late
against NYMEX Brent (common.ttf_row_late).

Run from the repository root with `python -m pytest tests/`. The time is passed in, so no clock and no network
are needed. After the settlement Yahoo Finance writes the evening electronic session into the day's row until New
York midnight and can rewrite it afterwards; after midnight it opens a row of the new day for the evening session.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

NY = "America/New_York"
OCT21 = pd.Timestamp("2026-10-21")
OCT22 = pd.Timestamp("2026-10-22")   # a Thursday
OCT23 = pd.Timestamp("2026-10-23")   # a Friday


def ny(text):
    return pd.Timestamp(text, tz=NY)


@pytest.mark.parametrize("now, row, state", [
    # 22 Oct 15:00 New York: after the 14:30 settlement; the 22 Oct row is today's.
    (ny("2026-10-22 15:00"), OCT22, 'live'),
    (ny("2026-10-22 15:00"), OCT21, 'unsettled'),
    # 22 Oct 20:00 New York (23 Oct 00:00 UTC): evening session, still written on the 22 Oct row.
    (ny("2026-10-22 20:00"), OCT22, 'live'),
    # The audit's moment: 03:17 UTC on 23 Oct is 23:17 on 22 Oct in New York; a naive time is read as UTC.
    (pd.Timestamp("2026-10-23 03:17", tz="UTC"), OCT22, 'live'),
    (pd.Timestamp("2026-10-23 03:17"), OCT22, 'live'),
    # 23 Oct 00:30 and 03:00 New York: the 22 Oct row may not yet be the settlement, the 23 Oct row is the
    # evening session that has just started, the 21 Oct row is a close.
    (ny("2026-10-23 00:30"), OCT22, 'unsettled'),
    (ny("2026-10-23 00:30"), OCT23, 'live'),
    (ny("2026-10-23 03:00"), OCT22, 'unsettled'),
    (ny("2026-10-23 03:00"), OCT23, 'live'),
    (ny("2026-10-23 03:00"), OCT21, 'close'),
    # Sunday 25 Oct: Friday's row is the last weekday before today; Thursday's is a close.
    (ny("2026-10-25 12:00"), OCT23, 'unsettled'),
    (ny("2026-10-25 12:00"), OCT22, 'close'),
    # Monday 26 Oct 03:00: Friday's row still is the last weekday before today.
    (ny("2026-10-26 03:00"), OCT23, 'unsettled'),
    (ny("2026-10-27 03:00"), OCT23, 'close'),
    # A past end date.
    (ny("2026-10-23 03:00"), pd.Timestamp("2020-12-31"), 'close'),
])
def test_futures_row_state(now, row, state):
    assert common.futures_row_state(row, now) == state


def test_threshold_dates():
    assert common.futures_live_from(ny("2026-10-22 15:00")) == OCT21
    assert common.futures_live_from(ny("2026-10-23 00:30")) == OCT22
    assert common.futures_live_from(ny("2026-10-25 12:00")) == OCT23
    assert common.new_york_today(pd.Timestamp("2026-10-23 03:17")) == OCT22


LIVE = (" The last value, of 22 Oct, is the latest price: it can be the current session or the evening electronic "
        "session after the settlement, not the settlement, so it can change within the hour.")
UNSETTLED = (" The last value, of 22 Oct, may not yet be the settlement: Yahoo Finance writes the evening session "
             "into the day's row until New York midnight and can rewrite it afterwards.")


@pytest.mark.parametrize("now, note", [
    (ny("2026-10-22 15:00"), LIVE),
    (ny("2026-10-22 20:00"), LIVE),
    (ny("2026-10-23 00:30"), UNSETTLED),
    (ny("2026-10-23 03:00"), UNSETTLED),
    (ny("2026-10-24 03:00"), ""),
])
def test_note(now, note):
    assert common.last_value_note("CL=F", OCT22.date(), OCT22, now) == note


def test_note_unsettled_says_neither_way():
    note = common.last_value_note("BZ=F", OCT22.date(), OCT22, ny("2026-10-23 00:30"))
    assert "may not yet be" in note and "is the settlement" not in note and "not the settlement" not in note


def test_note_past_end():
    assert common.last_value_note("CL=F", pd.Timestamp("2020-12-31").date(), pd.Timestamp("2020-12-31"),
                                  ny("2026-10-23 03:00")) == ""
    assert common.last_value_note("BZ=F", OCT21.date(), OCT21, ny("2026-10-23 03:00")) == ""


def test_note_carbon_has_no_evening_session():
    today = pd.Timestamp(common.berlin_today())
    note = common.last_value_note("CARB.L", today.date(), today - pd.Timedelta(days=1))
    assert note == " The last value is the latest close on Yahoo Finance (London Stock Exchange)."


def _curve(now, trades, front=None):
    trades = [pd.Timestamp(t) for t in trades]
    return common.futures_curve_words(pd.Series(trades), front or trades[0], "22 Oct 2026", now)


def test_curve_words_three_states():
    w = _curve(ny("2026-10-22 20:00"), ["2026-10-22"] * 3)
    assert w['state'] == 'live' and w['title'] == "latest prices 22 Oct 2026"
    assert w['hover'].startswith("Latest prices of 22 Oct")
    assert w['caption'].startswith("at its latest price: a row of 22 Oct can be the current session")
    w = _curve(ny("2026-10-23 00:30"), ["2026-10-22"] * 3)
    assert w['state'] == 'unsettled' and w['title'] == "prices of 22 Oct 2026 (may not yet be settlements)"
    assert w['hover'] == ("Prices of 22 Oct, which may not yet be settlements: Yahoo Finance writes the evening "
                          "session into the day's row until New York midnight and can rewrite it afterwards.")
    assert "latest price" not in w['caption'] and "may not yet be the settlement" in w['caption']
    w = _curve(ny("2026-10-24 03:00"), ["2026-10-22"] * 3)
    assert w['state'] == 'close' and w['title'] == "latest closes 22 Oct 2026"
    assert w['hover'] == "Both closes of 22 Oct."


def test_curve_words_mixed_rows_after_midnight():
    # 23 Oct 00:30: some contracts already have a 23 Oct evening row, others still end on 22 Oct.
    w = _curve(ny("2026-10-23 00:30"), ["2026-10-23", "2026-10-22", "2026-10-21"], front=pd.Timestamp("2026-10-23"))
    assert w['state'] == 'live'
    assert "a row of 23 Oct can be the current session" in w['caption']
    assert "a row of 22 Oct may not yet be the settlement" in w['caption']
    assert w['hover'].startswith("Latest prices of 23 Oct")


@pytest.mark.parametrize("now, ttf_last, bz_days, late", [
    # 23 Oct 00:30 New York: the 23 Oct BZ=F row is the evening session just opened, so TTF's 22 Oct is not late.
    (ny("2026-10-23 00:30"), OCT22, [OCT21, OCT22, OCT23], None),
    (ny("2026-10-23 03:00"), OCT22, [OCT21, OCT22, OCT23], None),
    # A Brent row of an earlier New York day than today that TTF lacks: late.
    (ny("2026-10-23 03:00"), OCT21, [OCT21, OCT22, OCT23], OCT22),
    (ny("2026-10-24 10:00"), OCT22, [OCT21, OCT22, OCT23], OCT23),
    (ny("2026-10-24 10:00"), OCT23, [OCT21, OCT22, OCT23], None),
    # During the New York day: today's Brent row is not compared.
    (ny("2026-10-22 15:00"), OCT21, [OCT21, OCT22], None),
    # Sunday: Friday's Brent row is before today, so a missing TTF Friday is late.
    (ny("2026-10-25 12:00"), OCT22, [OCT22, OCT23], OCT23),
])
def test_ttf_row_late(now, ttf_last, bz_days, late):
    bz = pd.Series(100.0, index=pd.DatetimeIndex(bz_days))
    assert common.ttf_row_late(ttf_last, bz, now) == late
