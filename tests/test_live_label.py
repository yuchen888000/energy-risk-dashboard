"""Offline test of the "latest price" or "close" rule for futures rows (common.futures_row_live), used by the Risk
page's forward curve, its "Data up to" line and common.last_value_note (Risk and Market pages).

Run from the repository root with `python -m pytest tests/`. The time is passed in, so no clock and no network
are needed. CME settles WTI in the New York afternoon; after that Yahoo Finance writes the evening electronic
session on the settled day's row, so that row is only a close once a full New York day has passed after it.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

NY = "America/New_York"
OCT22 = pd.Timestamp("2026-10-22")   # a Thursday
OCT21 = pd.Timestamp("2026-10-21")


def ny(text):
    return pd.Timestamp(text, tz=NY)


@pytest.mark.parametrize("now, row, live", [
    # 22 Oct 15:00 New York: after the 14:30 settlement, before the evening session.
    (ny("2026-10-22 15:00"), OCT22, True),
    # 22 Oct 20:00 New York (23 Oct 00:00 UTC): evening session, written on the 22 Oct row.
    (ny("2026-10-22 20:00"), OCT22, True),
    # 23 Oct 03:00 New York: the 22 Oct row can still hold the overnight price; 21 Oct has a full day after it.
    (ny("2026-10-23 03:00"), OCT22, True),
    (ny("2026-10-23 03:00"), OCT21, False),
    # The audit's moment: 03:17 UTC on 23 Oct is 23:17 on 22 Oct in New York.
    (pd.Timestamp("2026-10-23 03:17", tz="UTC"), OCT22, True),
    # A naive time is read as UTC.
    (pd.Timestamp("2026-10-23 03:17"), OCT22, True),
    # Weekend and Monday: Friday's row is the latest until Tuesday in New York.
    (ny("2026-10-24 10:00"), pd.Timestamp("2026-10-23"), True),
    (ny("2026-10-26 03:00"), pd.Timestamp("2026-10-23"), True),
    (ny("2026-10-27 03:00"), pd.Timestamp("2026-10-23"), False),
    # A past end date.
    (ny("2026-10-23 03:00"), pd.Timestamp("2020-12-31"), False),
])
def test_futures_row_live(now, row, live):
    assert common.futures_row_live(row, now) is live


def test_threshold_dates():
    assert common.futures_live_from(ny("2026-10-22 15:00")) == OCT21
    assert common.futures_live_from(ny("2026-10-22 20:00")) == OCT21
    assert common.futures_live_from(ny("2026-10-23 03:00")) == OCT22
    assert common.futures_live_from(ny("2026-10-25 12:00")) == pd.Timestamp("2026-10-23")


@pytest.mark.parametrize("now", [ny("2026-10-22 15:00"), ny("2026-10-22 20:00"), ny("2026-10-23 03:00")])
def test_note_live(now):
    note = common.last_value_note("CL=F", OCT22.date(), OCT22, now)
    assert "latest price" in note and "evening electronic session" in note and "22 Oct" in note
    assert "close" not in note


def test_note_past_end():
    assert common.last_value_note("CL=F", pd.Timestamp("2020-12-31").date(), pd.Timestamp("2020-12-31"),
                                  ny("2026-10-23 03:00")) == ""
    # 21 Oct row with a full trading day after it: a close, no note.
    assert common.last_value_note("BZ=F", OCT21.date(), OCT21, ny("2026-10-23 03:00")) == ""


def test_note_carbon_has_no_evening_session():
    today = pd.Timestamp.today().normalize()
    note = common.last_value_note("CARB.L", today.date(), today - pd.Timedelta(days=1))
    assert note == " The last value is the latest close on Yahoo Finance (London Stock Exchange)."

