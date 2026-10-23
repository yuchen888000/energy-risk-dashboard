"""Offline test of common.contract_expiry on the exchange holiday calendars (the `holidays` package): the dates
below are CME's published last trading days (CME product calendars, read on 2026-10-23: 425 WTI, 424 Brent
Last Day Financial, 8378 Dutch TTF). Every listed contract matched; these are the ones the old weekday-only rule
got wrong, plus a few ordinary months. Also common.rule_roll_days (candidate roll days by the rule).

Run from the repository root with `python -m pytest tests/`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


@pytest.mark.parametrize("root, delivery, last_trade", [
    # WTI (CME 200102.F): holidays before the 25th move the date.
    ("CL", "2026-11-01", "2026-10-20"),
    ("CL", "2026-12-01", "2026-11-20"),
    ("CL", "2027-01-01", "2026-12-21"),   # Christmas Friday 25 Dec 2026; weekday rule gave 22 Dec
    ("CL", "2027-12-01", "2027-11-19"),   # Thanksgiving 25 Nov 2027
    ("CL", "2028-01-01", "2027-12-20"),
    ("CL", "2029-01-01", "2028-12-19"),
    # Brent (ICE, NYMEX BZ 698102.E): London bank holidays at month end, December one day earlier.
    ("BZ", "2026-12-01", "2026-10-30"),
    ("BZ", "2027-02-01", "2026-12-30"),
    ("BZ", "2027-07-01", "2027-05-28"),   # Spring bank holiday 31 May 2027
    ("BZ", "2029-05-01", "2029-03-29"),   # Good Friday 30 Mar 2029
    # Dutch TTF (CME 1159102.E): second London business day before the month, also a NYMEX business day.
    ("TTF", "2026-11-01", "2026-10-29"),
    ("TTF", "2026-12-01", "2026-11-27"),
    ("TTF", "2027-06-01", "2027-05-27"),  # Spring bank holiday 31 May 2027
    ("TTF", "2027-09-01", "2027-08-27"),  # Summer bank holiday 30 Aug 2027
    ("TTF", "2030-12-01", "2030-11-27"),  # Thanksgiving 28 Nov 2030 is not a NYMEX business day
    ("TTF", "2031-12-01", "2031-11-26"),
])
def test_contract_expiry_matches_cme(root, delivery, last_trade):
    assert common.contract_expiry(root, delivery) == pd.Timestamp(last_trade)


def test_rule_roll_days_ttf_march_2026():
    # TTF Apr 26 stopped on Mon 30 Mar 2026; the series may switch that day or the next NYMEX business day.
    days = pd.bdate_range("2026-03-20", "2026-04-03")
    cand = common.rule_roll_days("TTF", days)
    assert sorted(cand) == [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
    text = common.rule_roll_sentence(cand)
    assert text.startswith("By the exchange rule, the Apr 26 contract's last trading day was 30 Mar 2026")
    assert "30 Mar 2026 or 31 Mar 2026" in text and "not verified" in text


def test_rule_roll_days_none_for_brent_and_carbon():
    days = pd.bdate_range("2026-03-20", "2026-04-03")
    assert common.rule_roll_days("BZ", days) == {}
    assert common.rule_roll_days("CARB", days) == {}


def test_roll_sentence_skips_candidate_of_identified_roll():
    # TTF Oct 26 stopped on 29 Sep 2026 and the roll check identified 30 Sep: 29 Sep is not offered as a candidate.
    idx = pd.bdate_range("2026-09-14", "2026-10-22")
    window = pd.Series(0.01, index=idx)
    rc = dict(rolls=[(pd.Timestamp("2026-09-30"), "TTFX26.NYM", 0.0366, 0.0416)],
              unknown=list(pd.bdate_range("2026-09-14", "2026-09-29")), rule_only=[], adjusted_vol=3.2)
    text = common.roll_sentence("TTF=F", rc, window)
    assert "By the exchange rule" not in text
    rc['rolls'] = []
    rc['adjusted_vol'] = None
    text = common.roll_sentence("TTF=F", rc, window)
    assert "the Oct 26 contract's last trading day was 29 Sep 2026" in text
