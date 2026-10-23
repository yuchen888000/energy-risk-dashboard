"""Offline test of common.contract_expiry on the exchange holiday calendars (the `holidays` package): the dates
below are CME's published last trading days (CME product calendars, read on 2026-10-23: 425 WTI, 424 Brent
Last Day Financial, 8378 Dutch TTF). Every listed contract matched; these are the ones the old weekday-only rule
got wrong, plus a few ordinary months. Also common.rule_roll_days (candidate roll days by the rule).

Run from the repository root with `python -m pytest tests/`.
"""
import sys
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("root, delivery, last_trade", [
    # US holidays on which CME trades but publishes no settlement are not NYMEX business days.
    ("CL", "2020-06-01", "2020-05-19"),   # Memorial Day 25 May 2020; The Energy Year, 19 May 2020
    ("CL", "2026-06-01", "2026-05-19"),   # Memorial Day 25 May 2026
    ("CL", "2037-06-01", "2037-05-19"),   # Memorial Day 25 May 2037
    ("TTF", "2022-06-01", "2022-05-27"),  # second London business day 30 May 2022 is Memorial Day
])
def test_contract_expiry_us_no_settlement_days(root, delivery, last_trade):
    assert common.contract_expiry(root, delivery) == pd.Timestamp(last_trade)


@pytest.mark.parametrize("day, open_", [
    ("2026-06-19", False),   # Juneteenth: CME's 2026 notice, no NYMEX settlement
    ("2026-01-19", False),   # Martin Luther King Jr. Day
    ("2026-09-07", False),   # Labor Day
    ("2026-07-03", False),   # Friday before Saturday 4 July
    ("2025-01-09", True),    # day of mourning, energy traded normally (CME notice of 30 Dec 2024)
    ("2026-04-06", True),    # Easter Monday: a UK holiday, NYMEX open
])
def test_nymex_business_days(day, open_):
    assert bool(np.is_busday(day, busdaycal=common.NYMEX_BDAYS)) is open_


def test_rule_roll_days_ttf_uk_only_holiday():
    # TTF Sep 21 stopped on Fri 27 Aug 2021; Mon 30 Aug 2021 is a UK bank holiday on which TTF=F has a price, so it
    # is the next day of the series, not 31 Aug.
    cand = common.rule_roll_days("TTF", pd.bdate_range("2021-08-20", "2021-09-03"))
    assert sorted(cand) == [pd.Timestamp("2021-08-27"), pd.Timestamp("2021-08-30")]


@pytest.mark.parametrize("now, expected", [
    ("2026-10-23 09:54", True),    # 10:54 London (BST), session open
    ("2026-10-23 15:59", True),    # 16:59 London
    ("2026-10-23 16:00", False),   # 17:00 London
    ("2026-10-24 09:00", False),   # next day: the 23 Oct row is a close
])
def test_london_session_row(now, expected):
    assert common.london_session_row("2026-10-23", pd.Timestamp(now, tz="UTC")) is expected


def test_brent_roll_days_follow_the_expiry_rule():
    # ICE Brent Oct 26 stopped on Fri 28 Aug 2026 (Mon 31 Aug is an English bank holiday); BZ=F trades on 31 Aug,
    # so that is the roll day, not 1 Sep (audit Risk 2026-10-23 round 5, 2-1).
    idx = pd.DatetimeIndex(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02"])
    assert list(common.brent_roll_days(idx)) == [pd.Timestamp("2026-08-31")]
    # December: the Feb contract stops one London business day earlier, so the roll is the last trading day of
    # December and there is none on the first trading day of January.
    idx = pd.bdate_range("2026-12-22", "2027-01-06")
    days = list(common.brent_roll_days(idx))
    assert pd.Timestamp("2026-12-31") in days and pd.Timestamp("2027-01-01") not in days
    assert pd.Timestamp("2027-01-04") not in days


def test_brent_roll_set_uses_the_identified_switch():
    # On 23 Oct 2026 BZ=F already followed the Jan 27 contract; the rule day of that switch (2 Nov, after the Dec 26
    # expiry on 30 Oct) is then not a roll day, and 23 Oct is (audit Market 2026-10-23 round 5, 1-1 and 2-1).
    idx = pd.bdate_range("2026-09-28", "2026-11-06")
    rc = dict(rolls=[(pd.Timestamp("2026-10-23"), "BZF27.NYM", -0.025, 0.0155)], unknown=[], rule_only=[],
              adjusted_vol=2.0)
    days = list(common.brent_roll_set(idx, rc))
    assert pd.Timestamp("2026-10-23") in days and pd.Timestamp("2026-11-02") not in days
    assert pd.Timestamp("2026-10-01") in days            # the earlier rule day is kept
    assert pd.Timestamp("2026-11-02") in list(common.brent_roll_set(idx, None))


@pytest.mark.parametrize("day, n, expected", [
    ("2025-07-04", 5, "2025-07-11"),   # 4 Jul 2025, a NYMEX holiday with a Yahoo row: counted from, no error
    ("2025-07-05", 5, "2025-07-11"),   # Saturday after it
    ("2023-11-23", 5, "2023-11-30"),   # Thanksgiving 2023, Yahoo row
    ("2026-01-16", 5, "2026-01-26"),   # skips Martin Luther King Jr. Day
    ("2022-05-23", 5, "2022-05-31"),   # skips Memorial Day
])
def test_nymex_days_after(day, n, expected):
    assert common.nymex_days_after(day, n) == pd.Timestamp(expected)


def test_nymex_path_days_start_on_a_holiday():
    path = common.nymex_path_days("2025-07-04", 5)
    assert path[0] == pd.Timestamp("2025-07-04") and path[-1] == pd.Timestamp("2025-07-11") and len(path) == 6


def test_ice_endex_ttf_expiry_differs_from_nymex_in_june_2022():
    assert common.ice_endex_ttf_expiry("2022-06-01") == pd.Timestamp("2022-05-30")
    assert common.contract_expiry("TTF", "2022-06-01") == pd.Timestamp("2022-05-27")
    assert common.ice_endex_ttf_expiry("2026-11-01") == common.contract_expiry("TTF", "2026-11-01")


def test_last_value_note_uses_now_for_the_three_day_rule():
    # The London-session note can be tested at a fixed past time (audit Risk round 5, 4-5).
    note = common.last_value_note("CARB.L", "2026-03-30", "2026-03-30", now=pd.Timestamp("2026-03-30 14:59", tz="UTC"))
    assert "may be a price during today's London session" in note
    assert common.last_value_note("CARB.L", "2026-03-30", "2026-03-30",
                                  now=pd.Timestamp("2026-04-10 12:00", tz="UTC")) == ""


@pytest.mark.parametrize("root, to", [("CL", "February 2037"), ("BZ", "March 2035"), ("TTF", "December 2036")])
def test_expiry_cal_note_names_each_products_checked_range(root, to):
    assert f"(to {to})" in common.expiry_cal_note(root)
