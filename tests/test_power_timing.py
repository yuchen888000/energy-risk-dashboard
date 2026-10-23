"""Offline test of the Power page's timing sentence: which day the gas (TTF) and carbon (EUA) prices of a delivery
day are from (views/power.py, _timing_sentence and _gas_late).

Run from the repository root with `python -m pytest tests/`. The page itself is not executed (it downloads prices
and calls Streamlit): the block between "# ─── Timing sentence" and "# ─── End of timing sentence ───" is cut out of
the page source and run on its own, so the test uses the page's own code and needs no network. The dates are those
of the page's data (TTF=F, BZ=F from Yahoo Finance, EEX general auctions), checked by an independent script.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common  # noqa: E402

SRC = (ROOT / "views" / "power.py").read_text(encoding="utf-8")
_start = SRC.index("# ─── Timing sentence")
_ns = {"pd": pd, "common": common}
exec(SRC[_start:SRC.index("# ─── End of timing sentence ───")], _ns)
sentence, gas_late = _ns["_timing_sentence"], _ns["_gas_late"]

T = pd.Timestamp
WHY = "Monday's power is auctioned on Sunday"


# The 7 delivery days since 2020 on which carbon is older than gas but gas is not from the last weekday before
# delivery either: the old sentence said gas was from that weekday, while the price printed next to it was older.
@pytest.mark.parametrize("trade, ttf, eua, eua_txt, ttf_txt", [
    ("2025-12-25", "2025-12-24", "2025-12-15", "any weekday from Tue 16 Dec to Thu 25 Dec", "Thu 25 Dec"),
    ("2024-12-25", "2024-12-24", "2024-12-16", "any weekday from Tue 17 Dec to Wed 25 Dec", "Wed 25 Dec"),
    ("2020-05-25", "2020-05-22", "2020-05-20", "any weekday from Thu 21 May to Mon 25 May", "Mon 25 May"),
    ("2022-12-26", "2022-12-23", "2022-12-19", "any weekday from Tue 20 Dec to Mon 26 Dec", "Mon 26 Dec"),
])
def test_neither_when_gas_is_not_from_the_trade_day(trade, ttf, eua, eua_txt, ttf_txt):
    s = sentence(T(trade), T(ttf), T(eua))
    assert s.startswith(f"Neither is from the last weekday before this delivery day, {T(trade):%a %d %b}:")
    assert f"no EUA auction on {eua_txt} and no TTF close on {ttf_txt}, so the last earlier ones are used." in s
    assert "Gas is taken from" not in s and WHY not in s


def test_easter_monday_2024_brent_clause_merged():
    # Delivery Tue 02 Apr 2024: TTF=F has no price on Fri 29 Mar (Good Friday) or Mon 01 Apr (Easter Monday);
    # BZ=F has one on 01 Apr. The Brent clause is part of the sentence, said once.
    bz = pd.Series([87.48, 87.42, 88.92], index=pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-01"]))
    late, recent = gas_late(T("2024-03-28"), T("2024-04-01"), bz, now=T("2026-10-23 12:00", tz="Europe/Berlin"))
    assert late == T("2024-04-01") and recent is False
    s = sentence(T("2024-04-01"), T("2024-03-28"), T("2024-03-28"), late, recent)
    assert s == ("Neither is from the last weekday before this delivery day, Mon 01 Apr: this page's data has no EUA "
                 "auction on Fri 29 Mar or Mon 01 Apr and no TTF close on Fri 29 Mar or Mon 01 Apr, although NYMEX "
                 "Brent (`BZ=F`, normally the same trading days) has one for Mon 01 Apr, so the last earlier ones "
                 "are used.")
    assert "than that" not in s and "Yahoo Finance's `TTF=F` has no price" not in s


def test_good_friday_2026_neither():
    s = sentence(T("2026-04-03"), T("2026-04-02"), T("2026-04-02"))
    assert s == ("Neither is from the last weekday before this delivery day, Fri 03 Apr: this page's data has no EUA "
                 "auction on Fri 03 Apr and no TTF close on Fri 03 Apr, so the last earlier ones are used.")


def test_gas_from_trade_day_carbon_older_keeps_reason():
    # Delivery Thu 24 Sep 2026: gas Wed 23 Sep, carbon Tue 22 Sep.
    s = sentence(T("2026-09-23"), T("2026-09-23"), T("2026-09-22"))
    assert s.startswith("Gas is taken from the last weekday before this delivery day, Wed 23 Sep, and carbon from "
                        "the last auction before it (no auction on Wed 23 Sep). " + WHY)


def test_both_from_trade_day_unchanged():
    s = sentence(T("2026-10-22"), T("2026-10-22"), T("2026-10-22"))
    assert s.startswith("Both are taken from the last weekday before this delivery day, Thu 22 Oct. " + WHY)


def test_carbon_from_trade_day_gas_not():
    # Delivery Fri 28 Nov 2025: US Thanksgiving on Thu 27 Nov, no TTF=F price; EUA auction that day.
    s = sentence(T("2025-11-27"), T("2025-11-26"), T("2025-11-27"))
    assert s == ("Carbon is taken from the last weekday before this delivery day, Thu 27 Nov, but gas is not: this "
                 "page's data has no TTF close on Thu 27 Nov, so an earlier gas price is used.")
    assert "Both are taken" not in s and WHY not in s


NY = "America/New_York"


def test_late_judged_only_on_brent_rows_before_new_york_today():
    # Trade day Thu 22 Oct 2026, TTF=F last row Wed 21 Oct. A BZ=F row of today's New York date does not count.
    bz = pd.Series([110.0, 111.0], index=pd.DatetimeIndex(["2026-10-21", "2026-10-22"]))
    late, recent = gas_late(T("2026-10-21"), T("2026-10-22"), bz, now=T("2026-10-22 15:00", tz=NY))
    assert late is None and recent is False
    # After New York midnight the 22 Oct row is of the day before: TTF is late, and the price can still arrive.
    late, recent = gas_late(T("2026-10-21"), T("2026-10-22"), bz, now=T("2026-10-23 01:00", tz=NY))
    assert late == T("2026-10-22") and recent is True
    s = sentence(T("2026-10-22"), T("2026-10-21"), T("2026-10-22"), late, recent)
    assert s == ("Carbon is taken from the last weekday before this delivery day, Thu 22 Oct, but gas is not: this "
                 "page's data has no TTF close on Thu 22 Oct yet, although NYMEX Brent (`BZ=F`, normally the same "
                 "trading days) has one for Thu 22 Oct, so an earlier gas price is used. This spread will change "
                 "when that TTF price arrives.")


def test_no_late_when_brent_also_missing():
    # US Thanksgiving: neither TTF=F nor BZ=F has a price on Thu 27 Nov 2025.
    bz = pd.Series([63.1, 63.2], index=pd.DatetimeIndex(["2025-11-25", "2025-11-26"]))
    assert gas_late(T("2025-11-26"), T("2025-11-27"), bz, now=T("2026-10-23 12:00", tz=NY)) == (None, False)


def test_no_late_check_when_gas_is_from_trade_day():
    bz = pd.Series([110.0], index=pd.DatetimeIndex(["2026-10-22"]))
    assert gas_late(T("2026-10-22"), T("2026-10-22"), bz, now=T("2026-10-24 12:00", tz=NY)) == (None, False)
