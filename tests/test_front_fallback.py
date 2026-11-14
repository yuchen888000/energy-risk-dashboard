"""front_month_contract on the download's last day, when the continuous row and the contract row differ
(13 Nov 2026, 07:51 UTC: TTF=F 93.70 for 12 Nov, TTFZ26 already 92.26)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def _batch(last_cont):
    idx = pd.to_datetime(["2026-11-10", "2026-11-11", "2026-11-12"])
    return pd.DataFrame({"TTF=F": [84.732, 86.652, last_cont],
                         "TTFZ26.NYM": [84.732, 86.652, 92.259],
                         "TTFF27.NYM": [84.539, 86.483, 92.081]}, index=idx)


def test_last_day_unmatched_uses_day_before():
    b = _batch(93.70)
    d = pd.Timestamp("2026-11-12")
    assert common.match_front_contract(b, "TTF=F", d) is None
    out = common._front_fallback(b, "TTF=F", "TTF", d)
    assert out["matched"] and out["symbol"] == "TTFZ26.NYM"
    assert out["matched_on"] == pd.Timestamp("2026-11-11")


def test_rule_disagrees_stays_unmatched():
    b = _batch(93.70)
    # A day on which the rule gives Jan 27 (after the Dec 26 expiry on 27 Nov): the day before's Dec 26 match
    # must not be carried over.
    b.index = pd.to_datetime(["2026-11-25", "2026-11-27", "2026-11-30"])
    out = common._front_fallback(b, "TTF=F", "TTF", pd.Timestamp("2026-11-30"))
    assert not out["matched"] and out["matched_on"] is None


def test_not_last_day_stays_unmatched():
    b = _batch(93.70)
    b.loc[pd.Timestamp("2026-11-13")] = [91.0, 91.0, 90.8]
    out = common._front_fallback(b, "TTF=F", "TTF", pd.Timestamp("2026-11-12"))
    assert not out["matched"]
