"""Offline test of level_facts in views/market.py: a current value and a level written the same on the page are
given to the AI as EQUAL, not above or below (TTF on 10 Nov 2026: 30-day volatility 3.436% daily, 3-year average
3.443% daily, both "55% ann. (3.44% daily)"; the summary said volatility was below a 3-year average of that same
figure). The block is cut out of the page source and run on its own, as in test_postcheck.py."""
import math
import re
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "views" / "market.py").read_text(encoding="utf-8")
ANNUAL = math.sqrt(252)


def _fmt_vol(daily, decimals=0):
    return f"{daily * ANNUAL:.{decimals}f}% ann. ({daily:.2f}% daily)"


_ns = {"re": re, "common": types.SimpleNamespace(ANNUAL=ANNUAL, fmt_vol=_fmt_vol),
       "VOL_HI_PCT": 95, "VOL_LO_PCT": 5, "ref_name": "the last 3 years", "ref_short": "3-year",
       "corr_name": "the last 3 years"}
exec(SRC[SRC.index("def _shown("):SRC.index("# ─── Post-check of the AI text ───")], _ns)
exec(SRC[SRC.index("# ─── Post-check of the AI text ───"):SRC.index("# No parameter starts with an underscore")], _ns)
level_facts, drop = _ns["level_facts"], _ns["drop_crossed_watch_sentences"]

# TTF on 10 Nov 2026 (GARCH figures from the page that day).
TTF = dict(vol_now=3.436119, vol_spike=5.2, vol_calm=2.0, ref_mean=3.442976, g_now=4.02, g_10d=4.10,
           garch_hi=1.3, garch_lo=0.7)


def test_equal_as_shown_is_equal():
    lines, levels = level_facts(**TTF)
    line = next(l for l in lines if l.startswith("Current 30-day volatility") and "3-year average" in l)
    assert "EQUAL TO" in line and "BELOW" not in line, line
    assert ("vol", TTF["ref_mean"], TTF["ref_mean"]) in levels


def test_different_as_shown_keeps_side():
    lines, _ = level_facts(**dict(TTF, vol_now=3.30))
    line = next(l for l in lines if l.startswith("Current 30-day volatility") and "3-year average" in l)
    assert " is BELOW " in line, line


def test_equal_level_drops_no_watch_sentence():
    _, levels = level_facts(**TTF)
    for s in ["Watch whether 30-day volatility rises above the 3-year average of 55% annualised (3.44% daily).",
              "Watch whether 30-day volatility falls below the 3-year average of 55% annualised (3.44% daily)."]:
        kept, dropped = drop(s, levels)
        assert not dropped, s


def test_correlation_equal_as_shown():
    lines, levels = level_facts(**TTF, corr_v=0.121, corr_lo_v=-0.3, corr_hi_v=0.124, corr_label="carbon")
    line = next(l for l in lines if "top of its" in l)
    assert "EQUAL TO" in line, line
