"""Offline test of the post-check of the AI text in views/market.py (drop_crossed_watch_sentences).

Run from the repository root with `python -m pytest tests/`. The page itself is not executed (it downloads
prices and calls Streamlit): the block is cut out of the page source and run on its own, so the test uses
the page's own code and needs no network.
"""
import math
import re
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "views" / "market.py").read_text(encoding="utf-8")


def _block(start, end):
    return SRC[SRC.index(start):SRC.index(end)]


_post = {"re": re, "common": types.SimpleNamespace(ANNUAL=math.sqrt(252))}
exec(_block("# ─── Post-check of the AI text ───", "# No parameter starts with an underscore"), _post)
drop = _post["drop_crossed_watch_sentences"]

# Levels as the page passes them: (subject, level, current value), volatility as daily %.
# Current 30-day volatility 3.19% daily: the unusual-calm threshold 1.74 and the 3-year mean 2.45 are already
# crossed upward, the spike threshold 6.49 is not reached. Correlation 0.39: its range is -0.43 to 0.79, and
# 0.30 (a level a model might name) is already crossed upward.
LEVELS = [("vol", 6.49, 3.19), ("vol", 1.74, 3.19), ("vol", 2.45, 3.19),
          ("garch", 4.54, 3.05), ("garch", 2.45, 3.05),
          ("corr", 0.79, 0.39), ("corr", -0.43, 0.39), ("corr", 0.30, 0.39)]
# The same correlation range with the current correlation below its bottom.
LEVELS_CORR_LOW = [("corr", 0.79, -0.50), ("corr", -0.43, -0.50)]
# WTI to 14 Nov 2025: correlation with carbon -0.26, its range -0.21 to 0.42.
LEVELS_WTI_1114 = [("vol", 2.6925, 1.83), ("vol", 1.2844, 1.83), ("corr", 0.42, -0.26), ("corr", -0.21, -0.26)]

MUST_DROP = [
    # volatility (round of 2026-10-23)
    ("Watch whether volatility rises above 28% annualised over the next ten days.", LEVELS),
    ("Watch whether volatility rises above 28% ann. (1.74% daily) over the next ten days.", LEVELS),
    ("Over the next ten days, watch whether 30-day volatility climbs above 39% annualised.", LEVELS),
    ("The coming days will show whether volatility rises above 39% annualised (2.45% daily).", LEVELS),
    # a number at the end of a sentence
    ("Watch whether the correlation rises above 0.30.", LEVELS),
    ("Watch whether the correlation rises above 0.30 in the next ten days.", LEVELS),
    ("Watch whether volatility rises above 28% annualised.", LEVELS),
    # a minus sign is read
    ("Watch whether the correlation rises above -0.43 in the next ten days.", LEVELS),
    ("Watch whether the correlation rises above −0.43 in the next ten days.", LEVELS),
    ("Watch whether the correlation falls below -0.43 in the next ten days.", LEVELS_CORR_LOW),
    ("Watch whether the correlation with carbon falls below -0.21.", LEVELS_WTI_1114),
    # more crossing verbs (round 3 of 2026-10-23): upward exceeds, tops, surpasses, breaks above;
    # downward drops below, falls through
    ("Watch whether volatility exceeds 28% annualised over the next ten days.", LEVELS),
    ("Over the next ten days, watch whether volatility tops 39% annualised (2.45% daily).", LEVELS),
    ("Watch whether 30-day volatility surpasses 39% annualised.", LEVELS),
    ("Watch whether the correlation breaks above 0.30 in the coming days.", LEVELS),
    ("Watch whether the correlation drops below -0.43 over the next ten days.", LEVELS_CORR_LOW),
    ("Watch whether the correlation with carbon falls through -0.21.", LEVELS_WTI_1114),
    # round 4 of 2026-10-23: "should" is a forward-looking cue
    ("Should volatility rise above 28% annualised, the calm reading ends.", LEVELS),
    ("Should the correlation climb above 0.30, the two markets move more closely together.", LEVELS),
    # the subject is the one nearest the verb, not the first one named
    ("Watch whether the correlation with Brent stays near 0.39 while volatility rises above 28% annualised.",
     LEVELS),
    ("Watch whether volatility stays near 51% annualised while the correlation rises above 0.30.", LEVELS),
    # a minus sign written as U+2013 (an en dash) is read as a minus sign
    ("Watch whether the correlation rises above \u20130.43 in the next ten days.", LEVELS),
    ("Watch whether the correlation with carbon falls below \u20130.21.", LEVELS_WTI_1114),
]
# The correlation above the top of its range.
LEVELS_CORR_HIGH = [("corr", 0.79, 0.85), ("corr", -0.43, 0.85)]
MUST_KEEP = [
    ("Watch whether volatility rises above 103% annualised (6.49% daily) over the next ten days.", LEVELS),
    ("Watch whether volatility falls back below 28% annualised (1.74% daily) over the next ten days.", LEVELS),
    ("Volatility is 51% ann. (3.19% daily), close to the 3-year average.", LEVELS),
    ("Watch whether the correlation rises above 0.79 over the next ten trading days.", LEVELS),
    ("Watch whether the correlation rises above 0.79.", LEVELS),
    ("Watch whether the correlation falls back below 0.30.", LEVELS),
    ("Watch whether the correlation falls below -0.43 in the next ten days.", LEVELS),
    # current -0.50 is below -0.43, so rising above it is a move still to come
    ("Watch whether the correlation rises above -0.43 in the next ten days.", LEVELS_CORR_LOW),
    ("Watch whether the correlation with carbon rises back above -0.21.", LEVELS_WTI_1114),
    # no forward-looking cue: a statement, not a watch
    ("The correlation of 0.39 is above 0.30.", LEVELS),
    # the same verbs where the move is still to come
    ("Watch whether volatility exceeds 103% annualised (6.49% daily) over the next ten days.", LEVELS),
    ("Watch whether the correlation tops 0.79 in the next ten days.", LEVELS),
    ("Watch whether the correlation surpasses 0.79.", LEVELS),
    ("Watch whether the correlation breaks above 0.79.", LEVELS),
    ("Watch whether the correlation drops below -0.43 over the next ten days.", LEVELS),
    ("Watch whether the correlation falls through 0.30 in the coming days.", LEVELS),
    # "top" as a noun is not a crossing verb
    ("Watch whether the correlation, now above the top of its range of 0.79, falls back.", LEVELS_CORR_HIGH),
    # a year range is not a negative number
    ("Watch whether volatility, low against the 2021-22 crisis, stays below 103% annualised.", LEVELS),
    # round 4 of 2026-10-23: the same three rules where the move is still to come
    ("Should volatility rise above 103% annualised (6.49% daily), the spike check fires.", LEVELS),
    ("Should volatility fall back below 28% annualised, the calm check fires.", LEVELS),
    ("Watch whether volatility stays near 51% annualised while the correlation rises above 0.79.", LEVELS),
    # nearest subject: the GARCH forecast of volatility is the forecast, whose 4.54% daily level is not reached
    ("Watch whether the GARCH forecast of volatility rises above 4.54% daily.", LEVELS),
    ("Watch whether the correlation rises above \u20130.43 in the next ten days.", LEVELS_CORR_LOW),
    ("Watch whether the correlation falls below \u20130.43 in the next ten days.", LEVELS),
    # an en dash between two years or two numbers is not a minus sign
    ("Watch whether volatility, low against the 2021\u201322 crisis, stays below 103% annualised.", LEVELS),
]


@pytest.mark.parametrize("sentence,levels", MUST_DROP)
def test_drops_watch_for_a_level_already_passed(sentence, levels):
    kept, dropped = drop(sentence, levels)
    assert kept == "" and len(dropped) == 1


@pytest.mark.parametrize("sentence,levels", MUST_KEEP)
def test_keeps_other_sentences(sentence, levels):
    kept, dropped = drop(sentence, levels)
    assert dropped == []


def test_whole_text_drops_only_the_crossed_sentence():
    text = ("Volatility is 51% ann. (3.19% daily), close to the 3-year average. The GARCH forecast is lower than "
            "today. Watch whether the correlation rises above 0.30.")
    kept, dropped = drop(text, LEVELS)
    assert dropped == ["Watch whether the correlation rises above 0.30."]
    assert kept.startswith("Volatility is 51% annualised (3.19% daily), close to the 3-year average.")
    assert kept.endswith("lower than today.")


def test_en_dash_minus_reads_as_negative():
    # The en dash case the audit found: read as +0.43 it would not match the level -0.43.
    kept, dropped = drop("Watch whether the correlation rises above \u20130.43.", LEVELS)
    assert kept == "" and len(dropped) == 1


def test_nearest_subject_over_first_named():
    s = "Watch whether the correlation with Brent stays near 0.39 while volatility rises above 28% annualised."
    assert _post["_subject"](s[:s.index("rises")]) == "vol"
    assert _post["_subject"]("Watch whether the GARCH forecast of volatility ") == "garch"
    assert _post["_subject"]("Watch whether the correlation ") == "corr"
