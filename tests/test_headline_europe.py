"""Offline tests of the TTF European-subject rule and the duplicate key (views/market.py), and of the dash before
a source word in news_data._clean_title. The page is not run: the blocks are cut out of its source."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "views" / "market.py").read_text(encoding="utf-8")
_ns = {"re": re}
exec(SRC[SRC.index("# ─── Headline relevance ───"):SRC.index("# With an end date in the past, every headline")], _ns)
exec(SRC[SRC.index("def dedup_key"):SRC.index("# GDELT asks for a citation")], _ns)
sys.path.insert(0, str(ROOT))
import news_data  # noqa: E402


@pytest.mark.parametrize("headline,keep", [
    ("Dutch turn to electric heaters amid highest natural gas prices in years", True),
    ("UK Rules Out Winter Energy Shortages as Gas Prices Soar", True),
    ("Europe Braces for a Winter Energy Crisis as Gas Reserves Run Low", True),
    ("Equinor lifts Troll output ahead of winter", True),
    ("TTF front month tops EUR 90", True),
    ("LNG Canada Doubles Capacity as Asia Looks Beyond the Gulf", False),
    ("Turkmenistan targets 80 billion cubic meters of gas production in 2026", False),
    ("B.C. Election: Questions raised about Conservative plan to triple LNG production", False),
])
def test_european_gas(headline, keep):
    assert _ns["is_european_gas"](headline) is keep


def test_dedup_label_before_colon():
    k = _ns["dedup_key"]
    assert (k("B.C. Election Analysis: Questions raised about Conservative plan to triple LNG production")
            == k("B.C. Election: Questions raised about Conservative plan to triple LNG production"))
    # A short part after the colon is not enough to call two headlines the same story.
    assert k("Gas Pact: Algeria Offers LNG") != k("Oil Pact: Algeria Offers LNG")


@pytest.mark.parametrize("raw,clean", [
    ("Methane cuts could open EU market to Turkmen gas - expert",
     "Methane cuts could open EU market to Turkmen gas - expert"),
    ("Oil climbs - analysts see tighter supply", "Oil climbs - analysts see tighter supply"),
    ("Germany adds gas - fired capacity as coal exits", "Germany adds gas-fired capacity as coal exits"),
])
def test_dash_before_source_word(raw, clean):
    assert news_data._clean_title(raw) == clean
