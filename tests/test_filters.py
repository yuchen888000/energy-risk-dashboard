"""Offline test of the headline filter in views/market.py (is_relevant).

Run from the repository root with `python -m pytest tests/`. The page itself is not executed (it downloads
prices and calls Streamlit): the block is cut out of the page source and run on its own, so the test uses
the page's own code and needs no network.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "views" / "market.py").read_text(encoding="utf-8")


def _block(start, end):
    return SRC[SRC.index(start):SRC.index(end)]


_filt = {"re": re}
exec(_block("# ─── Headline relevance ───", "# With an end date in the past, every headline"), _filt)
sys.path.insert(0, str(ROOT))
import common  # noqa: E402  (no download at import)

_COMS = common.build_commodities({"name": "EU Carbon ETC (CARB.L, USD)", "ticker": "CARB.L", "unit": "", "label": ""})
_KW = {k: _filt["_word_pattern"](v["keywords"]) for k, v in _COMS.items()}

FILTER_CASES = [
    ("WTI Crude Oil", "Tank barrel production returns to UK after 20 years", False, False),
    ("Brent Crude Oil", "Tank barrel production returns to UK after 20 years", False, False),
    ("Brent Crude Oil", "Brent rises above $95 a barrel as Hormuz traffic slows", False, True),
    ("WTI Crude Oil", "Prices top $100 per barrel for first time since March", False, True),
    ("TTF Natural Gas", "3 European Gas Stocks Investors Should Watch As Energy Costs Stay High", True, False),
    ("TTF Natural Gas", "European gas prices rise as Norwegian outage cuts supply", True, True),
    ("WTI Crude Oil", "EIA reports surprise draw in US crude inventories at Cushing", False, True),
    ("WTI Crude Oil", "US gas prices at the pump fall, AAA says", False, False),
    ("Brent Crude Oil", "EIA reports surprise draw in US crude inventories", False, False),
    ("Brent Crude Oil", "Shell shares jump as oil majors rally", False, False),
    # "gas stocks" are inventories here, not shares
    ("TTF Natural Gas", "Top 3 reasons European gas stocks are falling faster than expected", True, True),
    ("TTF Natural Gas", "Germany's 5 largest gas stocks sites near 80% full", True, True),
    ("TTF Natural Gas", "Top 5 gas stocks to buy as Europe heads into winter", True, False),
    ("TTF Natural Gas", "5 dividend stocks for a European gas winter", True, False),
]


@pytest.mark.parametrize("name,headline,gas,want", FILTER_CASES)
def test_headline_filter(name, headline, gas, want):
    keep_us = _COMS[name]["ticker"] == "CL=F"
    assert _filt["is_relevant"](headline, _KW[name], gas=gas, keep_us=keep_us) == want, \
        _filt["drop_reason"](headline, _KW[name], gas, keep_us)


# Stock picks with "gas stocks" or "oil stocks" (round 3 of 2026-10-23): a recommendation signal ("stocks for",
# "dividend", "to buy", "to watch", "top N ... stocks") without a storage signal is a stock pick; with a storage
# signal ("full", "storage", "injection" ...) the headline is about inventories and is kept.
STOCK_TIP_CASES = [
    ("Top 5 Gas Stocks for 2026", True),
    ("Top 10 oil stocks for dividends", True),
    ("Top 3 reasons European gas stocks are falling faster than expected", False),
    ("Germany's 5 largest gas stocks sites near 80% full", False),
    ("Top 5 gas stocks to buy as Europe heads into winter", True),
    ("Top 10 gas stocks: storage injections slow as sites near full", False),
    # equity-market stories that mention oil in passing (GDELT, 2026-10-24)
    ("Sharp drops for AI stocks overwhelm markets worldwide as oil prices jump", True),
    ("Wall Street loses ground under pressure from rising oil prices, more swings for Big Tech stocks", True),
    ("Oil prices jump as Hormuz traffic slows", False),
    ("Sensex, Nifty extend losses as crude surges to USD 114, all sectors close in red", True),
    ("Bullion extends losing streak to third day as crude tops $115 per barrel", True),
    ("Defiance Oil Enhanced Options Income ETF (USOY) To Go Ex-Dividend on October 22nd", True),
    ("Indian markets snap 1-week rally following surge in crude, FII outflows", True),
    ("Global Market: European stocks edge lower as Kering slumps, oil prices rise", True),
    ("European gas stocks fall to a five-year low for the time of year", False),
    ("Gas storage stocks close the week 2% lower", False),
]


@pytest.mark.parametrize("headline,want", STOCK_TIP_CASES)
def test_stock_tip(headline, want):
    assert _filt["is_stock_tip"](headline) == want


@pytest.mark.parametrize("name,headline,want", [
    ("TTF Natural Gas", "Top 5 Gas Stocks for 2026", False),
    ("Brent Crude Oil", "Top 10 oil stocks for dividends", False),
    ("WTI Crude Oil", "Top 10 oil stocks for dividends", False),
])
def test_stock_picks_dropped(name, headline, want):
    keep_us = _COMS[name]["ticker"] == "CL=F"
    gas = _COMS[name]["ticker"] == "TTF=F"
    assert _filt["is_relevant"](headline, _KW[name], gas=gas, keep_us=keep_us) == want
    assert _filt["drop_reason"](headline, _KW[name], gas, keep_us) == "share tip or stock story"
