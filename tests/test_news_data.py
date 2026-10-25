"""Offline tests of news_data (GDELT headlines): query building, parsing, rate limit and error handling.
No network: requests.get, sleep and the clock are replaced."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import news_data  # noqa: E402


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


def _fake(responses, calls):
    def get(url, params=None, timeout=None):
        calls.append(params)
        return responses.pop(0)
    return get


def _run(responses):
    calls, sleeps = [], []
    news_data._last_call[0] = -1e9
    out = news_data.fetch('"natural gas" Europe sourcelang:english', 7, get=_fake(list(responses), calls),
                          sleep=sleeps.append, clock=lambda: 0.0)
    return out, calls, sleeps


ARTS = ('{"articles": [ {"url": "https://example.com/a(1)", "title": "Europe&#39;s gas  prices rise\\n", '
        '"seendate": "20261023T101500Z", "domain": "example.com", "sourcecountry": "United Kingdom"}, '
        '{"url": "https://x.org/b", "title": "", "seendate": "20261023T090000Z", "domain": "x.org"}, '
        '{"url": "https://y.org/c", "title": "No date", "seendate": "bad", "domain": "y.org"} ] }')


def test_build_query_quotes_phrases_only():
    assert news_data.build_query("natural gas", "Europe", "price") == '"natural gas" Europe price sourcelang:english'


def test_parse_articles():
    out, calls, _ = _run([_Resp(200, ARTS)])
    assert len(out) == 1
    a = out[0]
    assert a["title"] == "Europe's gas prices rise"
    assert a["domain"] == "example.com" and a["country"] == "United Kingdom"
    assert str(a["seen"]) == "2026-10-23 10:15:00+00:00"
    assert calls[0]["timespan"] == "7d" and calls[0]["maxrecords"] == "250" and calls[0]["sort"] == "datedesc"


def test_control_characters_in_json():
    raw = ARTS.replace("\\n", "\n")          # a raw newline inside a string: strict JSON rejects it
    out, _, _ = _run([_Resp(200, raw)])
    assert out[0]["title"] == "Europe's gas prices rise"


def test_empty_answer_is_no_article():
    assert _run([_Resp(200, "")])[0] == []
    assert _run([_Resp(200, "{}")])[0] == []


def test_rate_limit_retried_twice():
    out, calls, sleeps = _run([_Resp(429, "Please limit requests"), _Resp(200, ARTS)])
    assert len(out) == 1 and len(calls) == 2 and news_data.RETRY_WAITS_S[0] in sleeps
    with pytest.raises(news_data.NewsError, match="429"):
        _run([_Resp(429, "x")] * 3)


@pytest.mark.parametrize("raw,clean", [
    ("Northeast Asian LNG Gauge Reaches $30 . 01 per Million British Thermal Units",
     "Northeast Asian LNG Gauge Reaches $30.01 per Million British Thermal Units"),
    ("Japan Electricity Prices Surge 25 % as LNG and Coal Costs Soar", "Japan Electricity Prices Surge 25% as LNG and Coal Costs Soar"),
    ("World Pays $29 for Natgas , Henry Hub $3", "World Pays $29 for Natgas, Henry Hub $3"),
    ("Moscow and Beijing could seal major gas pipeline deal by year - end", "Moscow and Beijing could seal major gas pipeline deal by year-end"),
    ("Bangladesh Gas Pact : Algeria Offers LNG and Urea", "Bangladesh Gas Pact: Algeria Offers LNG and Urea"),
    ("EU weakens call for ban on Arctic oil and gas drilling | Hellenic Shipping News Worldwide",
     "EU weakens call for ban on Arctic oil and gas drilling"),
    ("Baystreet . ca - Japan Electricity Prices Surge 25 % as LNG and Coal Costs Soar",
     "Japan Electricity Prices Surge 25% as LNG and Coal Costs Soar"),
    ("Gas prices rise - analysts warn of shortages", "Gas prices rise - analysts warn of shortages"),
    ("BofA sees EU carbon prices rising to USD 112 / tonne amid Iran war - driven energy shock",
     "BofA sees EU carbon prices rising to USD 112/tonne amid Iran war-driven energy shock"),
    ("China green finance helps drive global low - carbon transition", "China green finance helps drive global low-carbon transition"),
    ("Liquid - Cooled AI Halls Push Piping Demand", "Liquid-Cooled AI Halls Push Piping Demand"),
    ("Germany adds gas - fired capacity as coal exits", "Germany adds gas-fired capacity as coal exits"),
    ("Saudi Arabia crude exports rise 7.9% in August to 4 - month high", "Saudi Arabia crude exports rise 7.9% in August to 4-month high"),
    ("The Hidden $40 - Per - Barrel Cost Driving Oil Prices Higher", "The Hidden $40-Per-Barrel Cost Driving Oil Prices Higher"),
    ("U . S . Crude Benchmark Advances 5.02% to Reach $102 Per Barrel", "U.S. Crude Benchmark Advances 5.02% to Reach $102 Per Barrel"),
    ("Trump invokes Cold War - era Defense Production Act for oil production", "Trump invokes Cold War-era Defense Production Act for oil production"),
    ("U . S . naval blockade threatens to end Iran oil revenues by December",
     "U.S. naval blockade threatens to end Iran oil revenues by December"),
    ("FPIs withdraw Rs 48 , 056 cr from equities in Oct so far amid elevated crude oil prices",
     "FPIs withdraw Rs 48,056 cr from equities in Oct so far amid elevated crude oil prices"),
    ("Bitcoin Remains Stuck Below $85 , 000 . Here Why", "Bitcoin Remains Stuck Below $85,000. Here Why"),
    ("UK Oil & Gas ( LON : UKOG ) Share Price Crosses Above Fifty Day Moving Average",
     "UK Oil & Gas (LON: UKOG) Share Price Crosses Above Fifty Day Moving Average"),
    ("Crude oil near $120 , to intensify headwinds", "Crude oil near $120, to intensify headwinds"),
    ("Brent rises in 2025 , 300 traders say", "Brent rises in 2025, 300 traders say"),
    ("Oil prices threaten to undermine fuel levy relief .", "Oil prices threaten to undermine fuel levy relief."),
])
def test_clean_title(raw, clean):
    assert news_data._clean_title(raw) == clean


def test_text_error_and_http_error():
    with pytest.raises(news_data.NewsError, match="phrase is too short"):
        _run([_Resp(200, "The specified phrase is too short.")])
    with pytest.raises(news_data.NewsError, match="HTTP 503"):
        _run([_Resp(503, "")])


def test_gap_between_requests():
    calls, sleeps = [], []
    news_data._last_call[0] = 0.0
    news_data.fetch("q", 7, get=_fake([_Resp(200, "{}")], calls), sleep=sleeps.append, clock=lambda: 2.0)
    assert sleeps and abs(sleeps[0] - (news_data.MIN_GAP_S - 2.0)) < 1e-9
