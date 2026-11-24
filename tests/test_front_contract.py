"""The price header and the Power page take the front month from the exchange expiry rule, not from the day
Yahoo's continuous series moves to the next contract; the Market country table shows n/a for a missing value;
headlines from blocked sites are left out."""
import os

import numpy as np
import pandas as pd

import common
import news_data

HERE = os.path.dirname(__file__)


def _batch(days, cont, contracts):
    df = pd.DataFrame(contracts, index=days)
    df.insert(0, "BZ=F", cont)
    return df


def test_header_contract_when_yahoo_rolled_early(monkeypatch):
    # 24 Nov 2026: BZ=F already shows Feb 27 (BZG27), while Jan 27 (BZF27) trades until 30 Nov.
    days = pd.bdate_range("2026-11-17", "2026-11-24")
    jan = np.linspace(93.1, 89.4, len(days))
    feb = jan - 2.6
    cont = jan.copy()
    cont[-1] = feb[-1]
    batch = _batch(days, cont, {"BZF27.NYM": jan, "BZG27.NYM": feb})
    monkeypatch.setattr(common, "shared_download", lambda t, s: batch)
    hc = common.header_contract("BZ=F", days[0], days[-1])
    assert hc["symbol"] == "BZF27.NYM"
    assert hc["delivery"] == pd.Timestamp("2027-01-01").date()
    assert hc["tracked"] == "BZG27.NYM"
    assert hc["expiry"].normalize() == pd.Timestamp("2026-11-30")
    assert float(hc["own"].iloc[-1]) == jan[-1]


def test_header_contract_when_yahoo_tracks_the_front(monkeypatch):
    days = pd.bdate_range("2026-11-10", "2026-11-17")
    jan = np.linspace(90.0, 93.1, len(days))
    batch = _batch(days, jan, {"BZF27.NYM": jan, "BZG27.NYM": jan - 2.6})
    monkeypatch.setattr(common, "shared_download", lambda t, s: batch)
    hc = common.header_contract("BZ=F", days[0], days[-1])
    assert hc["symbol"] == hc["tracked"] == "BZF27.NYM"


def test_header_contract_none_for_carbon():
    assert common.header_contract("CARB.L", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-24")) is None


def test_front_by_rule_on_24_nov_2026():
    d = pd.Timestamp("2026-11-24")
    assert common.front_by_rule("BZ", d) == pd.Timestamp("2027-01-01").date()
    assert common.front_by_rule("TTF", d) == pd.Timestamp("2026-12-01").date()
    assert common.front_by_rule("TTF", pd.Timestamp("2026-11-30")) == pd.Timestamp("2027-01-01").date()


def test_header_uses_the_rule_and_the_contracts_own_prices():
    src = open(os.path.join(HERE, "..", "common.py"), encoding="utf-8").read()
    body = src[src.index("def show_price_header("):src.index("def _raw_close(")]
    assert "header_contract(ticker, start, last_day)" in body
    assert 'price, own_used = hc["own"], True' in body
    # The hourly bars are the continuous series': not compared with another contract's price.
    assert "None if own_used else last_trade_on(ticker, last_day)" in body


def test_power_front_month_from_the_rule():
    src = open(os.path.join(HERE, "..", "views", "power.py"), encoding="utf-8").read()
    assert 'common.front_by_rule("TTF", ttf_last_date)' in src
    assert 'common.front_by_rule("TTF", latest["TTF date"])' in src
    assert 'hist = (_c if (_switch or _off) else ttf).tail(40)' in src


def test_market_country_table_shows_na():
    src = open(os.path.join(HERE, "..", "views", "market.py"), encoding="utf-8").read()
    assert 'placeholder="n/a"' in src


def test_blocked_domains():
    assert news_data.is_blocked_domain("naturalnews.com")
    assert news_data.is_blocked_domain("www.NaturalNews.com")
    assert not news_data.is_blocked_domain("reuters.com")
    assert not news_data.is_blocked_domain("notnaturalnews.com")
    assert not news_data.is_blocked_domain("")


def test_streamlit_minimum_version():
    # st.dataframe(placeholder=...) needs 1.52 and app.py's st.Page(visibility=...) needs 1.55.
    req = open(os.path.join(HERE, "..", "requirements.txt"), encoding="utf-8").read()
    assert "streamlit>=1.55" in req
