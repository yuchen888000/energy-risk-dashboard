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
    assert 'price, own_used, shown_sym = hc["own"], True, front' in body
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


# ─── Behaviour of the header itself (independent check of 24 Nov 2026 evening) ───

class _Col:
    def __init__(self, out):
        self.out = out

    def metric(self, label, value, *a, **k):
        self.out.append((label, value))


def _run_header(monkeypatch, batch, last_trade=None):
    import streamlit as st
    metrics, captions = [], []
    monkeypatch.setattr(common, "shared_download", lambda t, s: batch)
    monkeypatch.setattr(common, "last_trade_on", lambda t, d: last_trade)
    monkeypatch.setattr(common, "last_value_note", lambda *a, **k: "")
    monkeypatch.setattr(st, "columns", lambda n: [_Col(metrics) for _ in range(n)])
    monkeypatch.setattr(st, "caption", lambda body, *a, **k: captions.append(body))
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    core = {"df_analysis": pd.DataFrame({"Price": batch["BZ=F"]}), "rolls": None, "var_99": None}
    common.show_price_header(core, "BZ=F", "Brent Crude Oil", "$/barrel", batch.index[-1].date(), batch.index[0])
    return metrics, " ".join(captions)


def test_header_on_a_past_date_without_the_contracts(monkeypatch):
    # A past end date: Yahoo no longer lists the contracts of that time. The page used to crash here.
    days = pd.bdate_range("2025-11-20", "2025-11-28")
    batch = pd.DataFrame({"BZ=F": np.linspace(62.0, 63.2, len(days))}, index=days)
    monkeypatch.setattr(common, "shared_download", lambda t, s: batch)
    hc = common.header_contract("BZ=F", days[0], days[-1])
    assert hc["own"].empty and hc["tracked"] is None
    metrics, _ = _run_header(monkeypatch, batch)
    assert metrics[0] == ("Latest price (28 Nov)", "63.20 $/barrel")   # no contract named when none is known


def test_header_uses_the_front_contract_when_yahoo_rolled_early(monkeypatch):
    days = pd.bdate_range("2026-11-17", "2026-11-24")
    jan = np.linspace(93.14, 91.03, len(days))
    feb = jan - 2.6
    cont = jan.copy()
    cont[-1] = feb[-1]
    batch = _batch(days, cont, {"BZF27.NYM": jan, "BZG27.NYM": feb})
    metrics, cap = _run_header(monkeypatch, batch)
    assert metrics[0] == ("Latest price, Jan 27 contract (24 Nov)", "91.03 $/barrel")
    assert metrics[1][1] == f"{(jan[-1] / jan[-2] - 1) * 100:+.2f}%"
    assert "shows the Feb 27 contract" in cap and "Jan 27 contract's own prices" in cap


def test_header_no_match_uses_the_front_contract_not_a_wrong_label(monkeypatch):
    # The continuous series' last value matches no contract (a live price between rows): the figures are the
    # front contract's own, and no other contract's price is labelled with its name.
    days = pd.bdate_range("2026-11-17", "2026-11-24")
    jan = np.linspace(93.14, 91.03, len(days))
    cont = jan.copy()
    cont[-1] = 90.0
    batch = _batch(days, cont, {"BZF27.NYM": jan, "BZG27.NYM": jan - 2.6})
    metrics, cap = _run_header(monkeypatch, batch)
    assert metrics[0] == ("Latest price, Jan 27 contract (24 Nov)", "91.03 $/barrel")


def test_last_trade_of_another_contract_is_not_shown(monkeypatch):
    # 23 Nov 2026: the hourly bars already show Feb 27 (86.72) while the daily close is Jan 27 (89.37).
    days = pd.bdate_range("2026-11-16", "2026-11-23")
    jan = np.linspace(93.14, 89.37, len(days))
    batch = _batch(days, jan, {"BZF27.NYM": jan, "BZG27.NYM": jan - 2.69})
    _, cap = _run_header(monkeypatch, batch, last_trade=(pd.Timestamp("2026-11-23 22:00"), 86.72))
    assert "Last trade" not in cap
    _, cap = _run_header(monkeypatch, batch, last_trade=(pd.Timestamp("2026-11-23 22:00"), 89.30))
    assert "Last trade on Yahoo Finance: 89.30" in cap
