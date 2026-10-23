"""Offline tests of the one 30-day volatility definition (common.rolling_vol), the line written when the latest
day has no 30-day volatility (common.vol_gap_sentence, common.vol_summary), the AI wording before the strikes of
28 Feb 2026, and the Market page's check window (common.three_year_reference).

Run from the repository root with `python -m pytest tests/`. Synthetic prices, no network.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def _series(n_before=300, n_stale=60, n_after=13, start="2019-01-01", seed=1):
    """Business-day closes: a random walk, then a stale run (one price repeated), then new prices."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_before + n_stale + n_after)
    before = 100 * np.cumprod(1 + rng.normal(0, 0.02, n_before))
    stale = np.full(n_stale, before[-1])
    after = before[-1] * np.cumprod(1 + rng.normal(0, 0.01, n_after))
    close = pd.Series(np.concatenate([before, stale, after]), index=idx)
    return close.mask(common._stale_mask(close))


def test_window_does_not_reach_across_a_stale_run():
    close = _series(n_after=13)
    vol = common.rolling_vol(close)
    # 13 prices after the gap give 12 returns (the first one after the gap has none): no figure.
    assert np.isnan(vol.iloc[-1])
    rets = common.simple_returns(close)
    assert rets.iloc[-30:].notna().sum() == 12


def test_figure_after_a_gap_uses_only_returns_after_it():
    close = _series(n_after=26)                    # 25 returns after the gap, inside the last 30 rows
    vol = common.rolling_vol(close)
    rets = common.simple_returns(close)
    after = rets.iloc[-25:]
    assert after.notna().all()
    assert vol.iloc[-1] == pytest.approx(after.std() * 100, rel=1e-12)


def test_same_figure_as_the_percentile_history():
    close = _series(n_after=40)
    hist = common.rolling_vol(close).dropna()
    on_priced_days = common.rolling_vol(close).reindex(close.dropna().index).dropna()
    common_days = on_priced_days.index
    assert (hist.reindex(common_days) - on_priced_days).abs().max() == 0


def _core(close, latest, ticker="CARB.L"):
    df = pd.DataFrame({'Price': close, 'Returns': common.simple_returns(close)}).dropna(subset=['Price'])
    as_of = df.index[-1]
    window = common.simple_returns(close).loc[:as_of].tail(common.VOL_DAYS)
    mask = common._stale_mask(close.ffill())
    runs = [r for r in common._stale_runs([(ticker, mask)], mask.index[0], as_of) if r[2] >= window.index[0]]
    gap = None if latest is not None else dict(n=int(window.notna().sum()), need=common.VOL_MIN,
                                               days=len(window), stale=runs)
    return dict(ticker=ticker, latest_vol=latest, vol_asof=as_of, vol_window=window, vol_gap=gap, vol_ctx=None,
                rolls=None, df_analysis=df)


def test_no_latest_figure_says_why():
    close = _series(n_after=13)
    core = _core(close, None)
    vs = common.vol_summary(core, name="EU Carbon", with_roll=True, past_end=True)
    assert vs['headline'] == "EU Carbon 30-day volatility: not available"
    assert vs['tile'] == "n/a"
    stale_first = close.index[300]          # first repeated day (the first day of the run is a real price)
    stale_last = close.index[300 + 60 - 1]
    assert vs['headline2'] == (
        f"No 30-day volatility on {close.index[-1]:%d %b %Y}: not enough prices since the stale CARB.L prices of "
        f"{stale_first:%d %b %Y} to {stale_last:%d %b %Y} ended (12 of the 20 returns needed in the last 30 "
        "trading days).")
    assert "%" not in vs['headline'] + vs['headline2'] + vs['ai']


def _ctx(as_of, ratio, complete):
    return dict(p_all=40.0, top_all=False, all_start=pd.Timestamp("2018-01-02"), prewar_incomplete=not complete,
                prewar_median=2.0 if ratio else None, prewar_ratio=ratio)


@pytest.mark.parametrize("as_of, ratio, complete", [
    ("2021-09-20", None, False),     # quarter not complete: the AI input leaves it out
    ("2026-01-15", 1.2, True),       # quarter complete, but before the strikes: named by its months only
    ("2026-02-27", 1.2, True),
])
def test_ai_input_names_no_later_event(as_of, ratio, complete):
    close = _series(n_after=40, start="2018-01-01")
    core = _core(close, 2.5, ticker="TTF=F")
    core.update(vol_asof=pd.Timestamp(as_of), vol_ctx=_ctx(as_of, ratio, complete))
    vs = common.vol_summary(core, name="TTF", past_end=True)
    for word in ("Iran", "strikes", "war", "pre-war", "28 Feb 2026"):
        assert not re.search(rf"\b{re.escape(word)}\b", vs['ai']), (word, vs['ai'])
    if ratio:
        assert "Oct to Dec 2025 level" in vs['ai'] and vs['prewar_ai'] == "the Oct to Dec 2025 level"
    else:
        assert "Oct to Dec 2025" not in vs['ai'] and vs['prewar_ai'] == ""
    # the page names no later event either (readability audit 2026-10-23, R-B1 and M-B5)
    for word in ("Iran", "strikes", "war", "pre-war", "28 Feb 2026"):
        assert not re.search(rf"\b{re.escape(word)}\b", vs['headline2'] + " " + vs['detail']), word


def test_ai_input_after_the_strikes_is_the_page_text():
    close = _series(n_after=40, start="2018-01-01")
    core = _core(close, 2.5, ticker="TTF=F")
    core.update(vol_asof=pd.Timestamp("2026-10-22"), vol_ctx=_ctx("2026-10-22", 1.85, True))
    vs = common.vol_summary(core, name="TTF", past_end=False)
    assert vs['ai'] == f"{vs['headline']}; {vs['headline2']}. {vs['detail']}"
    assert vs['prewar_ai'] == "the pre-war level (Oct to Dec 2025)"


def test_check_window_short_history_and_days_without_volatility():
    close = _series(n_before=400, n_stale=60, n_after=40)
    compare = close * 0 + 1 + np.arange(len(close)) * 1e-3
    df_fallback = pd.DataFrame({'Volatility': common.rolling_vol(close), 'Rolling Correlation': np.nan,
                                'Returns': common.simple_returns(close)})
    ref = common.three_year_reference(close, compare, df_fallback)
    assert not ref['fallback'] and ref['short'] and ref['n'] < common.CHECK_WINDOW
    # the window ends on the last day with a price, also when that day's figure is missing
    assert ref['end'] == close.index[-1]
    # days after the gap with no 30-day volatility are days with a price but not volatility values
    assert ref['rows'] > ref['n']
    assert ref['vol'].index.isin(ref['days']).all()


@pytest.mark.parametrize("past_end, today, whose", [
    # the last 30-day volatility is of 22 Oct and today is 23 Oct (TTF and carbon most mornings): dated
    (False, "2026-10-23", "The percentile ranks the latest figure (22 Oct) among"),
    # the last figure is of today
    (False, "2026-10-22", "The percentile ranks today's figure among"),
    # a past end date: the span already names the date
    (True, "2026-10-23", "The percentile ranks the latest figure among"),
])
def test_percentile_sentence_names_whose_figure(past_end, today, whose):
    close = _series(n_after=40, start="2018-01-01")
    core = _core(close, 2.5, ticker="TTF=F")
    core.update(vol_asof=pd.Timestamp("2026-10-22"), vol_ctx=_ctx("2026-10-22", 1.85, True))
    vs = common.vol_summary(core, name="TTF", past_end=past_end, today=pd.Timestamp(today))
    assert whose in vs['detail'] and whose in vs['ai']
