"""Offline tests of the one 30-day correlation definition (common.rolling_corr), the line written when the latest
day has no 30-day correlation (common.corr_gap_sentence), and the joining of day ranges (common._day_ranges).

Run from the repository root with `python -m pytest tests/`. Synthetic prices, no network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def _walk(idx, seed, sd=0.02):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, sd, len(idx))), index=idx)


def _with_stale(n_before=300, n_stale=60, n_after=13, seed=1):
    """A commodity with a stale run (one price repeated) and a comparison series that trades throughout."""
    idx = pd.bdate_range("2019-01-01", periods=n_before + n_stale + n_after)
    close = _walk(idx, seed)
    close.iloc[n_before:n_before + n_stale] = close.iloc[n_before - 1]
    close.iloc[n_before + n_stale:] = close.iloc[n_before - 1] * np.cumprod(
        1 + np.random.default_rng(seed + 1).normal(0, 0.01, n_after))
    return close.mask(common._stale_mask(close)), _walk(idx, seed + 7)


def test_window_does_not_reach_across_the_commodity_stale_run():
    close, compare = _with_stale(n_after=13)
    corr = common.rolling_corr(close, compare)
    assert np.isnan(corr.iloc[-1])                       # 12 pairs after the gap, 20 needed
    # The old way (rolling over the days with a price only) filled the window from before the gap.
    priced = close.dropna().index
    old = (common.simple_returns(close).reindex(priced).rolling(30, min_periods=20)
           .corr(common.simple_returns(compare).reindex(priced)))
    assert not np.isnan(old.iloc[-1])


def test_figure_after_a_gap_uses_only_pairs_after_it():
    close, compare = _with_stale(n_after=26)             # 25 returns after the gap, inside the last 30 rows
    corr = common.rolling_corr(close, compare)
    r, cr = common.simple_returns(close).iloc[-25:], common.simple_returns(compare).iloc[-25:]
    assert r.notna().all()
    assert corr.iloc[-1] == pytest.approx(r.corr(cr), rel=1e-9)


def test_counts_pairs_not_rows():
    idx = pd.bdate_range("2024-01-01", periods=60)
    close, compare = _walk(idx, 3), _walk(idx, 4)
    # 11 comparison days missing in the last 30 rows: 30 rows give 29 returns of close, 19 pairs
    compare_gap = compare.copy()
    compare_gap.iloc[-20:-10] = np.nan                    # 10 NaN prices make 11 NaN returns
    corr = common.rolling_corr(close, compare_gap)
    assert np.isnan(corr.iloc[-1])
    compare_gap = compare.copy()
    compare_gap.iloc[-19:-10] = np.nan                    # 9 NaN prices make 10 NaN returns: 20 pairs
    assert not np.isnan(common.rolling_corr(close, compare_gap).iloc[-1])


def test_comparison_on_its_own_days():
    # A comparison day the commodity does not have is dropped; its return still spans one day of its own.
    idx = pd.bdate_range("2024-01-01", periods=80)
    close, compare = _walk(idx, 5), _walk(idx, 6)
    close = close.drop(idx[50])
    corr = common.rolling_corr(close, compare)
    assert corr.index.equals(close.index)
    cr = common.simple_returns(compare).reindex(close.index)
    r = common.simple_returns(close)
    assert corr.iloc[-1] == pytest.approx(r.iloc[-30:].corr(cr.iloc[-30:]), rel=1e-9)


def test_same_as_before_without_gaps():
    # With no stale run and no non-positive price the rows are the same as the old ones: same figures.
    idx = pd.bdate_range("2023-01-01", periods=200)
    close, compare = _walk(idx, 8), _walk(idx, 9)
    df = pd.DataFrame({'R': common.simple_returns(close), 'CR': common.simple_returns(compare)})
    old = df['R'].rolling(30, min_periods=20).corr(df['CR'])
    new = common.rolling_corr(close, compare)
    assert (old - new).abs().max() < 1e-12


def test_gap_sentence():
    as_of = pd.Timestamp("2021-09-20")
    core = dict(vol_asof=as_of, corr_gap=dict(
        n=13, need=20, days=30, ongoing=False,
        stale=[("CARB.L", pd.Timestamp("2020-06-11"), pd.Timestamp("2021-08-31"))]))
    assert common.corr_gap_sentence(core) == (
        "No 30-day correlation on 20 Sep 2021: not enough prices since the stale EU Carbon ETC prices of 11 Jun 2020 to "
        "31 Aug 2021 ended (13 of the 20 days with both returns needed in the last 30 trading days).")
    core['corr_gap'].update(ongoing=True, n=0)
    assert common.corr_gap_sentence(core) == (
        "No 30-day correlation on 20 Sep 2021: the EU Carbon ETC prices from 11 Jun 2020 are stale (0 of the 20 days "
        "with both returns needed in the last 30 trading days).")
    assert common.corr_gap_sentence(dict(vol_asof=as_of, corr_gap=None)) == ""
    # The volatility sentence keeps its words.
    vcore = dict(vol_asof=as_of, vol_gap=dict(n=13, need=20, days=30, stale=core['corr_gap']['stale']))
    assert common.vol_gap_sentence(vcore) == (
        "No 30-day volatility on 20 Sep 2021: not enough prices since the stale EU Carbon ETC prices of 11 Jun 2020 to "
        "31 Aug 2021 ended (13 of the 20 returns needed in the last 30 trading days).")


@pytest.mark.parametrize("days, text", [
    (["2026-02-02"], "02 Feb 2026"),
    (["2026-02-02", "2026-03-02"], "02 Feb 2026 and 02 Mar 2026"),
    (["2026-02-02", "2026-03-02", "2026-04-01"], "02 Feb 2026, 02 Mar 2026 and 01 Apr 2026"),
    (["2026-09-08", "2026-09-09", "2026-09-10", "2026-10-19"], "08 Sep to 10 Sep 2026 and 19 Oct 2026"),
])
def test_day_ranges(days, text):
    index = pd.bdate_range("2026-01-01", "2026-12-31")
    assert common._day_ranges(pd.DatetimeIndex(days), index) == text


def _reference(compare_stale_from=None, n=900):
    """three_year_reference on synthetic series; the comparison series goes stale from row `compare_stale_from`."""
    idx = pd.bdate_range("2017-11-20", periods=n)
    close, compare = _walk(idx, 11), _walk(idx, 12)
    if compare_stale_from is not None:
        compare.iloc[compare_stale_from:] = compare.iloc[compare_stale_from - 1]
        compare = compare.mask(common._stale_mask(compare))
    fallback = pd.DataFrame({'Volatility': common.rolling_vol(close), 'Rolling Correlation': np.nan,
                             'Returns': common.simple_returns(close)})
    return common.three_year_reference(close, compare, fallback)


def test_correlation_window_named_when_it_ends_earlier():
    # Like TTF to 30 Jun 2020: the comparison series goes stale 14 rows before the end, so the correlation
    # values stop earlier and are fewer than the volatility values; both windows are under 3 years.
    ref = _reference(compare_stale_from=900 - 14, n=900)
    assert not ref['corr_long'] and not ref['long']
    assert ref['corr_end'] < ref['vol'].index[-1]
    assert ref['corr_own']


def test_correlation_window_not_named_when_it_is_the_volatility_window():
    ref = _reference(n=900)
    assert ref['corr_start'] == ref['start'] and ref['corr_end'] == ref['vol'].index[-1]
    assert len(ref['corr']) == ref['n'] and not ref['corr_own']
