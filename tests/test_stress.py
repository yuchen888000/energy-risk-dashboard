"""Offline tests of the Risk page's stress tests (spec of 23 Nov 2026, third version): the parametric comparison of
correlations, the FRTB scenarios and the nearest-correlation repair, the diversification alert, the euro
conversion, the fixed-quantity scenario replay, the 10-day moves and the borrowed severity, the conditional
moves, the reverse stress test and the page text.

Run from the repository root with `python -m pytest tests/`. Synthetic data with answers worked out by hand, no
network. The dates of the 2008 and 2020 scenario windows were checked against Yahoo Finance's BZ=F on
23 Nov 2026 with common.scenario_end (2008-09-15 to 2008-10-27, 2020-03-06 to 2020-03-31);
test_scenario_windows_follow_the_rule repeats that check when RUN_NETWORK_TESTS=1.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

Z = common.Z95


# ─── Parametric VaR, no-diversification VaR and the pair attribution ───
def test_param_var_identity_and_ones():
    w, s = np.array([10e6, 3e6]), np.array([0.02, 0.01])
    assert common.param_var(w, s, np.eye(2)) == pytest.approx(Z * np.sqrt((w ** 2 * s ** 2).sum()))
    assert common.param_var(w, s, np.ones((2, 2))) == pytest.approx(common.no_div_var(w, s))


def test_no_div_var_with_a_short_needs_the_sign_matrix():
    w, s = np.array([10e6, -5e6, 3e6]), np.array([0.05, 0.035, 0.018])
    sign = np.sign(np.outer(w, w))
    assert common.param_var(w, s, sign) == pytest.approx(common.no_div_var(w, s))
    assert common.param_var(w, s, np.ones((3, 3))) < common.no_div_var(w, s) - 1e5


def test_pair_deltas_sum_to_the_change_in_variance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(400, 4))
    c_old = np.corrcoef(a[:250], rowvar=False)
    c_new = np.corrcoef(a[-30:], rowvar=False)
    w, s = np.array([4e6, -2e6, 1e6, 3e6]), np.array([0.02, 0.03, 0.01, 0.015])
    d = common.pair_deltas(w, s, c_new, c_old)
    lhs = sum(d.values())
    rhs = (common.param_var(w, s, c_new) ** 2 - common.param_var(w, s, c_old) ** 2) / Z ** 2
    assert lhs == pytest.approx(rhs, rel=1e-9)


# ─── FRTB scenarios and the nearest correlation matrix ───
def test_frtb_scenarios_by_hand():
    c = np.array([[1, 0.9], [0.9, 1.0]])
    hi, _ = common.frtb_corr(c, "high")
    lo, _ = common.frtb_corr(c, "low")
    assert hi[0, 1] == pytest.approx(1.0)
    assert lo[0, 1] == pytest.approx(max(0.8, 0.675))
    neg = np.array([[1, -0.9], [-0.9, 1.0]])
    hi_n, _ = common.frtb_corr(neg, "high")
    assert hi_n[0, 1] == pytest.approx(-1.0)          # 1.25 x -0.9 = -1.125, floored at -1
    lo_n, _ = common.frtb_corr(neg, "low")
    assert lo_n[0, 1] == pytest.approx(max(2 * -0.9 - 1, 0.75 * -0.9))


def test_high_scenario_repair_gives_a_correlation_matrix():
    c = np.array([[1, 0.7, 0.7], [0.7, 1, 0.0], [0.7, 0.0, 1]])
    assert np.linalg.eigvalsh(c).min() > 0
    hi, repaired = common.frtb_corr(c, "high")
    assert repaired
    assert np.linalg.eigvalsh(hi).min() >= -1e-10
    assert np.allclose(np.diag(hi), 1.0)
    assert np.allclose(hi, hi.T)
    raw = c.copy()
    raw[~np.eye(3, dtype=bool)] = np.clip(1.25 * c[~np.eye(3, dtype=bool)], -1, 1)
    # Closer to the raw scenario than the identity, which is also a correlation matrix.
    assert np.linalg.norm(hi - raw) < np.linalg.norm(np.eye(3) - raw)


def test_nearest_corr_leaves_a_correlation_matrix_alone():
    c = np.array([[1, 0.3, -0.2], [0.3, 1, 0.1], [-0.2, 0.1, 1]])
    assert np.allclose(common.nearest_corr(c), c, atol=1e-10)


# ─── Diversification alert ───
def _three_assets(n=1100, jump=0.9, base=0.2, seed=5):
    """A and B with correlation `base`, then `jump` over the last 30 days; C independent and small."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    z = rng.normal(size=(n, 3))
    a = z[:, 0]
    rho = np.where(np.arange(n) >= n - 30, jump, base)
    b = rho * a + np.sqrt(1 - rho ** 2) * z[:, 1]
    return pd.DataFrame({"A": 0.02 * a, "B": 0.02 * b, "C": 0.01 * z[:, 2]}, index=idx)


def test_warning_for_two_longs_and_the_pair_named():
    r = _three_assets()
    pos = {"A": 10e6, "B": 8e6, "C": 1e6}
    dc = common.diversification_check(r, pos)
    assert dc["warning"]
    assert {dc["pairs"][0]["a"], dc["pairs"][0]["b"]} == {"A", "B"}
    kind, text = common.diversification_box(dc, pos)
    assert kind == "warning" and "A and B" in text and "A long and B long" in text
    assert "higher correlation raises" in text


def test_no_warning_for_a_long_and_a_short_and_favourable_sentence():
    r = _three_assets()
    pos = {"A": 10e6, "B": -8e6, "C": 1e6}
    dc = common.diversification_check(r, pos)
    assert not dc["warning"] and not dc["watch"]
    fav = common.favourable_pairs_sentence(dc, pos)
    assert "A and B" in fav and "helps diversification" in fav


def test_largest_contributor_is_by_variance_not_by_correlation_change():
    # A-B correlation jumps the most, but the positions in B are tiny; A-C moves less with large positions.
    rng = np.random.default_rng(11)
    n = 1100
    idx = pd.bdate_range("2021-01-04", periods=n)
    z = rng.normal(size=(n, 3))
    last = np.arange(n) >= n - 30
    a = z[:, 0]
    b = np.where(last, 0.95, 0.0) * a + np.sqrt(1 - np.where(last, 0.95, 0.0) ** 2) * z[:, 1]
    c = np.where(last, 0.6, 0.2) * a + np.sqrt(1 - np.where(last, 0.6, 0.2) ** 2) * z[:, 2]
    r = pd.DataFrame({"A": 0.02 * a, "B": 0.02 * b, "C": 0.02 * c}, index=idx)
    dc = common.diversification_check(r, {"A": 10e6, "B": 0.2e6, "C": 10e6})
    assert {dc["pairs"][0]["a"], dc["pairs"][0]["b"]} == {"A", "C"}


def test_short_history_is_not_judged():
    r = _three_assets(n=260 + 100)
    dc = common.diversification_check(r, {"A": 10e6, "B": 8e6}, n_hist=750, min_hist=250)
    assert dc["n_hist"] == 360 - 250 + 1 and not dc["judged"]
    assert common.diversification_box(dc, {"A": 10e6, "B": 8e6}) == (None, "")


def test_one_position_gives_no_check():
    assert common.diversification_check(_three_assets(), {"A": 1e6}) is None


def test_v250_is_z_times_the_pnl_standard_deviation():
    r = _three_assets()
    pos = {"A": 10e6, "B": -5e6, "C": 3e6}
    dc = common.diversification_check(r, pos)
    pnl = common.book_pnl(r.tail(250), pos)
    assert dc["v250"] == pytest.approx(Z * pnl.std(ddof=1), rel=1e-9)


# ─── Euro conversion ───
def test_fund_conversion_removes_the_exchange_rate():
    idx = pd.bdate_range("2024-01-01", periods=5)
    x = pd.Series([1.10, 1.12, 1.09, 1.09, 1.15], index=idx)   # irregular, so a one-day lag shows
    usd = 20.0 * x                                             # euro price constant, USD line = euro x X
    assert np.allclose(common.eur_returns(usd, "usd_fund", x).dropna(), 0.0)


def test_fund_and_future_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=2)
    x = pd.Series([1.10, 1.00], index=idx)
    p = pd.Series([100.0, 110.0], index=idx)
    assert common.eur_returns(p, "usd_fund", x).iloc[-1] == pytest.approx(0.21)          # 1.1 x 1.1 - 1
    f = pd.Series([80.0, 88.0], index=idx)
    assert 1e6 * common.eur_returns(f, "usd_future", x).iloc[-1] == pytest.approx(110_000)
    flat = pd.Series([80.0, 80.0], index=idx)
    assert common.eur_returns(flat, "usd_future", x).iloc[-1] == pytest.approx(0.0)


def test_missing_exchange_rate_gives_no_return():
    idx = pd.bdate_range("2024-01-01", periods=3)
    x = pd.Series([1.10, np.nan, 1.00], index=idx)
    r = common.eur_returns(pd.Series([80.0, 81.0, 82.0], index=idx), "usd_future", x)
    assert r.isna().iloc[1] and r.isna().iloc[2]


# ─── Scenario windows and the fixed-quantity replay ───
def _path(values, start="2020-03-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_scenario_end_rule():
    v = list(np.linspace(100, 90, 40))
    v[11] = 50.0                                      # lowest close on day 12
    assert common.scenario_end(_path(v), "2020-03-02") == _path(v).index[11]
    v2 = list(np.linspace(100, 80, 40))
    v2[34] = 10.0                                     # lowest on day 35: outside the 30 days
    assert common.scenario_end(_path(v2), "2020-03-02") == _path(v2).index[29]
    v3 = list(np.linspace(100, 80, 40))
    v3[29], v3[30] = 5.0, 1.0                         # day 30 is counted, day 31 is not
    assert common.scenario_end(_path(v3), "2020-03-02") == _path(v3).index[29]


def _scenarios_from_page():
    """SCENARIOS as written in views/risk.py (the page cannot be imported without running it)."""
    import ast
    src = (Path(__file__).resolve().parents[1] / "views" / "risk.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SCENARIOS":
            return ast.literal_eval(node.value)
    raise AssertionError("SCENARIOS not found")


def test_page_scenarios_include_2008_and_2020():
    sc = {d["name"]: d for d in _scenarios_from_page()}
    assert (sc["Lehman collapse (2008)"]["first"], sc["Lehman collapse (2008)"]["last"]) == ("2008-09-15", "2008-10-27")
    assert (sc["Covid and oil price war (2020)"]["first"],
            sc["Covid and oil price war (2020)"]["last"]) == ("2020-03-06", "2020-03-31")


@pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="needs Yahoo Finance")
def test_scenario_windows_follow_the_rule():
    """The written end days of the 2008 and 2020 windows are Brent's lowest close in the first 30 trading days,
    read the way the page reads Brent (common.long_close)."""
    bz = common.long_close("BZ=F", pd.Timestamp("2026-11-20").date())
    for d in _scenarios_from_page():
        if d["first"] in ("2008-09-15", "2020-03-06"):
            assert common.scenario_end(bz, d["first"]) == pd.Timestamp(d["last"])


def test_replay_fixed_quantity_through_a_negative_price():
    base = _path([20.0, 18.27, -37.63, 10.01], start="2020-04-16")
    fx = pd.Series(1.0, index=base.index)
    res = common.replay_fixed({"WTI": base}, {"WTI": "usd_future"}, fx, {"WTI": 1e6},
                              "2020-04-17", "2020-04-21")
    # Quantity 1e6 / 20 bought at the 16 Apr close; P&L = quantity x (10.01 - 20).
    assert res["book"] == pytest.approx(1e6 / 20 * (10.01 - 20.0))
    assert res["worst_pnl"] == pytest.approx(1e6 / 20 * (-37.63 - 18.27))


def test_replay_future_with_exchange_rate():
    p = _path([80.0, 88.0], start="2024-01-01")
    x = pd.Series([1.10, 1.00], index=p.index)
    res = common.replay_fixed({"Brent": p}, {"Brent": "usd_future"}, x, {"Brent": 1e6}, "2024-01-02", "2024-01-02")
    assert res["book"] == pytest.approx(110_000)


def test_replay_reports_a_missing_position():
    oil = _path(list(np.linspace(100, 60, 25)), start="2008-09-01")
    gas = _path(list(np.linspace(20, 30, 10)), start="2017-10-23")
    fx = pd.Series(1.0, index=oil.index.union(gas.index))
    res = common.replay_fixed({"Oil": oil, "Gas": gas}, {"Oil": "eur", "Gas": "eur"}, fx,
                              {"Oil": -5e6, "Gas": 10e6}, "2008-09-15", "2008-10-03")
    assert res["missing_held"] == ["Gas"] and "book" not in res
    assert res["inst"]["Oil"]["pnl"] == pytest.approx(-5e6 * (oil.loc["2008-10-03"] / oil.loc["2008-09-12"] - 1))


# ─── Conditional moves and the estimate of a missing leg ───
def _cov3():
    s = np.array([0.03, 0.02, 0.025])
    c = np.array([[1, 0.6, 0.3], [0.6, 1, 0.4], [0.3, 0.4, 1]])
    return np.outer(s, s) * c * 10


def test_conditional_moves_by_hand():
    names = ["a", "b", "c"]
    m = common.conditional_moves(_cov3(), names, {"a": -0.25})
    assert m["b"] == pytest.approx(-0.10, abs=1e-12)        # 0.6 x (0.02 / 0.03) x -0.25
    assert m["c"] == pytest.approx(-0.0625, abs=1e-12)      # 0.3 x (0.025 / 0.03) x -0.25
    m2 = common.conditional_moves(_cov3(), names, {"a": 0.30, "b": 0.20})
    assert m2["c"] == pytest.approx(0.109375, abs=1e-12)


def test_estimate_missing_uses_the_longest_common_history():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2017-01-02", periods=1500)
    a = rng.normal(0, 0.02, 1500)
    b = 0.5 * (0.03 / 0.02) * a + rng.normal(0, 0.03 * np.sqrt(0.75), 1500)
    daily = pd.DataFrame({"oil": a, "gas": b}, index=idx)
    daily.loc[:idx[700], "gas"] = np.nan                         # gas starts later
    e = common.estimate_missing(daily, {"oil": -0.10}, "gas")
    assert e["start"] == idx[701] and e["n"] == 799
    assert e["ret"] == pytest.approx(-0.075, abs=0.02)           # 0.5 x 1.5 x -0.10


# ─── 10-day moves and borrowed severity ───
def test_ten_day_moves_find_the_worst_fall_and_skip_long_spans():
    idx = pd.bdate_range("2020-01-01", periods=60)
    p = pd.Series(100.0, index=idx)
    p.iloc[20:31] = np.linspace(100, 60, 11)       # a 40% fall over 10 rows ending at row 30
    p.iloc[31:] = 60.0
    mv = common.ten_day_moves(p, "eur")
    assert mv.min() == pytest.approx(-0.40) and mv.idxmin() == idx[30]
    gap = p.copy()
    gap.index = list(idx[:25]) + list(idx[25:] + pd.Timedelta(days=30))   # a hole inside the falling run
    gap.index = pd.DatetimeIndex(gap.index)
    assert common.ten_day_moves(gap, "eur").min() > -0.40


def test_ten_day_span_limit_is_inclusive():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=10).tolist() + [pd.Timestamp("2020-01-22")])
    p = pd.Series(np.linspace(100, 90, 11), index=idx)       # 16 calendar days from first to last close
    assert len(common.ten_day_moves(p, "eur")) == 1
    idx2 = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=10).tolist() + [pd.Timestamp("2020-01-23")])
    assert len(common.ten_day_moves(pd.Series(np.linspace(100, 90, 11), index=idx2), "eur")) == 0


def test_ten_day_moves_skip_non_positive_prices_and_convert_futures():
    idx = pd.bdate_range("2020-04-01", periods=12)
    p = pd.Series(np.linspace(20, 10, 12), index=idx)
    p.iloc[5] = -5.0
    assert common.ten_day_moves(p, "usd_future", pd.Series(1.0, index=idx)).empty
    q = pd.Series([80.0] * 10 + [88.0], index=pd.bdate_range("2024-01-01", periods=11))
    x = pd.Series([1.10] * 10 + [1.00], index=q.index)
    assert common.ten_day_moves(q, "usd_future", x).iloc[0] == pytest.approx(0.11)


def _summ(sigma, up, down):
    sq = sigma * np.sqrt(10)
    return dict(sigma=sigma, up=up, down=down, k_up=up / sq, k_down=abs(down) / sq)


def test_borrowed_severity_by_hand():
    ref = _summ(0.02, 6 * 0.02 * np.sqrt(10), -6 * 0.02 * np.sqrt(10))         # k_ref = 6
    small = _summ(0.01, 3 * 0.01 * np.sqrt(10), -3 * 0.01 * np.sqrt(10))       # own k = 3
    big = _summ(0.01, 8 * 0.01 * np.sqrt(10), -8 * 0.01 * np.sqrt(10))         # own k = 8
    out, capped = common.severity_shocks({"Brent Crude Oil": ref, "s": small, "b": big}, short=("s", "b"))
    assert out["s"]["up"] == (pytest.approx(6 * 0.01 * np.sqrt(10)), "B")
    assert out["s"]["down"] == (pytest.approx(-6 * 0.01 * np.sqrt(10)), "B")
    assert out["b"]["up"][1] == "W" and out["b"]["down"][1] == "W" and not capped


def test_borrowed_fall_of_100_percent_or_more_is_not_used():
    ref = _summ(0.02, 0.3, -6.7 * 0.02 * np.sqrt(10))
    vol = _summ(0.05, 1.0, -0.576)
    out, capped = common.severity_shocks({"Brent Crude Oil": ref, "TTF": vol}, short=("TTF",))
    assert capped == ["TTF"] and out["TTF"]["down"] == (-0.576, "W")


# ─── Reverse stress test ───
def test_reverse_stress_is_the_minimum_distance():
    cov = _cov3()
    w = np.array([10e6, -5e6, 3e6])
    L = 7.5e6
    rv = common.reverse_stress(cov, w, L)
    assert w @ rv["r"] == pytest.approx(-L, rel=1e-12)
    assert rv["pnl"].sum() == pytest.approx(-L, rel=1e-12)
    inv = np.linalg.inv(cov)
    opt = minimize(lambda r: r @ inv @ r, np.full(3, -0.1), method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda r: w @ r + L}], options={"ftol": 1e-14})
    assert np.allclose(opt.x, rv["r"], atol=1e-6)
    assert np.sqrt(rv["r"] @ inv @ rv["r"]) == pytest.approx(rv["d"])
    # Any other scenario with the same loss is further away: r'S^-1 r = d*^2 + (r - r*)' S^-1 (r - r*).
    rng = np.random.default_rng(0)
    for _ in range(200):
        e = rng.normal(size=3)
        e -= w * (w @ e) / (w @ w)                    # a direction inside the constraint plane
        r = rv["r"] + 0.05 * e
        assert r @ inv @ r == pytest.approx(rv["d"] ** 2 + (r - rv["r"]) @ inv @ (r - rv["r"]), rel=1e-10)


def test_reverse_stress_with_a_diagonal_covariance():
    s2 = np.array([0.04, 0.01])
    w = np.array([2.0, 3.0])
    rv = common.reverse_stress(np.diag(s2), w, 1.0)
    assert np.allclose(rv["r"] / (w * s2), rv["r"][0] / (w[0] * s2[0]))


def test_mahalanobis_of_joint_moves():
    cov = np.diag([0.04, 0.01])
    assert common.mahalanobis(np.array([[0.2, 0.0], [0.0, 0.2]]), cov) == pytest.approx([1.0, 2.0])


# ─── Hypothetical rows ───
def test_hypothetical_rows_single_factor_and_narratives():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2021-10-01", periods=250)
    z = rng.normal(size=(250, 3))
    rets = pd.DataFrame({"TTF Natural Gas": 0.05 * z[:, 0], "Brent Crude Oil": 0.02 * (0.3 * z[:, 0] + z[:, 1]),
                         "EU Carbon ETC": 0.02 * z[:, 2]}, index=idx)
    shocks = {k: dict(up=(0.5, "W"), down=(-0.4, "W")) for k in rets}
    pos = {"TTF Natural Gas": 10e6, "Brent Crude Oil": -5e6}
    rows = common.hypothetical_rows(shocks, rets, pos, "EU Carbon ETC", True)
    single = [r for r in rows if r["type"] == "single factor"]
    assert len(single) == 4 and single[0]["pnl"] == pytest.approx(5e6)
    gulf = next(r for r in rows if r["name"] == "Gulf supply disruption")
    assert gulf["pnl"] == pytest.approx(10e6 * 0.5 - 5e6 * 0.5)
    rets.loc[idx[100]:, "EU Carbon ETC"] = np.nan                 # too few carbon returns in the stress period
    rows = common.hypothetical_rows(shocks, rets, pos, "EU Carbon ETC", True)
    carbon = next(r for r in rows if r["name"] == "Carbon market shock")
    assert carbon["pnl"] is None and carbon["missing"] == ["EU Carbon ETC"]
    assert all(r["name"] != "Carbon market shock"
               for r in common.hypothetical_rows(shocks, rets, pos, "EU Carbon ETC", False))


# ─── Page text: no forecasts, no dashes, no semicolons ───
FORBIDDEN = re.compile(r"\b(will|likely|may|might|expected to|could)\b|[—–−;]", re.IGNORECASE)


def _texts():
    r = _three_assets()
    out = []
    for pos in ({"A": 10e6, "B": 8e6, "C": 1e6}, {"A": 10e6, "B": -8e6, "C": 1e6}):
        dc = common.diversification_check(r, pos)
        out.append(common.diversification_box(dc, pos)[1])
        out.append(common.favourable_pairs_sentence(dc, pos))
        for p in dc["pairs"]:
            out.append(common.pair_sentence(p, pos))
    # A WATCH box: one adverse pair, no WARNING.
    dc = common.diversification_check(r, {"A": 10e6, "B": 8e6, "C": 1e6})
    dc = dict(dc, warning=False, watch=True)
    dc["pairs"] = [dict(p, adverse=True, side="above") for p in dc["pairs"][:1]]
    out.append(common.diversification_box(dc, {"A": 10e6, "B": 8e6, "C": 1e6})[1])
    rv = common.reverse_stress(_cov3(), np.array([10e6, -5e6, 3e6]), 30e6)
    for worst in (dict(name="X", type="historical", horizon="18 trading days", pnl=-9e6, days=18),
                  dict(name="Y", type="single factor", horizon="10 days", pnl=-2e6, days=10)):
        out += common.summary_lines(worst, 2, 7.5e6, True, 1.5e6, rv, ["a", "b", "c"], 5.2,
                                    pd.Timestamp("2022-03-08"))
    return [t for t in out if t]


def test_text_has_no_forecast_words_dashes_or_semicolons():
    for t in _texts():
        assert not FORBIDDEN.search(t), (FORBIDDEN.search(t).group(0), t)


def test_summary_lines_above_and_below():
    worst = dict(name="Y", type="single factor", horizon="10 days", pnl=-5.76e6, days=10)
    below = common.summary_lines(worst, 0, 7.5e6, False, 1.5e6, None, [], 0, None)
    assert "below it by €1.74m" in below[1] and "-€5.76m" in below[0] and "placeholder" not in below[1]
    above = common.summary_lines(dict(worst, pnl=-9e6), 2, 7.5e6, True, 1.5e6, None, [], 0, None)
    assert "above it by €1.50m" in above[1] and "2 historical scenarios are left out" in above[0]


def test_summary_flags_a_move_below_zero():
    rv = common.reverse_stress(_cov3(), np.array([1e6, 0.1e6, 0.1e6]), 30e6)
    line = common.summary_lines(None, 0, 30e6, False, 1.5e6, rv, ["a", "b", "c"], 5.0,
                                pd.Timestamp("2022-03-08"))[0]
    assert "a, b, c need a price below zero" in line


# ─── Checks that tell a right implementation from a wrong one (from the code review of 23 Nov 2026) ───
def _independent_check(r, pos, stress=None, window=250, short=30, n_hist=750):
    """The diversification check written out again, day by day, without the library function."""
    names = list(pos)
    X = r[names].dropna().values
    w = np.array([pos[k] for k in names])
    n = len(X)
    k = min(n_hist, n - window + 1)
    ratios, pair = [], {(0, 1): [], (0, 2): [], (1, 2): []}
    for t in range(n - k, n):
        blk = X[t - window + 1:t + 1]
        s_t = blk.std(axis=0, ddof=1)
        ca, cb = np.corrcoef(blk, rowvar=False), np.corrcoef(X[t - short + 1:t + 1], rowvar=False)
        ratios.append(np.sqrt((w * s_t) @ cb @ (w * s_t)) / np.sqrt((w * s_t) @ ca @ (w * s_t)) - 1)
        for p in pair:
            pair[p].append(cb[p])
    return np.array(ratios), {p: np.percentile(v, [5, 95]) for p, v in pair.items()}


def test_history_thresholds_match_an_independent_calculation():
    r = _three_assets(seed=9)
    pos = {"A": 10e6, "B": -4e6, "C": 6e6}
    dc = common.diversification_check(r, pos)
    ratios, ranges = _independent_check(r, pos)
    assert dc["n_hist"] == len(ratios) == 750
    assert dc["p95"] == pytest.approx(np.percentile(ratios, 95), rel=1e-12)
    assert dc["ratio"] == pytest.approx(ratios[-1], rel=1e-9)
    names = list(pos)
    for p in dc["pairs"]:
        i, j = sorted((names.index(p["a"]), names.index(p["b"])))
        assert (p["lo"], p["hi"]) == pytest.approx(tuple(ranges[(i, j)]), rel=1e-12)


def test_warning_needs_the_95th_percentile():
    r = _three_assets(seed=9)
    pos = {"A": 10e6, "B": -4e6, "C": 6e6}
    ratios, _ = _independent_check(r, pos)
    dc = common.diversification_check(r, pos)
    assert dc["warning"] == (ratios[-1] > np.percentile(ratios, 95))
    # A ratio between the 90th and the 95th percentile is no WARNING.
    fake = dict(dc, ratio=float(np.percentile(ratios, 92)))
    assert not (fake["ratio"] > dc["p95"])


def test_stress_frtb_and_pair_values_match_by_hand():
    r = _three_assets(seed=9)
    pos = {"A": 10e6, "B": -4e6, "C": 6e6}
    stress = dict(start=r.index[100], end=r.index[349])
    dc = common.diversification_check(r, pos, stress=stress)
    X = r.values
    w = np.array(list(pos.values()))
    s = X[-250:].std(axis=0, ddof=1)
    cs = np.corrcoef(r.loc[stress["start"]:stress["end"]].values, rowvar=False)
    assert dc["vstress"] == pytest.approx(Z * np.sqrt((w * s) @ cs @ (w * s)))
    vals = {k: common.param_var(w, s, m) for k, m in (("high", common.frtb_corr(dc["c250"], "high")[0]),
                                                       ("250-day", dc["c250"]),
                                                       ("low", common.frtb_corr(dc["c250"], "low")[0]))}
    assert dc["frtb_which"] == max(vals, key=vals.get)
    c = dc["c250"].copy()
    top = dc["pairs"][0]
    i, j = list(pos).index(top["a"]), list(pos).index(top["b"])
    c[i, j] = c[j, i] = top["rho30"]
    assert top["v_pair"] == pytest.approx(Z * np.sqrt((w * s) @ c @ (w * s)))


def test_watch_needs_enough_history():
    r = _three_assets(n=400)
    dc = common.diversification_check(r, {"A": 10e6, "B": 8e6}, min_hist=250)
    assert not dc["judged"] and not dc["watch"] and not dc["warning"]


def test_higham_paper_example():
    a = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], float)
    x = common.nearest_corr(a)
    assert x[0] == pytest.approx([1.0, -0.8084, 0.1916, 0.1068], abs=1e-4)


def test_ten_day_summary_sigma_is_the_daily_volatility():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2018-01-01", periods=600)
    p = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.02, 600)), index=idx)
    sm = common.ten_day_summary(p, "eur")
    assert sm["sigma"] == pytest.approx(common.simple_returns(p).std())
    mv = common.ten_day_moves(p, "eur")
    assert sm["k_down"] == pytest.approx(abs(mv.min()) / (sm["sigma"] * np.sqrt(10)))


def test_narrative_moves_the_other_positions_by_their_conditional_mean():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2021-10-01", periods=250)
    z = rng.normal(size=(250, 3))
    rets = pd.DataFrame({"TTF Natural Gas": 0.04 * (0.5 * z[:, 1] + z[:, 0]), "Brent Crude Oil": 0.02 * z[:, 1],
                         "EU Carbon ETC": 0.02 * (0.3 * z[:, 0] + z[:, 2])}, index=idx)
    rets.iloc[5, 0] = np.nan                                     # a gap: the covariance uses common rows only
    shocks = {k: dict(up=(0.5, "W"), down=(-0.4, "W")) for k in rets}
    pos = {"TTF Natural Gas": 10e6, "Brent Crude Oil": -5e6, "EU Carbon ETC": 3e6}
    demand = next(r for r in common.hypothetical_rows(shocks, rets, pos, "EU Carbon ETC", True)
                  if r["name"] == "Demand collapse")
    blk = rets.dropna()
    cov = blk.cov().values
    beta = cov[[0, 2], 1] / cov[1, 1]                          # TTF and carbon on Brent
    expected = -5e6 * -0.4 + 10e6 * beta[0] * -0.4 + 3e6 * beta[1] * -0.4
    assert demand["pnl"] == pytest.approx(expected, rel=1e-10)
    assert demand["moves"]["TTF Natural Gas"][1] == "C"


def test_joint_ten_day_skips_long_spans():
    idx = pd.DatetimeIndex(list(pd.bdate_range("2022-01-03", periods=15)) +
                           list(pd.bdate_range("2022-03-01", periods=15)))
    r = pd.DataFrame({"a": 0.01, "b": -0.01}, index=idx)
    j = common.joint_ten_day(r)
    assert all((e - s).days <= 16 for s, e in zip(idx[:-10][[idx.get_loc(t) - 10 for t in j.index]], j.index))
    assert len(j) == 10                                           # 5 runs before the gap, 5 after it
    assert j.iloc[0].tolist() == pytest.approx([0.1, -0.1])


def test_hist_diversification_is_positive_for_two_imperfect_longs():
    r = _three_assets()
    div, book, alone = common.hist_diversification(r.tail(250), {"A": 5e6, "C": 5e6})
    assert div > 0 and alone == pytest.approx(book + div)


def test_summary_multiple_uses_the_var_limit():
    worst = dict(name="Y", type="single factor", horizon="10 days", pnl=-5.76e6, days=10)
    line = common.summary_lines(worst, 0, 7.5e6, False, 1.5e6, None, [], 0, None)[0]
    assert "3.8x the VaR 95% limit (1-day)" in line


def test_replay_coverage_rule():
    oil = _path(list(np.linspace(100, 60, 25)), start="2020-03-02")
    thin = oil.copy()
    thin.iloc[6:20] = np.nan                                     # prices on far fewer than 70% of the days
    res = common.replay_fixed({"Oil": oil, "Thin": thin}, {"Oil": "eur", "Thin": "eur"},
                              pd.Series(1.0, index=oil.index), {"Oil": 1e6, "Thin": 1e6}, "2020-03-09", "2020-04-03")
    assert res["inst"]["Thin"] is None and res["missing_held"] == ["Thin"]


def test_pair_sentence_names_the_range_only_when_outside():
    p = dict(a="A", b="B", rho30=0.9, rho250=0.2, lo=-0.1, hi=0.6, side="above", same_sign=True)
    assert "above its 3-year range of -0.10 to 0.60" in common.pair_sentence(p, {"A": 1, "B": 1})
    assert "range" not in common.pair_sentence(dict(p, side=None), {"A": 1, "B": 1})
    assert "a lower correlation raises" in common.pair_sentence(dict(p, same_sign=False), {"A": 1, "B": -1})


def test_small_helpers():
    assert [common.quote_kind(t) for t in ("BZ=F", "CL=F", "CARB.L", "ICLN", "TTF=F", "CARB.MI")] == \
        ["usd_future", "usd_future", "usd_fund", "usd_fund", "eur", "eur"]
    assert common.book_panel_tickers("EU Carbon ETC", "CARB.L")["EU Carbon ETC"] == "CARB.MI"
    assert common.book_panel_tickers("Clean Energy Proxy", "ICLN")["Clean Energy Proxy"] == "ICLN"
    assert [n["name"] for n in common.narratives("EU Carbon ETC", False)] == ["Gulf supply disruption",
                                                                             "Demand collapse"]
    assert common.pct(-0.001) == "-0.1%" and common.pct(-0.76) == "-76%"
    assert common.eur_m(-5.76e6) == "-€5.76m" and common.eur_m(4e4) == "+€0.04m"


def test_page_text_of_the_new_sections():
    """Every string literal of views/risk.py from the positions section on: no forecast words, no dashes, no
    semicolons (an HTML style attribute is not text)."""
    import ast
    src = (Path(__file__).resolve().parents[1] / "views" / "risk.py").read_text()
    line0 = src[:src.index("# ─── Section 5a")].count("\n") + 1
    bad = re.compile(r"\b(will|likely|may|might|expected to|could)\b|[—–−;]|rather than|not only")
    hits = [(n.lineno, n.value[:60]) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.lineno >= line0
            and "font-weight" not in n.value and bad.search(n.value)]
    assert hits == []


# ─── Fixes of 24 Nov 2026: blank cells, two decimals, Brent's own last day ───

def _fmt(cfg, col):
    c = cfg.get(col)
    return None if c is None else (c.get("type_config") or {}).get("format")


def test_table_config_keeps_numbers_and_formats():
    df = pd.DataFrame({"Position (€m)": [0.25, -5.0, np.nan],
                       "Book P&L (€m)": [-3.0, np.nan, 16.89],
                       "Trading days": pd.array([31, None, 9], dtype="Int64"),
                       "Trading days as float": [31.0, np.nan, 9.0],
                       "Worst day": ["25 Feb 2022", None, ""],
                       "x VaR 95% limit (1-day)": ["3.8", "", "0.0"]},
                      index=["a", "b", "c"])
    cfg = common.table_config(df)
    assert _fmt(cfg, "Book P&L (€m)") == "%.2f"
    assert _fmt(cfg, "Trading days") == "%d" and _fmt(cfg, "Trading days as float") == "%d"
    # Positions are shown as entered (0.25 stays 0.25), text columns are left alone.
    assert "Position (€m)" not in cfg and "Worst day" not in cfg and "x VaR 95% limit (1-day)" not in cfg
    # Numbers stay numbers, so a click on a column header sorts them as numbers.
    assert pd.api.types.is_float_dtype(df["Book P&L (€m)"])


def test_show_table_blanks_missing_cells(monkeypatch):
    import streamlit as st
    seen = {}
    monkeypatch.setattr(st, "dataframe", lambda df, **kw: seen.update(df=df, **kw))
    df = pd.DataFrame({"Book P&L (€m)": [np.nan, 11.66]}, index=["Lehman", "Russia"])
    common.show_table(df)
    assert seen["placeholder"] == ""
    assert seen["df"] is df and pd.isna(seen["df"].iloc[0, 0])
    assert _fmt(seen["column_config"], "Book P&L (€m)") == "%.2f"


def test_ten_day_summary_has_last_data_day():
    idx = pd.bdate_range("2024-01-01", periods=40)
    close = pd.Series(np.linspace(50, 60, 40), index=idx)
    close.iloc[-2:] = np.nan
    s = common.ten_day_summary(close, "eur")
    assert s["last"] == idx[-3]


def test_risk_page_tables_and_brent_date():
    src = open(os.path.join(os.path.dirname(__file__), "..", "views", "risk.py"), encoding="utf-8").read()
    sec5 = src[src.index("# ─── Section 5a: Positions & Limits ───"):]
    # Section 5 tables go through _table; only the backtest table, which has no missing cells, is shown directly.
    assert sec5.count("st.dataframe(") == 1
    assert "st.table(" not in sec5 and "st.write(pd." not in sec5
    assert "def _table(df):" in src and "common.show_table(df)" in src
    # Positions in the Positions & Limits table stay numbers (0.25 is not shown as "+0.2").
    assert "f\"{v:+.1f}\"" not in sec5
    assert "Jul 2007 to {end_date" not in src and "Jul 2007 to {_brent_last" in src
    assert '_brent_last = _ref["last"] if _ref else end_date' in src


def test_streamlit_minimum_version():
    # show_table needs st.dataframe(placeholder=...) (from 1.52) and app.py needs st.Page(visibility=...) (from 1.55).
    req = open(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"), encoding="utf-8").read()
    assert "streamlit>=1.55" in req
