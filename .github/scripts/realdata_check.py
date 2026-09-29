"""Real-data check: runs every page on live data and writes realdata_report.md."""
import datetime as dt, sys, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import common, country_data, power_data
from streamlit.testing.v1 import AppTest

out = []
w = out.append
start, end = common.DEFAULT_START, dt.date.today()
w(f"# Real-data report {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC\n")

try:
    carbon = common.resolve_carbon_benchmark()
    comm = common.build_commodities(carbon)
    w(f"Carbon benchmark: {carbon['ticker']} ({carbon['label']})\n")
    for name, c in comm.items():
        t = c['ticker']
        raw = common._raw_close(t, start, end)
        close = common.load_close(t, start, end)
        w(f"## {name} ({t})")
        w(f"- rows {len(raw)}, first {raw.index.min()}, last {raw.index.max()}, NaN after stale mask {int(close.isna().sum())}")
        w(f"- last 5 closes: {[(str(d.date()), round(v, 3)) for d, v in raw.tail(5).items()]}")
        w(f"- min {raw.min():.3f} on {raw.idxmin().date()}, max {raw.max():.3f} on {raw.idxmax().date()}")
        w(f"- stale periods: {common.stale_periods(t, start, end)}")
        cmp_t = "TTF=F" if t == carbon['ticker'] else carbon['ticker']
        core = common.compute_core(t, cmp_t, start, end)
        if core is None:
            w("- compute_core returned None"); continue
        r = core['returns_clean']
        w(f"- returns n={len(r)}, largest up {r.max()*100:.2f}% on {r.idxmax().date()}, largest down {r.min()*100:.2f}% on {r.idxmin().date()}")
        w(f"- vol latest {core['latest_vol']:.2f}%, avg {core['avg_vol']:.2f}%, regime {core['current_regime']}, thresholds {core['regime_thr']}")
        w(f"- VaR95 {core['var_95']:.2f}%, VaR99 {core['var_99']:.2f}%, ES97.5 {core['es_975']:.2f}% (days {core['var_days']}); full {core['var_95_full']:.2f}/{core['var_99_full']:.2f}/{core['es_975_full']:.2f}")
        g = common.fit_garch(r)
        if g:
            w(f"- GARCH params {g['params']}, persistence {g['persistence']:.4f}, nu {g['nu']}, reference {g['long_run_vol']:.2f}%, garch long-run {g['garch_long_run']} ok={g['garch_long_run_ok']}, 10d forecast {g['forecast_10d']:.2f}%")
        else:
            w("- GARCH fit failed")
        bt = common.var_backtest(r, window=core['var_days'], test_days=250)
        for conf, col in [(0.95, 'Exc95'), (0.99, 'Exc99')]:
            x = int(bt[col].sum()); lr, p = common.kupiec_pof(len(bt), x, 1 - conf)
            w(f"- backtest {conf:.0%}: {x} exceptions / {len(bt)} days, expected {len(bt)*(1-conf):.1f}, Kupiec p {p:.3f}")
        w(f"- Basel zone {common.basel_zone(int(bt['Exc99'].sum()), len(bt))}\n")
except Exception:
    w("```\n" + traceback.format_exc() + "```")

w("## Country data (Eurostat)")
try:
    data, complete, failed = country_data.load_country_data()
    w(f"- failed datasets: {failed}")
    for y in country_data.YEARS:
        cov = {k: sum(v is not None for v in data[k][y]) for k in country_data.SERIES}
        unscored = [c for c, ok in zip(country_data.COUNTRIES, complete[y]) if not ok]
        w(f"- {y}: coverage {cov}; not scored {unscored}")
    for c in ('Germany', 'France', 'Poland', 'Norway', 'Netherlands', 'Greece', 'Switzerland', 'United Kingdom', 'Turkey'):
        i = country_data.COUNTRIES.index(c)
        w(f"- {c}: " + "; ".join(f"{y}: " + ", ".join(f"{k} {data[k][y][i]}" for k in country_data.SERIES) for y in (2022, 2024)))
except Exception:
    w("```\n" + traceback.format_exc() + "```")

w("\n## Roll check")
try:
    import yfinance as yf
    for t in ("BZ=F", "BZX26.NYM", "BZZ26.NYM", "CL=F", "CLX26.NYM", "CLZ26.NYM", "TTF=F"):
        d = yf.download(t, start="2026-09-18", end=str(dt.date.today() + dt.timedelta(days=1)), progress=False)
        c = d['Close']; c = c.iloc[:, 0] if hasattr(c, 'columns') else c
        w(f"- {t}: " + ", ".join(f"{i.date()} {v:.2f}" for i, v in c.items()))
except Exception:
    w("```\n" + traceback.format_exc() + "```")

w("\n## EEX contracts")
try:
    import io, requests, pandas as pd
    for yr in (2021, 2023, 2026):
        s_ = power_data.fetch_eex_year(yr)
        w(f"- {yr}: {len(s_)} auction days, first {s_.index.min().date()}, last {s_.index.max().date()}, mean {s_.mean():.2f}")
except Exception:
    w("```\n" + traceback.format_exc() + "```")

w("\n## Power data")
try:
    p, lic, err = power_data.load_power(start, end)
    w(f"- DE-LU day-ahead: n {len(p)}, last {p.index.max() if len(p) else None}, last 5 {[(str(d.date()), round(v, 2)) for d, v in p.tail(5).items()]}, errors {err}")
    e, err2 = power_data.load_eua_auctions(start, end)
    w(f"- EEX EUA auctions: n {len(e)}, last {e.index.max() if len(e) else None}, last 5 {[(str(d.date()), round(v, 2)) for d, v in e.tail(5).items()]}, errors {err2}")
except Exception:
    w("```\n" + traceback.format_exc() + "```")

w("\n## Pages")
try:
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "app.py"), default_timeout=600); at.run()
    for o in at.selectbox(key="commodity").options:
        at.selectbox(key="commodity").set_value(o)
        for page in ("views/risk.py", "views/power.py", "views/market.py"):
            if page == "views/power.py" and o != at.selectbox(key="commodity").options[0]:
                continue
            at.switch_page(page); at.run()
            w(f"### {page} | {o}")
            w(f"- exceptions: {[str(e.value)[:500] for e in at.exception]}")
            w(f"- errors: {[str(e.value)[:300] for e in at.error]}")
            w(f"- warnings: {[str(e.value)[:300] for e in at.warning]}")
            w(f"- info: {[str(e.value)[:300] for e in at.info]}")
            w(f"- markdown with Anomaly/GARCH: {[str(m.value)[:200] for m in at.markdown if 'GARCH Forecast' in str(m.value) or 'Anomal' in str(m.value)][:4]}")
            w("- metrics: " + "; ".join(f"{m.label}={m.value}" + (f" ({m.delta})" if m.delta else "") for m in at.metric))
            for cpt in at.caption:
                w(f"  - caption: {str(cpt.value)[:400]}")
except Exception:
    w("```\n" + traceback.format_exc() + "```")

open("realdata_report.md", "w").write("\n".join(out))
