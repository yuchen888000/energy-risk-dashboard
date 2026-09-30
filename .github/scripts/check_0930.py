"""Temporary real-data check for the 30 Sep fixes."""
import subprocess, sys, time, pathlib, traceback, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "check_out"; OUT.mkdir(exist_ok=True)
rep = []; w = rep.append

# 1. Rolling-correlation gaps, old vs new, on real prices
try:
    import yfinance as yf, pandas as pd
    def close(t):
        d = yf.download(t, start="2020-01-01", progress=False, auto_adjust=False)["Close"]
        return d.squeeze().dropna()
    for a, b in (("TTF=F", "CARB.L"), ("BZ=F", "CARB.L"), ("CARB.L", "TTF=F")):
        pa, pb = close(a), close(b)
        df = pd.DataFrame({"P": pa, "C": pb, "R": pa.pct_change(fill_method=None),
                           "RC": pb.pct_change(fill_method=None)}).dropna(subset=["P"])
        old = df["R"].rolling(30).corr(df["RC"]); new = df["R"].rolling(30, min_periods=20).corr(df["RC"])
        since = df.index >= "2022-01-01"
        w(f"- {a} vs {b}: days {since.sum()} since 2022; NaN correlation old {old[since].isna().sum()}, new {new[since].isna().sum()}; "
          f"last value old {old.dropna().iloc[-1]:.3f} new {new.dropna().iloc[-1]:.3f}")
except Exception:
    w("```\n" + traceback.format_exc() + "```")

# 2. Screenshots
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                         "--server.headless", "true", "--server.port", "8501"], cwd=ROOT)
time.sleep(15)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": 1500, "height": 1000})
        for path in ("risk", "power"):
            pg.goto(f"http://localhost:8501/{path}", timeout=120000)
            for _ in range(60):
                time.sleep(5)
                if pg.locator("[data-testid='stStatusWidget']").count() == 0:
                    break
            time.sleep(10)
            pg.screenshot(path=str(OUT / f"{path}.png"), full_page=True)
            w(f"screenshot {path} ok")
        br.close()
except Exception:
    w("```\n" + traceback.format_exc() + "```")
proc.terminate()

# 3. AppTest on every page and commodity
from streamlit.testing.v1 import AppTest
try:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600); at.run()
    for o in at.selectbox(key="commodity").options:
        at.selectbox(key="commodity").set_value(o)
        for page in ("views/risk.py", "views/power.py", "views/market.py"):
            at.switch_page(page); at.run()
            w(f"### {page} | {o}")
            w(f"- exceptions {[str(e.value)[:400] for e in at.exception]}")
            w(f"- errors {[str(e.value)[:200] for e in at.error]}")
            caps = [str(c.value) for c in at.sidebar.caption]
            w(f"- sidebar captions: {[c[:140] for c in caps]}")
            w(f"- full-period captions: {[str(c.value)[:80] for c in at.caption if 'full period' in str(c.value)]}")
except Exception:
    w("```\n" + traceback.format_exc() + "```")
(OUT / "report.md").write_text("\n".join(rep))
