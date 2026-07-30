"""
Keep the European Energy & Commodity Risk Dashboard awake on Streamlit Community Cloud.

Streamlit Community Cloud hibernates any app that receives no traffic for ~12 hours.
A plain HTTP GET is NOT enough to wake it: the sleeping page returns HTTP 200 with a
static HTML shell while the Python process stays down. We therefore drive a real
headless Chromium so the WebSocket handshake happens, and click the wake-up button
when the app is found asleep.

Run locally:
    pip install playwright && playwright install chromium
    python .github/scripts/keepalive.py
"""

from __future__ import annotations

import os
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_URL",
    "https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app",
)

# Text that only renders once the real app is up (the st.title of app.py).
APP_READY_TEXT = re.compile(r"European Energy\s*&\s*Commodity Risk Dashboard", re.I)

# Text on the Streamlit hibernation page.
WAKE_BUTTON_TEXT = re.compile(r"get this app back up", re.I)

NAV_TIMEOUT_MS = 60_000
WAKE_CHECK_MS = 15_000
# Cold start is slow: 4x yfinance download + GARCH fit + 500-draw bootstrap
# + 5 RSS feeds + FinBERT API warm-up.
READY_TIMEOUT_MS = 240_000

FAILURE_SCREENSHOT = "keepalive-failure.png"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    log(f"Target: {APP_URL}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeout:
            log("FAIL: navigation timed out.")
            page.screenshot(path=FAILURE_SCREENSHOT, full_page=True)
            browser.close()
            return 1

        # --- Was it asleep? ---
        wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
        was_asleep = False
        try:
            wake_button.wait_for(state="visible", timeout=WAKE_CHECK_MS)
            was_asleep = True
            log("App was ASLEEP -> clicking wake-up button.")
            wake_button.click()
        except PlaywrightTimeout:
            log("No hibernation page found -> app was already awake or is booting.")

        # --- Wait until the app has actually rendered ---
        try:
            page.get_by_text(APP_READY_TEXT).first.wait_for(
                state="visible", timeout=READY_TIMEOUT_MS
            )
        except PlaywrightTimeout:
            log("FAIL: app did not finish rendering in time.")
            page.screenshot(path=FAILURE_SCREENSHOT, full_page=True)
            browser.close()
            return 1

        # Stay connected a little longer so Streamlit registers a real session
        # and the @st.cache_data blocks finish populating.
        page.wait_for_timeout(20_000)

        state = "woken up" if was_asleep else "kept awake"
        log(f"OK: app is {state} and rendering.")

        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
