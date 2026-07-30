import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_URL",
    "https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app",
)

WAKE_BUTTON = "get this app back up"
DWELL_MS = 120_000
FAILURE_SCREENSHOT = "keepalive-failure.png"


def main():
    print("Target: " + APP_URL, flush=True)

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
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
        except PlaywrightTimeout:
            print("FAIL: could not reach the app.", flush=True)
            page.screenshot(path=FAILURE_SCREENSHOT, full_page=True)
            browser.close()
            return 1

        try:
            button = page.get_by_role("button", name=WAKE_BUTTON)
            button.wait_for(state="visible", timeout=20_000)
            button.click()
            print("App was ASLEEP -> clicked the wake-up button.", flush=True)
        except PlaywrightTimeout:
            print("No hibernation page -> app was already awake.", flush=True)

        print("Holding the session open to let the app boot...", flush=True)
        page.wait_for_timeout(DWELL_MS)

        print("Done.", flush=True)
        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
