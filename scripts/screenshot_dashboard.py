"""Capture dashboard screenshots for the README with headless Chrome/Edge.

Usage: python scripts/screenshot_dashboard.py [--channel chrome|msedge]
Starts the Streamlit app on a spare port, captures docs/images/dashboard_*.png, then stops it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

from tram_mlops.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "images"
PORT = 8765
URL = f"http://localhost:{PORT}"


def wait_for_server(timeout_s: float = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("Streamlit did not start")


def capture(channel: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel)
        # A viewport taller than the page means Streamlit's inner container never scrolls,
        # so every section is rendered and can be clipped from one frame.
        page = browser.new_page(viewport={"width": 1400, "height": 3400}, device_scale_factor=1.5)
        page.goto(URL, wait_until="networkidle")

        def settle() -> None:
            page.wait_for_selector("[data-testid='stMetric']", timeout=60_000)
            page.wait_for_timeout(8000)  # charts and map tiles render after the metrics

        def top(heading: str) -> float:
            return page.get_by_role("heading", name=heading).first.bounding_box()["y"]

        def shoot(name: str, start: str | None, end: str) -> None:
            main = page.locator("[data-testid='stMain']").bounding_box()
            y0 = top(start or "Helsinki Tram Delays") - 24
            page.screenshot(path=OUT / name, clip={
                "x": main["x"], "y": y0, "width": main["width"], "height": top(end) - 24 - y0,
            })
            print(f"wrote docs/images/{name}")

        settle()
        # The dashboard opens in Live mode only while the collector is writing fresh data.
        if page.get_by_role("heading", name="Line status").count():
            shoot("dashboard_live.png", None, "Delay forecast")
            page.get_by_text("History", exact=True).click()
            page.wait_for_timeout(1000)
            settle()
        else:
            print("collector not running; skipping dashboard_live.png")

        shoot("dashboard_overview.png", None, "Latest positions")
        shoot("dashboard_map.png", "Latest positions", "Delay forecast")
        shoot("dashboard_forecast.png", "Delay forecast", "Most delayed trams")
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="chrome", help="installed browser to drive (chrome or msedge)")
    args = parser.parse_args()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "src/tram_mlops/dashboard.py"),
         "--server.headless=true", f"--server.port={PORT}", "--theme.base=light",
         "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal"],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server()
        capture(args.channel)
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
