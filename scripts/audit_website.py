from __future__ import annotations

import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "website_audit"
OUT.mkdir(parents=True, exist_ok=True)


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,1100")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--disable-gpu")
    return webdriver.Chrome(options=opts)


def save_state(driver: webdriver.Chrome, name: str) -> dict:
    time.sleep(0.7)
    png = OUT / f"{name}.png"
    txt = OUT / f"{name}.txt"
    driver.save_screenshot(str(png))
    body = driver.find_element(By.TAG_NAME, "body").text
    txt.write_text(body, encoding="utf-8")
    return {
        "name": name,
        "url": driver.current_url,
        "screenshot": str(png),
        "text": body,
    }


def click_text(driver: webdriver.Chrome, text: str) -> None:
    q = json.dumps(text)
    xpath = (
        f"//button[contains(normalize-space(.), {q})] | "
        f"//a[contains(normalize-space(.), {q})] | "
        f"//*[@role='button' and contains(normalize-space(.), {q})]"
    )
    el = WebDriverWait(driver, 8).until(EC.element_to_be_clickable((By.XPATH, xpath)))
    el.click()


def main() -> None:
    driver = make_driver()
    states = []
    try:
        driver.get("http://localhost:5173/")
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        states.append(save_state(driver, "01_home"))

        # Open first catalogue segment.
        links = driver.find_elements(By.CSS_SELECTOR, "a[href^='/viewer/'], a[href*='/viewer/']")
        if links:
            links[0].click()
        else:
            driver.get("http://localhost:5173/viewer/20231221180251")
        WebDriverWait(driver, 15).until(EC.url_contains("/viewer/"))
        states.append(save_state(driver, "02_viewer_default"))

        for label, name in [
            ("Transcription", "03_transcription"),
            ("Text Deciphering", "04_text_deciphering"),
            ("Scholar", "05_scholar"),
            ("Process", "06_process"),
        ]:
            try:
                click_text(driver, label)
                states.append(save_state(driver, name))
            except Exception as e:
                states.append({"name": name, "error": repr(e), "url": driver.current_url})

        # Inspect a line modal in transcription view.
        try:
            click_text(driver, "Transcription")
            buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label^='Open line']")
            if buttons:
                buttons[0].click()
                states.append(save_state(driver, "07_line_modal"))
        except Exception as e:
            states.append({"name": "07_line_modal", "error": repr(e), "url": driver.current_url})

        (OUT / "audit_states.json").write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {OUT}")
        for s in states:
            print(s.get("name"), s.get("url"), s.get("error", ""))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
