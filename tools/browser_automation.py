"""Browser automation tool using Playwright."""

from __future__ import annotations

import logging
from typing import Any, Optional

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, Page, Browser  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


class BrowserAutomation(BaseTool):
    """Playwright-based browser automation: navigate, click, type, extract text."""

    name = "browser_automation"
    description = "Browser automation: navigate URLs, click elements, extract page text."
    requires_permission = True

    def __init__(self, llm_adapter: Any | None = None, headless: bool = True) -> None:
        super().__init__(llm_adapter)
        self.headless = headless
        self._mock_mode = not _PLAYWRIGHT_AVAILABLE
        if self._mock_mode:
            logger.warning("BrowserAutomation: Playwright not installed — MOCK mode active")

    def run(self, query: str, **kwargs: Any) -> str:
        """High-level dispatch: parse action from query or kwargs."""
        action = kwargs.get("action", "navigate")
        url = kwargs.get("url", "")
        selector = kwargs.get("selector", "")
        text = kwargs.get("text", "")

        if self._mock_mode:
            return f"[BrowserAutomation MOCK] Would {action} on {url or query}. Install playwright for live automation."

        action_map = {
            "navigate": lambda: self.navigate(url or query),
            "extract": lambda: self.extract_text(url or query),
            "screenshot": lambda: self.screenshot(url or query, kwargs.get("output", "screenshot.png")),
        }
        fn = action_map.get(action)
        if fn:
            return fn()
        return f"Unknown action: {action}"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def navigate(self, url: str) -> str:
        """Navigate to a URL and return the page title."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            title = page.title()
            browser.close()
        logger.info("Navigated to %s — title: %r", url, title)
        return f"Navigated to: {url}\nPage title: {title}"

    def extract_text(self, url: str, selector: str = "body") -> str:
        """Extract text content from a URL."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            text = page.inner_text(selector)[:3000]
            browser.close()
        logger.info("Extracted %d chars from %s", len(text), url)
        return text

    def screenshot(self, url: str, output_path: str = "screenshot.png") -> str:
        """Take a screenshot of a URL."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
        logger.info("Screenshot saved: %s", output_path)
        return f"Screenshot saved to: {output_path}"

    def click_element(self, page: Any, selector: str) -> bool:
        """Click an element by CSS selector."""
        try:
            page.click(selector, timeout=10000)
            return True
        except Exception as exc:
            logger.error("Click failed on %r: %s", selector, exc)
            return False
