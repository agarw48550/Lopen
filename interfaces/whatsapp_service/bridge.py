"""WhatsApp Web automation bridge using Playwright."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

WHATSAPP_WEB_URL = "https://web.whatsapp.com"


@dataclass
class WhatsAppMessage:
    contact: str
    text: str
    timestamp: str = ""


class WhatsAppBridge:
    """Automates WhatsApp Web to poll and send messages."""

    def __init__(
        self,
        headless: bool = True,
        session_dir: str = "storage/whatsapp_session",
    ) -> None:
        self.headless = headless
        self.session_dir = session_dir
        self._playwright: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._context: Optional[Any] = None
        self._page: Optional[Any] = None
        self._mock_mode = not _PLAYWRIGHT_AVAILABLE
        logger.info("WhatsAppBridge initialised (mock=%s, headless=%s)", self._mock_mode, headless)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._mock_mode:
            logger.warning("Playwright not available — WhatsApp bridge in MOCK mode")
            return

        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox"],
        )
        self._context = await self._browser.new_context(
            storage_state=self._session_file() if self._session_exists() else None
        )
        self._page = await self._context.new_page()
        await self._page.goto(WHATSAPP_WEB_URL)
        logger.info("WhatsApp Web opened — waiting for QR scan if needed…")

    async def stop(self) -> None:
        if self._context:
            await self._context.storage_state(path=self._session_file())
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("WhatsAppBridge stopped")

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def poll_messages(self) -> list[WhatsAppMessage]:
        """Return a list of unread messages."""
        if self._mock_mode:
            logger.debug("[MOCK] poll_messages -> []")
            return []
        if self._page is None:
            logger.warning("WhatsApp page not initialised; call connect() first")
            return []
        # TODO: implement real DOM-based message polling via Playwright selectors
        logger.warning(
            "poll_messages is not yet implemented (stub). "
            "Real WhatsApp Web DOM scraping is required here."
        )
        return []

    async def send_message(self, contact: str, text: str) -> bool:
        """Send a message to a contact. Returns True on success."""
        if self._mock_mode:
            logger.info("[MOCK] Would send to %r: %r", contact, text)
            return True
        if self._page is None:
            logger.warning("WhatsApp page not initialised; call connect() first")
            return False
        # TODO: implement real Playwright message sending via WhatsApp Web UI
        logger.warning(
            "send_message is not yet implemented (stub). "
            "Real Playwright automation for WhatsApp Web is required here."
        )
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_file(self) -> str:
        import os
        os.makedirs(self.session_dir, exist_ok=True)
        return f"{self.session_dir}/session.json"

    def _session_exists(self) -> bool:
        import os
        return os.path.isfile(self._session_file())
