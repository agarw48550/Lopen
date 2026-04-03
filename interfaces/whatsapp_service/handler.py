"""WhatsApp message polling and sending handler."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from interfaces.whatsapp_service.bridge import WhatsAppBridge, WhatsAppMessage

logger = logging.getLogger(__name__)


class WhatsAppHandler:
    """Polls for new messages and routes them through the Lopen agent."""

    def __init__(
        self,
        bridge: WhatsAppBridge,
        on_message: Optional[Callable[[WhatsAppMessage], Any]] = None,
        poll_interval: float = 30.0,
    ) -> None:
        self.bridge = bridge
        self.on_message = on_message
        self.poll_interval = poll_interval
        self._running = False
        logger.info("WhatsAppHandler initialised (poll_interval=%.0fs)", poll_interval)

    async def start(self) -> None:
        await self.bridge.start()
        self._running = True
        logger.info("WhatsApp polling started")
        await self._poll_loop()

    async def stop(self) -> None:
        self._running = False
        await self.bridge.stop()
        logger.info("WhatsApp polling stopped")

    async def send(self, contact: str, text: str) -> bool:
        return await self.bridge.send_message(contact, text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                messages = await self.bridge.poll_messages()
                for msg in messages:
                    await self._dispatch(msg)
            except Exception as exc:
                logger.error("WhatsApp poll error: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _dispatch(self, msg: WhatsAppMessage) -> None:
        logger.info("WhatsApp message from %r: %r", msg.contact, msg.text[:80])
        if self.on_message:
            try:
                result = self.on_message(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("on_message handler raised: %s", exc)
