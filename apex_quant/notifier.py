"""Telegram HTML notification dispatcher."""

from __future__ import annotations

import html
import logging
import os
from typing import Any

from .security import redact


class TelegramNotifier:
    """Transmits signals formatted for instantaneous manual execution in MT5 Mobile."""

    def __init__(self, http: Any, dry_run: bool = False) -> None:
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.http = http
        self.dry_run = dry_run

    @staticmethod
    def escape(text: str) -> str:
        return html.escape(str(text), quote=False)

    def send(self, message: str) -> bool:
        if self.dry_run or not (self.token and self.chat_id):
            logging.info(f"[DRY-RUN / TELEGRAM OUTPUT]\n{message}")
            return True

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            self.http.request("POST", url, json=payload)
            logging.info("Telegram alert dispatched successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to dispatch Telegram alert: {redact(e)}")
            return False
