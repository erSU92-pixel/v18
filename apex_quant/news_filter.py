"""High-impact economic news blackout and XAUUSD daily rollover spread safety filter."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

from .config import UTC, Config
from .security import redact

_HIGH_IMPACT_KEYWORDS = (
    "FOMC",
    "FED",
    "POWELL",
    "NFP",
    "NON-FARM",
    "NONFARM",
    "CPI",
    "PCE",
    "PPI",
    "GDP",
    "ISM",
    "RETAIL SALES",
    "UNEMPLOYMENT",
    "INTEREST RATE",
    "PAYROLLS",
)


def parse_iso_utc(raw: str) -> datetime:
    """Parses ISO-8601 timestamps safely on Python 3.10+ (handles trailing 'Z')."""
    normalized = raw.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class NewsSafetyFilter:
    """Blocks new entries around high-impact USD/Gold events and daily broker rollover."""

    def __init__(self, cfg: Config, http: Any, kv: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.http = http
        self.kv = kv
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def fetch_events(self, now: Optional[datetime] = None) -> Tuple[List[dict], bool]:
        """Returns (events, fetch_ok) using KVStore cache to survive rate limits."""
        now = now or datetime.now(UTC)
        if self.kv is not None:
            cache = self.kv.get("news_cache", {}) or {}
            cached_at_str = cache.get("fetched_at", "")
            cached_events = cache.get("events")
            if cached_at_str and isinstance(cached_events, list):
                try:
                    cached_at = parse_iso_utc(cached_at_str)
                    if (now - cached_at).total_seconds() < self.cfg.news_cache_ttl_sec:
                        return cached_events, True
                except Exception:
                    pass

        try:
            r = self.http.request("GET", self.url)
            data = r.json()
            if not isinstance(data, list):
                return [], False
            if self.kv is not None:
                self.kv.set(
                    "news_cache",
                    {"fetched_at": now.isoformat(), "events": data},
                )
            return data, True
        except Exception as e:
            logging.warning(f"News calendar fetch failed: {redact(e)}.")
            # Return stale cache if available
            if self.kv is not None:
                cache = self.kv.get("news_cache", {}) or {}
                if isinstance(cache.get("events"), list) and cache["events"]:
                    logging.info("Using stale cached economic calendar due to network error.")
                    return cache["events"], True
            return [], False

    def is_rollover_blackout(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Protects against XAUUSD daily rollover spread widening (21:55–22:15 UTC)."""
        if self.cfg.skip_rollover_filter:
            return False, ""
        now = now or datetime.now(UTC)
        minutes_utc = now.hour * 60 + now.minute
        if (21 * 60 + 55) <= minutes_utc <= (22 * 60 + 15):
            return True, "XAUUSD Daily Rollover Spread Window (21:55–22:15 UTC)"
        return False, ""

    def is_blackout(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        now = now or datetime.now(UTC)

        rollover, rollover_msg = self.is_rollover_blackout(now)
        if rollover:
            return True, rollover_msg

        events, fetch_ok = self.fetch_events(now=now)
        if not fetch_ok and self.cfg.news_fail_closed:
            return True, "News calendar unreachable (fail-closed safety enabled)"

        window = timedelta(minutes=self.cfg.news_blackout_minutes)

        for ev in events:
            if not isinstance(ev, dict):
                continue
            country = str(ev.get("country", "")).upper()
            impact = str(ev.get("impact", ""))
            title = str(ev.get("title", "High-Impact Economic Release"))
            is_usd_high = country == "USD" and impact in ("High", "Holiday")
            is_gold_catalyst = country in ("USD", "ALL") and any(
                kw in title.upper() for kw in _HIGH_IMPACT_KEYWORDS
            ) and impact in ("High", "Medium")

            if not (is_usd_high or is_gold_catalyst):
                continue

            date_str = str(ev.get("date", "")).strip()
            if not date_str:
                continue
            try:
                ev_time = parse_iso_utc(date_str)
                if (ev_time - window) <= now <= (ev_time + window):
                    diff_min = int((ev_time - now).total_seconds() / 60)
                    if diff_min > 0:
                        status = f"in {diff_min}m"
                    elif diff_min == 0:
                        status = "NOW"
                    else:
                        status = f"{abs(diff_min)}m ago"
                    return True, f"{title} ({status})"
            except Exception:
                continue

        return False, ""
