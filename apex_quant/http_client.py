"""Resilient HTTP client with connection pooling, Retry-After handling, and fast-fail 4xx."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

from .config import VERSION
from .exceptions import DataFetchError
from .security import redact

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class ResilientHttpClient:
    """Enterprise HTTP wrapper supporting pooling, Retry-After, exponential backoff, and jitter."""

    def __init__(
        self,
        timeout: float = 12.0,
        max_retries: int = 3,
        backoff: float = 1.5,
        max_retry_after_sec: float = 30.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff = backoff
        self.max_retry_after_sec = max_retry_after_sec
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {"User-Agent": f"ApexQuantInstitutional/{VERSION} (XAUUSD Quantitative Engine)"}
        )

    def __enter__(self) -> ResilientHttpClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            retry_after_delay: Optional[float] = None
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    if resp.status_code == 429:
                        ra = resp.headers.get("Retry-After", "").strip()
                        if ra.isdigit():
                            retry_after_delay = min(float(ra), self.max_retry_after_sec)
                    raise requests.HTTPError(
                        f"Retryable HTTP {resp.status_code}: {resp.text[:160]}",
                        response=resp,
                    )
                if 400 <= resp.status_code < 500:
                    # Non-retryable client error (400, 401, 403, 404, etc.) -> fail fast
                    raise DataFetchError(
                        f"Non-retryable HTTP {resp.status_code} on {method} {url}: "
                        f"{redact(resp.text[:160])}"
                    )
                resp.raise_for_status()
                return resp
            except DataFetchError:
                raise
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                sleep_time = (
                    retry_after_delay
                    if retry_after_delay is not None
                    else (self.backoff ** attempt + random.uniform(0.1, 0.4))
                )
                logging.warning(
                    f"[HTTP Retry {attempt}/{self.max_retries}] {method} {url} "
                    f"failed: {redact(e)}. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

        raise DataFetchError(
            f"HTTP Request failed after {self.max_retries} retries: {redact(last_exc)}"
        )
