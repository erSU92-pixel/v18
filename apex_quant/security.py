"""Security and credential redaction utilities."""

from __future__ import annotations

import re
from typing import Any

_REDACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})", r"gh**_****"),
    (r"(bot[0-9]{6,}:[A-Za-z0-9_-]{20,})", r"bot****:****"),
    (r"((?:apikey|api_key|token|access_token|secret)=)[^&\s'\"]+", r"\1****"),
    (r"(Bearer\s+)[A-Za-z0-9_.\-]+", r"\1****"),
)


def redact(val: Any) -> str:
    """Masks secrets, tokens, and API credentials from application logs and alerts."""
    s = str(val)
    for pat, rep in _REDACTION_PATTERNS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s
