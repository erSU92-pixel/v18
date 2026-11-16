"""
APEX QUANT v19.0 — Institutional Quantitative Trading Engine for XAUUSD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import VERSION, Config

if TYPE_CHECKING:
    from .orchestrator import TradingBotOrchestrator

__all__ = ["Config", "TradingBotOrchestrator", "VERSION"]
__version__ = VERSION


def __getattr__(name: str) -> Any:
    if name == "TradingBotOrchestrator":
        from .orchestrator import TradingBotOrchestrator

        return TradingBotOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
