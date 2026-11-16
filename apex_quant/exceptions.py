"""Custom domain exceptions for APEX QUANT."""


class ApexError(Exception):
    """Base exception for all trading engine operational anomalies."""


class ConfigError(ApexError):
    """Raised when configuration values fail mathematical or structural validation."""


class DataFetchError(ApexError):
    """Raised when market data cannot be acquired, calibrated, or validated."""


class RiskViolationError(ApexError):
    """Raised when risk bounds or circuit breakers are breached."""


class StateError(ApexError):
    """Raised when state persistence encounters unrecoverable corruption or lock errors."""


class ExecutionError(ApexError):
    """Raised when order routing or notification delivery fails critically."""
