"""Configuration management and strict runtime validation."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import timezone
from zoneinfo import ZoneInfo

from .exceptions import ConfigError

VERSION = "19.0-PRODUCTION"
NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
UTC = timezone.utc


@dataclass
class Config:
    # State Persistence
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v19.json"
    local_state_file: str = "apex_state_v19.json"
    lock_file: str = ".apex_engine.lock"
    journal_max_entries: int = 1000

    # Asset & Broker Parameters
    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # 1 Standard Lot = 100 Troy Ounces of Gold
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0
    max_broker_offset: float = 2.0

    # Execution & Broker Integration
    execution_mode: str = "signal_only"  # "signal_only" or "oanda"
    oanda_env: str = "practice"          # "practice" or "live"
    oanda_account_id: str = ""

    # Capital & Risk Management
    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01

    # Circuit Breakers & Drawdown Lock
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24
    max_daily_signals: int = 5
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120

    # Volatility, ATR Zones & Partial TP Weights
    atr_period: int = 14
    atr_sl_mult: float = 1.8
    min_sl_atr_mult: float = 0.8
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.4
    atr_tp3_mult: float = 4.0
    tp1_weight: float = 1.0 / 3.0
    tp2_weight: float = 1.0 / 3.0
    tp3_weight: float = 1.0 / 3.0
    spread_buffer: float = 0.35
    slippage_r_cost: float = 0.0
    min_atr_usd: float = 0.35
    max_atr_usd: float = 20.0

    # Market Data Constraints
    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    m5_fetch_size: int = 800
    max_stale_mult: float = 3.0

    # Entry & Limit Order Parameters
    entry_pullback_atr: float = 0.25
    entry_max_chase_atr: float = 2.0
    pending_stale_hours: float = 4.0
    max_unfilled_limits: int = 2

    # Filter Thresholds & Multi-Timeframe Rules
    adx_min: float = 20.0
    hurst_trend_min: float = 0.52
    hurst_meanrev_max: float = 0.48
    entropy_chaos_max: float = 0.85
    allow_ranging_regime: bool = True
    enforce_htf_filter: bool = True
    counter_trend_allow_ratio: float = 0.85
    news_blackout_minutes: int = 30
    news_cache_ttl_sec: int = 14400
    news_fail_closed: bool = False
    skip_rollover_filter: bool = False

    # Parliament Consensus Quorum
    parliament_min_active: int = 6
    parliament_full_ratio: float = 0.70
    parliament_half_ratio: float = 0.60
    parliament_min_ratio: float = 0.55

    # Operational Controls
    skip_killzone_filter: bool = False
    dry_run: bool = False
    kill_switch: bool = False
    enable_ml_models: bool = True
    ml_min_val_acc: float = 0.52

    # HTTP Network Resilience
    http_timeout_sec: float = 12.0
    http_max_retries: int = 3
    http_backoff_factor: float = 1.5
    http_max_retry_after_sec: float = 30.0
    td_min_interval_sec: float = 8.0

    @classmethod
    def from_env(cls) -> Config:
        """Hydrates configuration from environment variables with strict sanitization."""
        def env_str(key: str, default: str) -> str:
            return os.getenv(key, default).strip()

        def env_float(key: str, default: float) -> float:
            v = os.getenv(key, "").strip()
            if not v:
                return default
            try:
                parsed = float(v)
                if not math.isfinite(parsed):
                    raise ValueError("Non-finite float")
                return parsed
            except ValueError:
                logging.warning(f"Invalid float for {key}={v!r}, using default {default}")
                return default

        def env_int(key: str, default: int) -> int:
            v = os.getenv(key, "").strip()
            if not v:
                return default
            try:
                return int(v)
            except ValueError:
                logging.warning(f"Invalid int for {key}={v!r}, using default {default}")
                return default

        def env_bool(key: str, default: bool) -> bool:
            v = os.getenv(key, "").strip().lower()
            if not v:
                return default
            return v in ("1", "true", "yes", "on")

        cfg = cls()
        cfg.gist_id = env_str("GIST_ID", cfg.gist_id)
        cfg.gist_token = env_str("GIST_TOKEN", cfg.gist_token)
        cfg.gist_state_file = env_str("APEX_STATE_FILE", cfg.gist_state_file)
        cfg.local_state_file = env_str("APEX_LOCAL_STATE_FILE", cfg.local_state_file)
        cfg.symbol = env_str("APEX_SYMBOL", cfg.symbol)
        cfg.execution_mode = env_str("APEX_EXECUTION_MODE", cfg.execution_mode).lower()
        cfg.oanda_env = env_str("OANDA_ENV", cfg.oanda_env).lower()
        cfg.oanda_account_id = env_str("OANDA_ACCOUNT_ID", cfg.oanda_account_id)
        cfg.account_balance = env_float("APEX_BALANCE", cfg.account_balance)
        cfg.max_risk_pct_per_trade = env_float("APEX_RISK_PCT", cfg.max_risk_pct_per_trade)
        cfg.max_daily_loss_pct = env_float("APEX_MAX_DAILY_LOSS", cfg.max_daily_loss_pct)
        cfg.max_consecutive_losses = env_int("APEX_MAX_CONSEC_LOSSES", cfg.max_consecutive_losses)
        cfg.mt5_offset = env_float("APEX_MT5_OFFSET", cfg.mt5_offset)
        cfg.max_broker_offset = env_float("APEX_MAX_OFFSET", cfg.max_broker_offset)
        cfg.skip_killzone_filter = env_bool("APEX_SKIP_KILLZONE_FILTER", cfg.skip_killzone_filter)
        cfg.skip_rollover_filter = env_bool("APEX_SKIP_ROLLOVER_FILTER", cfg.skip_rollover_filter)
        cfg.dry_run = env_bool("APEX_DRY_RUN", cfg.dry_run)
        cfg.kill_switch = env_bool("APEX_KILL_SWITCH", cfg.kill_switch)
        cfg.enable_ml_models = env_bool("APEX_ENABLE_ML", cfg.enable_ml_models)
        cfg.allow_ranging_regime = env_bool("APEX_ALLOW_RANGING", cfg.allow_ranging_regime)
        cfg.enforce_htf_filter = env_bool("APEX_ENFORCE_HTF_FILTER", cfg.enforce_htf_filter)
        cfg.counter_trend_allow_ratio = env_float(
            "APEX_COUNTER_TREND_ALLOW_RATIO", cfg.counter_trend_allow_ratio
        )
        cfg.news_fail_closed = env_bool("APEX_NEWS_FAIL_CLOSED", cfg.news_fail_closed)
        return cfg

    def validate(self) -> None:
        """Enforces structural and mathematical boundaries (active even with python -O)."""
        if self.contract_size != 100.0:
            raise ConfigError("Gold contract size must be 100.0 troy oz per standard lot")
        if self.account_balance < 10.0:
            raise ConfigError("Account balance must be at least $10.00")
        if not (0.05 <= self.max_risk_pct_per_trade <= 5.0):
            raise ConfigError("Risk % per trade must be between 0.05% and 5.0%")
        if not (1.0 <= self.hard_leverage_cap <= 100.0):
            raise ConfigError("Hard leverage cap must be between 1.0 and 100.0")
        if not (1.0 <= self.max_daily_loss_pct <= 15.0):
            raise ConfigError("Daily loss limit must be between 1.0% and 15.0%")
        if self.max_consecutive_losses < 1:
            raise ConfigError("max_consecutive_losses must be >= 1")
        if not (0.01 <= self.min_lot <= self.max_lot):
            raise ConfigError("Invalid lot boundaries (0.01 <= min_lot <= max_lot)")
        if self.atr_sl_mult <= 0 or self.min_sl_atr_mult <= 0:
            raise ConfigError("SL ATR multipliers must be positive")
        if not (0 < self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult):
            raise ConfigError("TP ATR multipliers must satisfy 0 < tp1 < tp2 < tp3")
        weight_sum = self.tp1_weight + self.tp2_weight + self.tp3_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ConfigError(f"TP weights must sum to 1.0 (got {weight_sum:.6f})")
        if not (0.50 <= self.parliament_min_ratio <= self.parliament_half_ratio <= self.parliament_full_ratio <= 1.0):
            raise ConfigError("Invalid parliament ratio hierarchy")
        if self.execution_mode not in ("signal_only", "oanda"):
            raise ConfigError("execution_mode must be 'signal_only' or 'oanda'")
        if self.oanda_env not in ("practice", "live"):
            raise ConfigError("oanda_env must be 'practice' or 'live'")
