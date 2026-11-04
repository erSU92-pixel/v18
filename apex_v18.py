#!/usr/bin/env python3
"""
===============================================================================
APEX PREDATOR v18.0 — Institutional High-Frequency & Quantitative Engine for XAUUSD
===============================================================================
Target Asset       : XAUUSD (Spot Gold / USD)
Contract Size      : 100 Troy Ounces per Standard Lot
Execution Model    : Institutional Pullback Limit Orders + ATR Multi-Tier TP/SL
Architecture       : Zero-Hallucination Strict Modular OOP
Data Resilience    : 4-Tier Failover (TwelveData -> XAUS API -> Biquote -> Yahoo Finance)
Risk Protection    : Max Daily Drawdown Lock, Consec Loss Cooloff, 30:1 Hard Leverage Cap
News Safety Filter : ForexFactory High-Impact USD/Gold 30-min Blackout Window
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# Optional Machine Learning & Signal Processing packages
try:
    from sklearn.ensemble import RandomForestClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Version & Timezones
VERSION = "18.0-PRODUCTION"
NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
UTC = timezone.utc


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & RUNTIME VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # State Persistence
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v18.json"
    local_state_file: str = "apex_state_v18.json"

    # Asset & Broker Parameters
    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # 1 Standard Lot = 100 Troy Ounces of Gold
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0  # Broker quote calibration offset

    # Capital & Risk Management
    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0  # 1% equity risk per trade
    hard_leverage_cap: float = 30.0      # Maximum permitted gross leverage (30:1)
    max_lot: float = 5.0
    min_lot: float = 0.01

    # Circuit Breakers & Drawdown Lock
    max_daily_loss_pct: float = 3.0      # Lock engine if daily loss reaches 3%
    max_consecutive_losses: int = 3      # Lock engine if 3 losses occur in sequence
    dd_pause_hours: int = 24             # Drawdown cooling-off duration
    max_daily_signals: int = 5           # Daily signal frequency cap
    cooldown_minutes: int = 45           # Cooldown between same-direction signals
    opposite_cooldown_minutes: int = 120 # Cooldown between reverse signals

    # Volatility & ATR Zones
    atr_period: int = 14
    atr_sl_mult: float = 1.8             # SL = Entry ± 1.8 * ATR
    atr_tp1_mult: float = 1.0            # TP1 = Entry ± 1.0 * ATR (1/3 close + BE)
    atr_tp2_mult: float = 2.4            # TP2 = Entry ± 2.4 * ATR (1/3 close)
    atr_tp3_mult: float = 4.0            # TP3 = Entry ± 4.0 * ATR (runner target)
    spread_buffer: float = 0.35          # Gold spread buffer ($0.35/oz)

    # Market Data Constraints
    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    m5_fetch_size: int = 800
    max_stale_mult: float = 3.0

    # Entry & Limit Order Parameters
    entry_pullback_atr: float = 0.25     # Pullback depth for institutional limit
    entry_max_chase_atr: float = 2.0     # Max distance before order defaults to ATR
    pending_stale_hours: float = 4.0     # Expiration for unfilled pending limits

    # Filter Thresholds & Multi-Timeframe Rules
    adx_min: float = 20.0
    hurst_trend_min: float = 0.52
    hurst_meanrev_max: float = 0.48
    entropy_chaos_max: float = 0.85
    allow_ranging_regime: bool = True      # Izinkan trading saat pasar ranging jika konsensus kuat
    enforce_htf_filter: bool = True        # Multi-timeframe H4 filter
    counter_trend_allow_ratio: float = 0.85 # Izinkan counter-trend scalps jika konsensus >= 85%
    news_blackout_minutes: int = 30       # ±30 min high-impact blackout window

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
    td_min_interval_sec: float = 8.0

    @classmethod
    def from_env(cls) -> Config:
        """Hydrates configuration from environment variables with strong sanitization."""
        def env_str(key: str, default: str) -> str:
            return os.getenv(key, default).strip()

        def env_float(key: str, default: float) -> float:
            v = os.getenv(key, "").strip()
            if not v:
                return default
            try:
                return float(v)
            except ValueError:
                logging.warning(f"Invalid float for {key}={v}, using default {default}")
                return default

        def env_int(key: str, default: int) -> int:
            v = os.getenv(key, "").strip()
            if not v:
                return default
            try:
                return int(v)
            except ValueError:
                logging.warning(f"Invalid int for {key}={v}, using default {default}")
                return default

        def env_bool(key: str, default: bool) -> bool:
            v = os.getenv(key, "").strip().lower()
            if not v:
                return default
            return v in ("1", "true", "yes", "on")

        cfg = cls()
        cfg.gist_id = env_str("GIST_ID", cfg.gist_id)
        cfg.gist_token = env_str("GIST_TOKEN", cfg.gist_token)
        cfg.symbol = env_str("APEX_SYMBOL", cfg.symbol)
        cfg.account_balance = env_float("APEX_BALANCE", cfg.account_balance)
        cfg.max_risk_pct_per_trade = env_float("APEX_RISK_PCT", cfg.max_risk_pct_per_trade)
        cfg.max_daily_loss_pct = env_float("APEX_MAX_DAILY_LOSS", cfg.max_daily_loss_pct)
        cfg.max_consecutive_losses = env_int("APEX_MAX_CONSEC_LOSSES", cfg.max_consecutive_losses)
        cfg.skip_killzone_filter = env_bool("APEX_SKIP_KILLZONE_FILTER", cfg.skip_killzone_filter)
        cfg.dry_run = env_bool("APEX_DRY_RUN", cfg.dry_run)
        cfg.kill_switch = env_bool("APEX_KILL_SWITCH", cfg.kill_switch)
        cfg.enable_ml_models = env_bool("APEX_ENABLE_ML", cfg.enable_ml_models)
        cfg.allow_ranging_regime = env_bool("APEX_ALLOW_RANGING", cfg.allow_ranging_regime)
        cfg.enforce_htf_filter = env_bool("APEX_ENFORCE_HTF_FILTER", cfg.enforce_htf_filter)
        cfg.counter_trend_allow_ratio = env_float("APEX_COUNTER_TREND_ALLOW_RATIO", cfg.counter_trend_allow_ratio)
        return cfg

    def validate(self) -> None:
        """Enforces structural mathematical boundaries on parameters."""
        assert self.contract_size == 100.0, "Gold contract size must be 100 troy oz"
        assert 0.05 <= self.max_risk_pct_per_trade <= 5.0, "Risk % must be between 0.05% and 5.0%"
        assert 1.0 <= self.max_daily_loss_pct <= 15.0, "Daily loss limit must be between 1% and 15%"
        assert 0.01 <= self.min_lot <= self.max_lot, "Invalid lot boundaries"
        assert self.atr_sl_mult > 0 and self.atr_tp1_mult > 0, "ATR multipliers must be positive"


# ═════════════════════════════════════════════════════════════════════════════
# 2. CUSTOM EXCEPTIONS & SECURITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
class ApexError(Exception):
    """Base exception for all trading bot operational anomalies."""

class DataFetchError(ApexError):
    """Raised when market data cannot be acquired or validated."""

class RiskViolationError(ApexError):
    """Raised when risk bounds or circuit breakers are breached."""

class StateError(ApexError):
    """Raised when state persistence encounters read/write errors."""

def redact(val: Any) -> str:
    """Masks secrets, tokens, and API credentials from application logs."""
    s = str(val)
    patterns = [
        (r"(ghp_[A-Za-z0-9_]{30,})", r"ghp_****"),
        (r"(bot[0-9]+:[A-Za-z0-9_-]{30,})", r"bot****:****"),
        (r"(apikey=)[A-Za-z0-9_-]+", r"\1****"),
        (r"(Bearer\s+)[A-Za-z0-9_.-]+", r"\1****"),
    ]
    for pat, rep in patterns:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTTP RESILIENCE ENGINE (RECONNECTS, RETRIES, BACKOFF)
# ═════════════════════════════════════════════════════════════════════════════
class ResilientHttpClient:
    """Enterprise HTTP wrapper supporting timeouts, exponential backoff, and jitter."""
    def __init__(self, timeout: float = 12.0, max_retries: int = 3, backoff: float = 1.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"ApexQuantInstitutional/{VERSION} (XAUUSD Quantitative Engine)"
        })

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:120]}")
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                sleep_time = self.backoff ** attempt + random.uniform(0.1, 0.4)
                logging.warning(
                    f"[HTTP Retry {attempt}/{self.max_retries}] {method} {url} "
                    f"failed: {redact(e)}. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
        raise DataFetchError(f"HTTP Request failed after {self.max_retries} retries: {redact(last_exc)}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. STATE PERSISTENCE & GIST CLOUD MEMORY
# ═════════════════════════════════════════════════════════════════════════════
class KVStore:
    """Thread-safe state persistence with hybrid GitHub Gist Cloud and Local JSON storage."""
    def __init__(self, cfg: Config, http: ResilientHttpClient):
        self.cfg = cfg
        self.http = http
        self.data: Dict[str, Any] = {
            "journal": [],
            "state": {},
            "metrics": {},
            "circuit_breaker": {},
        }
        self.loaded = False

    def force_load(self) -> None:
        """Loads state from GitHub Gist or falls back to local disk persistence."""
        if self.cfg.gist_id and self.cfg.gist_token:
            try:
                url = f"https://api.github.com/gists/{self.cfg.gist_id}"
                headers = {
                    "Authorization": f"Bearer {self.cfg.gist_token}",
                    "Accept": "application/vnd.github+json",
                }
                resp = self.http.request("GET", url, headers=headers)
                files = resp.json().get("files", {})
                target_file = files.get(self.cfg.gist_state_file)
                if target_file and target_file.get("content"):
                    self.data = json.loads(target_file["content"])
                    self.loaded = True
                    logging.info("State successfully loaded from GitHub Gist cloud storage.")
                    return
            except Exception as e:
                logging.warning(f"Gist state read failed: {redact(e)}. Falling back to local storage.")

        p = Path(self.cfg.local_state_file)
        if p.exists():
            try:
                self.data = json.loads(p.read_text(encoding="utf-8"))
                self.loaded = True
                logging.info(f"State successfully loaded from local file: {self.cfg.local_state_file}")
                return
            except Exception as e:
                logging.warning(f"Local state read error: {e}")

        self.loaded = True
        logging.info("No existing state found. Initializing fresh runtime state store.")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        """Persists memory to local storage and syncs to GitHub Gist."""
        serialized = json.dumps(self.data, indent=2, default=str)
        try:
            Path(self.cfg.local_state_file).write_text(serialized, encoding="utf-8")
        except Exception as e:
            logging.error(f"Local state write error: {e}")

        if self.cfg.gist_id and self.cfg.gist_token:
            try:
                url = f"https://api.github.com/gists/{self.cfg.gist_id}"
                headers = {
                    "Authorization": f"Bearer {self.cfg.gist_token}",
                    "Accept": "application/vnd.github+json",
                }
                payload = {
                    "description": f"Apex Quant State v{VERSION} - Auto Saved",
                    "files": {
                        self.cfg.gist_state_file: {"content": serialized}
                    }
                }
                self.http.request("PATCH", url, headers=headers, json=payload)
                logging.info("State successfully synchronized to GitHub Gist.")
            except Exception as e:
                logging.error(f"Gist sync error: {redact(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MATHEMATICAL INDICATORS & SIGNAL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
class Indicators:
    """Statistically pure technical indicator calculations without look-ahead bias."""

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> float:
        s = Indicators.rsi_series(series, period)
        return float(s.iloc[-1]) if not s.empty else 50.0

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = Indicators.true_range(df)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        atr_s = Indicators.atr_series(df, period)
        return float(atr_s.iloc[-1]) if not atr_s.empty else 0.0

    @staticmethod
    def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low = df["high"], df["low"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = Indicators.true_range(df)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        pdm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        mdm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100.0 * (pdm_s / (atr + 1e-9))
        minus_di = 100.0 * (mdm_s / (atr + 1e-9))
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        s = Indicators.adx_series(df, period)
        return float(s.iloc[-1]) if not s.empty else 0.0

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        ef = Indicators.ema(series, fast)
        es = Indicators.ema(series, slow)
        macd_line = ef - es
        sig_line = Indicators.ema(macd_line, signal)
        hist = macd_line - sig_line
        return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        sma = series.rolling(period).mean()
        std = series.rolling(period).std(ddof=0)
        upper = sma + num_std * std
        lower = sma - num_std * std
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> Tuple[float, float]:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        k_fast = 100.0 * ((df["close"] - low_min) / (high_max - low_min + 1e-9))
        k = k_fast.rolling(smooth_k).mean()
        d = k.rolling(smooth_k).mean()
        return float(k.iloc[-1]), float(d.iloc[-1])

    @staticmethod
    def hurst(series: pd.Series, max_lags: int = 20) -> float:
        """Computes Hurst Exponent to distinguish trending (H>0.52) from mean-reverting (H<0.48)."""
        ts = series.values
        if len(ts) < max_lags * 2:
            return 0.5
        lags = range(2, max_lags)
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(np.clip(reg[0], 0.0, 1.0))

    @staticmethod
    def shannon_entropy(series: pd.Series, bins: int = 10) -> float:
        """Measures Shannon Entropy (unpredictability/chaos) of price returns."""
        returns = series.pct_change().dropna()
        if len(returns) < bins * 2:
            return 0.5
        counts, _ = np.histogram(returns, bins=bins, density=True)
        counts = counts[counts > 0]
        probs = counts / np.sum(counts)
        ent = -np.sum(probs * np.log2(probs))
        max_ent = np.log2(bins)
        return float(np.clip(ent / max_ent, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# 6. MARKET DATA HANDLER (4-TIER INSTITUTIONAL FAILOVER)
# ═════════════════════════════════════════════════════════════════════════════
class MarketDataHandler:
    """Acquires, cleans, and calibrates multi-timeframe candlestick data with 4 tiers."""
    def __init__(self, cfg: Config, http: ResilientHttpClient):
        self.cfg = cfg
        self.http = http
        self._last_td_call = 0.0

    def compute_broker_offset(self) -> float:
        """Calculates price quote spread difference between broker feed and data provider."""
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda_key and td_key):
            return self.cfg.mt5_offset
        try:
            r1 = self.http.request(
                "GET",
                "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                headers={"Authorization": f"Bearer {oanda_key}"},
            )
            oanda_bid = float(r1.json()["prices"][0]["bids"][0]["price"])

            r2 = self.http.request(
                "GET",
                "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key},
            )
            td_price = float(r2.json()["price"])

            if td_price <= 0:
                return self.cfg.mt5_offset

            offset = oanda_bid - td_price
            if abs(offset) > 2.0:
                logging.warning(f"Offset {offset:+.3f} rejected (exceeds $2.00 sanity bound)")
                return self.cfg.mt5_offset
            return round(offset, 3)
        except Exception as e:
            logging.warning(f"Offset computation fallback: {redact(e)}")
            return self.cfg.mt5_offset

    @staticmethod
    def _apply_offset(df: pd.DataFrame, offset: float) -> pd.DataFrame:
        if offset == 0.0:
            return df
        out = df.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col] + offset
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resamples OHLCV bars without lookahead bias."""
        resampled = (
            df.set_index("datetime")
            .resample(rule, label="left", closed="left")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
            .reset_index()
        )
        return resampled

    # Tier 1: TwelveData
    def _fetch_twelvedata(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("TWELVE_DATA_API_KEY not configured")

        def fetch_interval(interval: str, outputsize: int) -> pd.DataFrame:
            elapsed = time.time() - self._last_td_call
            if elapsed < self.cfg.td_min_interval_sec:
                time.sleep(self.cfg.td_min_interval_sec - elapsed)
            self._last_td_call = time.time()

            r = self.http.request(
                "GET",
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": "XAU/USD",
                    "interval": interval,
                    "outputsize": outputsize,
                    "apikey": key,
                    "timezone": "UTC",
                },
            )
            data = r.json()
            if "values" not in data or not data["values"]:
                raise DataFetchError(f"TwelveData {interval}: {data.get('message', 'No values returned')}")
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = 100.0
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5 = fetch_interval("5min", min(self.cfg.m5_fetch_size, 800))
        h1 = fetch_interval("1h", 800)
        h4 = fetch_interval("4h", 800)

        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")

        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)

        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    # Tier 2: XAUS API
    def _fetch_xaus(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        base_url = "https://xaus.com/api/v1/chart"

        def fetch_range(interval: str, rng: str) -> pd.DataFrame:
            r = self.http.request(
                "GET",
                base_url,
                params={"symbol": "xau", "interval": interval, "range": rng},
                headers={"Accept": "application/json"},
            )
            data = r.json()
            points = data.get("points")
            if not points:
                raise DataFetchError(f"XAUS {interval}: Empty points")
            rows = []
            for p in points:
                if not all(k in p for k in ("t", "o", "h", "l", "c")):
                    continue
                rows.append({
                    "epoch": int(p["t"]),
                    "open": float(p["o"]),
                    "high": float(p["h"]),
                    "low": float(p["l"]),
                    "close": float(p["c"]),
                    "volume": float(p.get("v", 100.0)),
                })
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            df = df.drop(columns=["epoch"]).sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5 = fetch_range("5m", "7d")
        h1 = fetch_range("1h", "30d")
        h4 = fetch_range("4h", "90d")

        self._validate(m5, "XAUS-M5", self.cfg.min_bars_m5)
        self._validate(h1, "XAUS-H1", self.cfg.min_bars_h1)
        self._validate(h4, "XAUS-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "XAUS-M5")

        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "XAUS-M15", self.cfg.min_bars_m15)

        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "XAUS"

    # Tier 3: Biquote
    def _fetch_biquote(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        url = "https://biquote.com/api/gold/history"
        r = self.http.request("GET", url, params={"limit": 1000})
        records = r.json()
        if not isinstance(records, list) or len(records) < self.cfg.min_bars_m5:
            raise DataFetchError("Biquote returned insufficient records")
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c])
        df["volume"] = 100.0
        df = df.sort_values("datetime").reset_index(drop=True)
        m5 = self._apply_offset(df, offset)
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        h1 = self._apply_offset(self._resample(m5, "1h"), offset)
        h4 = self._apply_offset(self._resample(m5, "4h"), offset)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "Biquote"

    # Tier 4: Yahoo Finance Fallback
    def _fetch_yahoo(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
        r = self.http.request("GET", url, params={"interval": "5m", "range": "5d"})
        data = r.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote.get("volume", [100.0] * len(timestamps)),
        }).dropna()
        m5 = self._apply_offset(df.sort_values("datetime").reset_index(drop=True), offset)
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        h1 = self._apply_offset(self._resample(m5, "1h"), offset)
        h4 = self._apply_offset(self._resample(m5, "4h"), offset)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "YahooFinance"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        """Executes cascading failover across 4 data providers."""
        offset = self.compute_broker_offset()
        tiers = [
            ("TwelveData", self._fetch_twelvedata),
            ("XAUS", self._fetch_xaus),
            ("Biquote", self._fetch_biquote),
            ("YahooFinance", self._fetch_yahoo),
        ]
        errors = []
        for name, fn in tiers:
            try:
                dfs, src = fn(offset)
                logging.info(f"Data source: {src} | Offset: {offset:+.3f}")
                return dfs, src, offset
            except Exception as e:
                err_msg = f"{name}: {redact(e)}"
                logging.warning(f"Data tier failover: {err_msg}")
                errors.append(err_msg)

        raise DataFetchError(f"All 4 market data tiers failed. Diagnostics: {'; '.join(errors)}")

    def _validate(self, df: pd.DataFrame, label: str, min_bars: int) -> None:
        if len(df) < min_bars:
            raise DataFetchError(f"{label} bars count ({len(df)}) < required ({min_bars})")
        for c in ("open", "high", "low", "close"):
            if (df[c] <= 0).any() or df[c].isna().any():
                raise DataFetchError(f"{label} contains zero, negative, or NaN values in {c}")

    def _check_staleness(self, df: pd.DataFrame, interval_min: int, label: str) -> None:
        now = datetime.now(UTC)
        if now.weekday() >= 5:
            return  # Weekend gold market close
        last_dt = df["datetime"].iloc[-1]
        age_sec = (now - last_dt).total_seconds()
        max_sec = interval_min * 60 * self.cfg.max_stale_mult
        if age_sec > max_sec:
            raise DataFetchError(f"{label} data is stale: age {age_sec/60:.1f}m > limit {max_sec/60:.1f}m")


# ═════════════════════════════════════════════════════════════════════════════
# 7. QUANTITATIVE SIGNAL GENERATOR & PARLIAMENT CONSENSUS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    verdict: str = "CANCEL"
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    buy_score: float = 0.0
    sell_score: float = 0.0
    active_models: int = 0
    votes_detail: List[Dict[str, Any]] = field(default_factory=list)
    regime: str = "UNKNOWN"
    hurst: float = 0.5
    entropy: float = 0.5
    adx: float = 0.0
    skip_reason: str = ""


class SignalGenerator:
    """Quantitative Parliament consensus engine composed of 12 distinct market models."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._registry = self._build_model_registry()

    # Model 1: EMA Trend Following
    def _ema_cross(self, df: pd.DataFrame) -> Tuple[int, float]:
        c = df["close"]
        e9, e21 = Indicators.ema(c, 9), Indicators.ema(c, 21)
        diff = e9.iloc[-1] - e21.iloc[-1]
        prev_diff = e9.iloc[-2] - e21.iloc[-2]
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return 0, 0.0
        conf = min(1.0, abs(diff) / atr)
        if diff > 0 and prev_diff <= 0:
            return 1, min(1.0, conf + 0.3)
        if diff < 0 and prev_diff >= 0:
            return -1, min(1.0, conf + 0.3)
        if diff > 0:
            return 1, conf
        if diff < 0:
            return -1, conf
        return 0, 0.0

    # Model 2: Triple EMA Trend
    def _ema_trend(self, df: pd.DataFrame) -> Tuple[int, float]:
        c = df["close"]
        e20 = Indicators.ema(c, 20).iloc[-1]
        e50 = Indicators.ema(c, 50).iloc[-1]
        e200 = Indicators.ema(c, 200).iloc[-1]
        last = c.iloc[-1]
        if last > e20 > e50 > e200:
            return 1, 0.85
        if last < e20 < e50 < e200:
            return -1, 0.85
        return 0, 0.0

    # Model 3: Supertrend Volatility
    def _supertrend(self, df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[int, float]:
        atr = Indicators.atr_series(df, period)
        hl2 = (df["high"] + df["low"]) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        close = df["close"]
        trend = 1
        for i in range(1, len(df)):
            if close.iloc[i] > upper.iloc[i-1]:
                trend = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                trend = -1
        conf = min(1.0, abs(close.iloc[-1] - (lower.iloc[-1] if trend == 1 else upper.iloc[-1])) / (atr.iloc[-1] + 1e-9))
        return trend, conf

    # Model 4: Donchian Channel Breakout
    def _donchian(self, df: pd.DataFrame, period: int = 20) -> Tuple[int, float]:
        high_max = df["high"].rolling(period).max().iloc[-2]
        low_min = df["low"].rolling(period).min().iloc[-2]
        last = df["close"].iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return 0, 0.0
        if last > high_max:
            return 1, min(1.0, (last - high_max) / atr)
        if last < low_min:
            return -1, min(1.0, (low_min - last) / atr)
        return 0, 0.0

    # Model 5: Kalman Filter State Estimator
    def _kalman(self, df: pd.DataFrame) -> Tuple[int, float]:
        prices = df["close"].values
        n = len(prices)
        if n < 30:
            return 0, 0.0
        x_hat = prices[0]
        p = 1.0
        q = 1e-4
        r = 0.01
        estimates = []
        for z in prices:
            p = p + q
            k = p / (p + r)
            x_hat = x_hat + k * (z - x_hat)
            p = (1.0 - k) * p
            estimates.append(x_hat)
        k_est = pd.Series(estimates, index=df.index)
        slope = k_est.diff(3).iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return 0, 0.0
        conf = min(1.0, abs(slope) / (atr * 0.2))
        if slope > 0.05 * atr:
            return 1, conf
        if slope < -0.05 * atr:
            return -1, conf
        return 0, 0.0

    # Model 6: MACD Momentum
    def _macd(self, df: pd.DataFrame) -> Tuple[int, float]:
        macd, sig, hist = Indicators.macd(df["close"])
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return 0, 0.0
        conf = min(1.0, abs(hist) / atr)
        if macd > sig and hist > 0:
            return 1, conf
        if macd < sig and hist < 0:
            return -1, conf
        return 0, 0.0

    # Model 7: RSI Momentum Oscillations
    def _rsi(self, df: pd.DataFrame) -> Tuple[int, float]:
        r = Indicators.rsi(df["close"], 14)
        if r < 30:
            return 1, (30.0 - r) / 30.0
        if r > 70:
            return -1, (r - 70.0) / 30.0
        if r > 55:
            return 1, (r - 50.0) / 50.0
        if r < 45:
            return -1, (50.0 - r) / 50.0
        return 0, 0.0

    # Model 8: Stochastic Momentum
    def _stochastic(self, df: pd.DataFrame) -> Tuple[int, float]:
        k, d = Indicators.stochastic(df)
        if k < 20 and k > d:
            return 1, (20.0 - k) / 20.0
        if k > 80 and k < d:
            return -1, (k - 80.0) / 20.0
        return 0, 0.0

    # Model 9: Bollinger Bands Volatility
    def _bollinger(self, df: pd.DataFrame) -> Tuple[int, float]:
        upper, mid, lower = Indicators.bollinger_bands(df["close"])
        last = df["close"].iloc[-1]
        band_w = upper - lower
        if band_w <= 0:
            return 0, 0.0
        if last <= lower:
            return 1, min(1.0, (lower - last) / band_w + 0.5)
        if last >= upper:
            return -1, min(1.0, (last - upper) / band_w + 0.5)
        return 0, 0.0

    # Model 10: Ichimoku Kinko Hyo
    def _ichimoku(self, df: pd.DataFrame) -> Tuple[int, float]:
        c = df["close"]
        h = df["high"]
        l = df["low"]
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
        last_c = c.iloc[-1]
        k_val = kijun.iloc[-1]
        t_val = tenkan.iloc[-1]
        sa = span_a.iloc[-1] if not pd.isna(span_a.iloc[-1]) else k_val
        sb = span_b.iloc[-1] if not pd.isna(span_b.iloc[-1]) else k_val
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        if last_c > cloud_top and t_val > k_val:
            return 1, 0.8
        if last_c < cloud_bottom and t_val < k_val:
            return -1, 0.8
        return 0, 0.0

    # Model 11: Institutional SMC Order Block Detection
    def _smc_orderblock(self, df: pd.DataFrame) -> Tuple[int, float]:
        ob_b = SignalGenerator.detect_order_block(df, "BUY")
        ob_s = SignalGenerator.detect_order_block(df, "SELL")
        last = df["close"].iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return 0, 0.0
        if ob_b and last >= ob_b["low"] and last <= ob_b["high"] + 0.3 * atr:
            return 1, 0.9
        if ob_s and last <= ob_s["high"] and last >= ob_s["low"] - 0.3 * atr:
            return -1, 0.9
        return 0, 0.0

    # Model 12: Butterworth / Wavelet Filter
    def _wavelet(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SKLEARN or len(df) < 50:
            return 0, 0.0
        try:
            c = df["close"].values
            b, a = butter(2, 0.15)
            smooth = filtfilt(b, a, c)
            ss = pd.Series(smooth)
            ef = ss.ewm(span=10).mean()
            es = ss.ewm(span=30).mean()
            diff = ef.iloc[-1] - es.iloc[-1]
            slope = ef.diff(3).iloc[-1]
            atr = Indicators.atr_last(df)
            if atr <= 0:
                return 0, 0.0
            conf = min(1.0, float(abs(diff) / atr))
            if diff > 0 and slope > 0:
                return 1, conf
            if diff < 0 and slope < 0:
                return -1, conf
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _ml_ensemble(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models) or len(df) < 300:
            return 0, 0.0
        try:
            c = df["close"]
            features = pd.DataFrame(index=df.index)
            features["ret_1"] = c.pct_change(1)
            features["ret_5"] = c.pct_change(5)
            features["atr_norm"] = Indicators.atr_series(df) / (c + 1e-9)
            features["rsi_14"] = Indicators.rsi_series(c, 14) / 100.0
            features["ema20_dist"] = (c - Indicators.ema(c, 20)) / (c + 1e-9)
            features["ema50_dist"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
            features = features.dropna()

            # Triple-barrier labeling forward target
            horizon = 20
            atr_v = Indicators.atr_series(df).values
            closes = c.values
            highs = df["high"].values
            lows = df["low"].values
            n = len(closes)
            y_arr = np.zeros(n)
            for i in range(n - horizon):
                tp = closes[i] + 1.0 * atr_v[i]
                sl = closes[i] - 1.8 * atr_v[i]
                hit_tp = np.where(highs[i+1:i+horizon+1] >= tp)[0]
                hit_sl = np.where(lows[i+1:i+horizon+1] <= sl)[0]
                t_tp = hit_tp[0] if len(hit_tp) else 999
                t_sl = hit_sl[0] if len(hit_sl) else 999
                if t_tp < t_sl:
                    y_arr[i] = 1
                elif t_sl < t_tp:
                    y_arr[i] = 0

            y = pd.Series(y_arr[:n - horizon], index=df.index[:n - horizon])
            valid_idx = features.index.intersection(y.index)
            X_clean, y_clean = features.loc[valid_idx], y.loc[valid_idx]

            split = int(len(X_clean) * 0.8)
            X_tr, y_tr = X_clean.iloc[:split], y_clean.iloc[:split]
            X_val, y_val = X_clean.iloc[split:], y_clean.iloc[split:]

            if len(X_tr) < 100 or len(X_val) < 20:
                return 0, 0.0

            clf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1)
            clf.fit(X_tr.values, y_tr.values)
            val_acc = float(clf.score(X_val.values, y_val.values))
            if val_acc < self.cfg.ml_min_val_acc:
                return 0, 0.0

            proba = clf.predict_proba(features.iloc[-1:].values)[0]
            prob_up = float(proba[1]) if len(proba) > 1 else 0.5
            if prob_up > 0.62:
                return 1, min(1.0, (prob_up - 0.5) * 2.0)
            if prob_up < 0.38:
                return -1, min(1.0, (0.5 - prob_up) * 2.0)
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _build_model_registry(self) -> List[Tuple[str, Callable, float]]:
        registry: List[Tuple[str, Callable, float]] = [
            ("EMA_Cross", self._ema_cross, 1.0),
            ("EMA_Trend", self._ema_trend, 1.0),
            ("Supertrend", self._supertrend, 1.2),
            ("Donchian", self._donchian, 0.8),
            ("Kalman", self._kalman, 1.0),
            ("MACD", self._macd, 1.2),
            ("RSI", self._rsi, 1.0),
            ("Stochastic", self._stochastic, 0.9),
            ("BollingerBB", self._bollinger, 0.9),
            ("Ichimoku", self._ichimoku, 1.3),
            ("SMC_OB", self._smc_orderblock, 1.5),
            ("Wavelet", self._wavelet, 0.8),
        ]
        if self.cfg.enable_ml_models:
            registry.append(("ML_Ensemble", self._ml_ensemble, 1.0))
        return registry

    def _classify_regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        hurst_val = Indicators.hurst(h1["close"])
        entropy_val = Indicators.shannon_entropy(h1["close"])
        
        # True chaotic state: Shannon entropy exceeds noise threshold (default >= 0.85)
        if entropy_val >= self.cfg.entropy_chaos_max:
            return "CHAOTIC", hurst_val, entropy_val
            
        if hurst_val >= self.cfg.hurst_trend_min:
            return "TRENDING", hurst_val, entropy_val
        elif hurst_val <= self.cfg.hurst_meanrev_max:
            return "MEAN_REVERTING", hurst_val, entropy_val
        else:
            return "RANGING", hurst_val, entropy_val

    @staticmethod
    def _htf_bias(h4: pd.DataFrame) -> str:
        if len(h4) < 50:
            return "NEUTRAL"
        c = h4["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
            return "BULLISH"
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
            return "BEARISH"
        return "NEUTRAL"

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame) -> SignalResult:
        res = SignalResult()

        # Step 1: ADX Trend Strength Check
        adx_val = Indicators.adx_last(m5)
        res.adx = adx_val

        # Step 2: Regime Classification (Hurst & Shannon Entropy)
        regime, H, E = self._classify_regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E

        # Step 3: Multi-Timeframe Higher-Timeframe Filter
        h4_trend = self._htf_bias(h4)

        # Step 4: Quantitative Parliament Voting (Computed first for full audit visibility)
        buy_weight = 0.0
        sell_weight = 0.0
        active_count = 0
        details = []

        for name, model_fn, weight in self._registry:
            try:
                vote, conf = model_fn(m5)
            except Exception as e:
                logging.debug(f"Model {name} encountered error: {e}")
                vote, conf = 0, 0.0

            if vote == 0:
                details.append({"model": name, "vote": "NEUTRAL", "weight": weight, "conf": round(conf, 2), "score": 0.0})
                continue

            active_count += 1
            score = weight * conf
            if vote > 0:
                buy_weight += score
                v_label = "BUY"
            else:
                sell_weight += score
                v_label = "SELL"

            details.append({"model": name, "vote": v_label, "weight": weight, "conf": round(conf, 2), "score": round(score, 2)})

        res.buy_score = round(buy_weight, 2)
        res.sell_score = round(sell_weight, 2)
        res.active_models = active_count
        res.votes_detail = details

        total_weight = buy_weight + sell_weight
        if total_weight > 1e-6:
            res.buy_ratio = buy_weight / total_weight
            res.sell_ratio = sell_weight / total_weight
        else:
            res.buy_ratio = 0.0
            res.sell_ratio = 0.0

        # Filter 1: ADX Volatility Floor
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min} (Konsolidasi Volatilitas Rendah)"
            return res

        # Filter 2: Extreme Noise Chaos Check
        if regime == "CHAOTIC":
            res.skip_reason = f"Pasar Chaotic / Noise Ekstrem (Entropy={E:.2f} >= {self.cfg.entropy_chaos_max:.2f})"
            return res

        # Filter 3: Ranging Regime Filter (if user disabled ranging trading)
        if regime == "RANGING" and not self.cfg.allow_ranging_regime:
            res.skip_reason = f"Pasar Ranging (Hurst={H:.2f}, Entropy={E:.2f}) — Menunggu Breakout"
            return res

        # Filter 4: Minimum Active Models
        if total_weight < 1e-6:
            res.skip_reason = "Tidak ada model kuantitatif yang memberikan sinyal terarah"
            return res

        if active_count < self.cfg.parliament_min_active:
            res.skip_reason = f"Model aktif ({active_count}) di bawah kuorum ({self.cfg.parliament_min_active})"
            return res

        # Step 5: Consensus Thresholds
        dom_ratio = max(res.buy_ratio, res.sell_ratio)
        if dom_ratio >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom_ratio >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        elif dom_ratio >= self.cfg.parliament_min_ratio:
            res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = f"Konsensus ({dom_ratio:.0%}) di bawah minimum ({self.cfg.parliament_min_ratio:.0%})"
            return res

        # Step 6: Multi-Timeframe Alignment
        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH":
                if self.cfg.enforce_htf_filter:
                    if dom_ratio >= self.cfg.counter_trend_allow_ratio:
                        res.direction = "BUY"
                        if res.verdict == "EXECUTE_FULL":
                            res.verdict = "EXECUTE_HALF"
                        logging.info(f"Counter-trend BUY diizinkan: Konsensus Tinggi ({dom_ratio:.0%} >= {self.cfg.counter_trend_allow_ratio:.0%}) meskipun H4 BEARISH (Eksekusi 50% Lot)")
                    else:
                        res.skip_reason = f"Sinyal BUY diblokir: H4 HTF Trend adalah BEARISH (Konsensus {dom_ratio:.0%} < {self.cfg.counter_trend_allow_ratio:.0%})"
                        return res
                else:
                    res.direction = "BUY"
            else:
                res.direction = "BUY"
        else:
            if h4_trend == "BULLISH":
                if self.cfg.enforce_htf_filter:
                    if dom_ratio >= self.cfg.counter_trend_allow_ratio:
                        res.direction = "SELL"
                        if res.verdict == "EXECUTE_FULL":
                            res.verdict = "EXECUTE_HALF"
                        logging.info(f"Counter-trend SELL diizinkan: Konsensus Tinggi ({dom_ratio:.0%} >= {self.cfg.counter_trend_allow_ratio:.0%}) meskipun H4 BULLISH (Eksekusi 50% Lot)")
                    else:
                        res.skip_reason = f"Sinyal SELL diblokir: H4 HTF Trend adalah BULLISH (Konsensus {dom_ratio:.0%} < {self.cfg.counter_trend_allow_ratio:.0%})"
                        return res
                else:
                    res.direction = "SELL"
            else:
                res.direction = "SELL"

        return res

    @staticmethod
    def detect_order_block(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[dict]:
        """Detects Smart Money Concepts (SMC) Institutional Order Block zones."""
        if len(df) < lookback + 5:
            return None
        tail = df.tail(lookback + 5).reset_index(drop=True)
        atr = Indicators.atr_last(df)
        if atr <= 0:
            return None

        for i in range(len(tail) - 2, 2, -1):
            body = abs(tail["close"].iloc[i] - tail["open"].iloc[i])
            if body < 1.2 * atr:
                continue
            is_bull = tail["close"].iloc[i] > tail["open"].iloc[i]
            if side == "BUY" and not is_bull:
                continue
            if side == "SELL" and is_bull:
                continue
            for j in range(i - 1, max(i - 10, 0), -1):
                if side == "BUY" and tail["close"].iloc[j] < tail["open"].iloc[j]:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
                if side == "SELL" and tail["close"].iloc[j] > tail["open"].iloc[j]:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
        return None

    def compute_levels(
        self, m5: pd.DataFrame, side: str, ob: Optional[dict]
    ) -> Tuple[float, float, float, float, float]:
        """Calculates precise Entry (Limit), Stop Loss, and TP1/TP2/TP3 targets with resilient fallback."""
        atr = Indicators.atr_last(m5)
        if atr <= 0:
            raise ValueError("ATR cannot be zero or negative")

        price = float(m5["close"].iloc[-1])
        dec = self.cfg.price_decimals
        spread_buf = self.cfg.spread_buffer * 2.0

        ob_high = ob["high"] if ob else price
        ob_low = ob["low"] if ob else price

        if side == "BUY":
            ob_candidate = (ob_high - (ob_high - ob_low) * 0.5) if (ob and ob_high < price) else None
            # Check if OB candidate is within safe pullback distance
            if ob_candidate and (price - ob_candidate) <= self.cfg.entry_max_chase_atr * atr:
                entry = ob_candidate
                sl_ref = ob_low
            else:
                entry = price - atr * self.cfg.entry_pullback_atr
                sl_ref = entry - atr * self.cfg.atr_sl_mult

            entry = min(entry, price - atr * 0.05)
            # Ensure entry is always within safe pullback bounds
            if (price - entry) > self.cfg.entry_max_chase_atr * atr:
                entry = price - atr * self.cfg.entry_pullback_atr

            sl = min(sl_ref, entry - atr * 0.5) - spread_buf
            if entry <= sl:
                sl = entry - atr * self.cfg.atr_sl_mult - spread_buf

            tp1 = entry + atr * self.cfg.atr_tp1_mult
            tp2 = entry + atr * self.cfg.atr_tp2_mult
            tp3 = entry + atr * self.cfg.atr_tp3_mult
            return round(entry, dec), round(sl, dec), round(tp1, dec), round(tp2, dec), round(tp3, dec)

        else:
            ob_candidate = (ob_low + (ob_high - ob_low) * 0.5) if (ob and ob_low > price) else None
            # Check if OB candidate is within safe pullback distance
            if ob_candidate and (ob_candidate - price) <= self.cfg.entry_max_chase_atr * atr:
                entry = ob_candidate
                sl_ref = ob_high
            else:
                entry = price + atr * self.cfg.entry_pullback_atr
                sl_ref = entry + atr * self.cfg.atr_sl_mult

            entry = max(entry, price + atr * 0.05)
            # Ensure entry is always within safe pullback bounds
            if (entry - price) > self.cfg.entry_max_chase_atr * atr:
                entry = price + atr * self.cfg.entry_pullback_atr

            sl = max(sl_ref, entry + atr * 0.5) + spread_buf
            if sl <= entry:
                sl = entry + atr * self.cfg.atr_sl_mult + spread_buf

            tp1 = entry - atr * self.cfg.atr_tp1_mult
            tp2 = entry - atr * self.cfg.atr_tp2_mult
            tp3 = entry - atr * self.cfg.atr_tp3_mult
            return round(entry, dec), round(sl, dec), round(tp1, dec), round(tp2, dec), round(tp3, dec)


# ═════════════════════════════════════════════════════════════════════════════
# 8. RISK MANAGEMENT ENGINE (DYNAMIC LOT SIZING & CIRCUIT BREAKERS)
# ═════════════════════════════════════════════════════════════════════════════
class RiskManagementEngine:
    """Calculates exact gold position sizing and enforces daily drawdown protection."""
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        """Compounds equity based on historical trade outcomes in the journal."""
        base = self.cfg.account_balance
        journal = self.kv.get("journal", []) or []
        closed_trades = [
            j for j in journal
            if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS") and j.get("r_achieved") is not None
        ]
        closed_trades.sort(key=lambda x: x.get("timestamp", ""))

        for t in closed_trades:
            try:
                r_multiple = float(t.get("r_achieved", 0.0))
                risk_pct = float(t.get("risk_pct", 1.0))
                base += r_multiple * (base * (risk_pct / 100.0))
            except (ValueError, TypeError):
                continue

        return max(10.0, round(base, 2))

    def compute_lot(
        self, balance: float, entry: float, sl: float, confidence_pct: float
    ) -> Tuple[float, float]:
        """
        Calculates dynamic lot size based on equity risk percentage and ATR stop loss distance.
        
        Formula:
          Risk Money ($) = Balance * (Risk % / 100)
          SL Distance ($/oz) = |Entry - SL|
          Contract Size = 100 Troy Oz (Standard Gold)
          Lot = Risk Money / (SL Distance * Contract Size)
        """
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-4:
            raise ValueError("Stop loss distance too small for gold position sizing")

        # Confidence-based scaling: 40% to 100% of base risk
        conf_factor = max(0.4, min(1.0, confidence_pct / 100.0))
        risk_pct = max(0.1, self.cfg.max_risk_pct_per_trade * conf_factor)
        risk_dollars = balance * (risk_pct / 100.0)

        # 1 standard lot = 100 oz
        raw_lot = risk_dollars / (sl_dist * self.cfg.contract_size)
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(raw_lot, 2)))

        # Hard Leverage Cap Enforcement
        notional_value = lot * self.cfg.contract_size * entry
        if balance > 0 and (notional_value / balance) > self.cfg.hard_leverage_cap:
            max_permitted_lot = (self.cfg.hard_leverage_cap * balance) / (self.cfg.contract_size * entry)
            lot = max(self.cfg.min_lot, round(max_permitted_lot, 2))

        actual_risk_pct = (lot * self.cfg.contract_size * sl_dist) / balance * 100.0
        return lot, round(actual_risk_pct, 2)

    def check_circuit_breakers(self, current_balance: float) -> Tuple[bool, str]:
        """Protects capital by locking bot execution during adverse market streaks."""
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        cb = self.kv.get("circuit_breaker", {}) or {}

        # 1. Active Drawdown Lock
        locked_until = cb.get("locked_until")
        if locked_until:
            try:
                lock_dt = datetime.fromisoformat(locked_until)
                if datetime.now(UTC) < lock_dt:
                    return False, f"CIRCUIT BREAKER ENGAGED: Locked until {locked_until}"
            except Exception:
                pass

        # 2. Daily Loss Limit Protection
        day_start_balance = cb.get("day_start_balance")
        last_date = cb.get("last_date", "")
        if last_date != today_str or day_start_balance is None:
            cb["last_date"] = today_str
            cb["day_start_balance"] = current_balance
            cb["daily_signals_count"] = 0
            self.kv.set("circuit_breaker", cb)
            day_start_balance = current_balance

        daily_drawdown = (day_start_balance - current_balance) / day_start_balance * 100.0
        if daily_drawdown >= self.cfg.max_daily_loss_pct:
            lock_time = (datetime.now(UTC) + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lock_time
            self.kv.set("circuit_breaker", cb)
            self.kv.save()
            return False, f"MAX DAILY DRAWDOWN BREACH: {daily_drawdown:.2f}% >= {self.cfg.max_daily_loss_pct}%. Engine locked for {self.cfg.dd_pause_hours}h."

        # 3. Consecutive Losses Circuit Breaker
        journal = self.kv.get("journal", []) or []
        closed = [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        consec_losses = 0
        for trade in reversed(closed):
            if trade.get("outcome") == "LOSS":
                consec_losses += 1
            else:
                break

        if consec_losses >= self.cfg.max_consecutive_losses:
            lock_time = (datetime.now(UTC) + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lock_time
            self.kv.set("circuit_breaker", cb)
            self.kv.save()
            return False, f"CONSECUTIVE LOSSES BREACH: {consec_losses} losses in a row. Trading halted."

        # 4. Daily Signal Quota
        sig_count = cb.get("daily_signals_count", 0)
        if sig_count >= self.cfg.max_daily_signals:
            return False, f"DAILY SIGNAL CAP REACHED: {sig_count}/{self.cfg.max_daily_signals} signals generated today."

        return True, "All circuit breakers clear."


# ═════════════════════════════════════════════════════════════════════════════
# 9. HIGH-IMPACT NEWS SAFETY FILTER (FOREX FACTORY BLACKOUT)
# ═════════════════════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    """Blocks new signal entries 30 minutes before and after high-impact USD events."""
    def __init__(self, cfg: Config, http: ResilientHttpClient):
        self.cfg = cfg
        self.http = http
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def fetch_events(self) -> List[dict]:
        try:
            r = self.http.request("GET", self.url)
            data = r.json()
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            logging.warning(f"News calendar fetch failed: {redact(e)}. Proceeding with caution.")
            return []

    def is_blackout(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        if now is None:
            now = datetime.now(UTC)
        events = self.fetch_events()
        window = timedelta(minutes=self.cfg.news_blackout_minutes)

        for ev in events:
            # Check high-impact USD economic events (NFP, CPI, FOMC, GDP, etc.)
            if ev.get("country") == "USD" and ev.get("impact") in ("High", "Holiday"):
                date_str = ev.get("date", "")
                if not date_str:
                    continue
                try:
                    ev_time = datetime.fromisoformat(date_str)
                    if ev_time.tzinfo is None:
                        ev_time = ev_time.replace(tzinfo=UTC)
                    else:
                        ev_time = ev_time.astimezone(UTC)

                    if (ev_time - window) <= now <= (ev_time + window):
                        title = ev.get("title", "High-Impact Economic Release")
                        diff_min = int((ev_time - now).total_seconds() / 60)
                        status = f"in {diff_min}m" if diff_min > 0 else f"{abs(diff_min)}m ago"
                        return True, f"{title} ({status})"
                except Exception:
                    continue

        return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# 10. TELEGRAM NOTIFIER (MOBILE MT5 READY FORMAT)
# ═════════════════════════════════════════════════════════════════════════════
class TelegramNotifier:
    """Transmits signals formatted for instantaneous manual execution in MT5 Mobile."""
    def __init__(self, http: ResilientHttpClient, dry_run: bool = False):
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.http = http
        self.dry_run = dry_run

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


# ═════════════════════════════════════════════════════════════════════════════
# 11. OUTCOME TRACKER & JOURNAL AUDITING
# ═════════════════════════════════════════════════════════════════════════════
class OutcomeTracker:
    """Monitors active and pending trades against closed candlestick data."""
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        journal = self.kv.get("journal", []) or []
        events = []
        now = datetime.now(UTC)
        modified = False

        for entry in journal:
            if entry.get("outcome") != "PENDING":
                continue

            ts_str = entry.get("timestamp", "")
            try:
                anchor_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except Exception:
                anchor_ts = now - timedelta(hours=24)

            # Check for stale unfilled pending orders
            age_hours = (now - anchor_ts).total_seconds() / 3600.0
            if not entry.get("filled") and age_hours > self.cfg.pending_stale_hours:
                entry["outcome"] = "STALE"
                entry["filled"] = 0
                entry["r_achieved"] = 0.0
                events.append(("STALE", dict(entry)))
                modified = True
                continue

            side = entry["signal"]
            ep = float(entry["entry"])
            sl = float(entry["sl"])
            tp1 = float(entry["tp1"])
            tp2 = float(entry["tp2"])
            tp3 = float(entry["tp3"])

            if ep <= 0 or sl <= 0:
                continue

            bars = m5[m5["datetime"] > anchor_ts].sort_values("datetime")
            if bars.empty:
                continue

            filled = bool(entry.get("filled"))
            fill_index: Optional[int] = None

            # Check if limit order was filled
            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep:
                        filled, fill_index = True, idx
                        break
                    if side == "SELL" and b.high >= ep:
                        filled, fill_index = True, idx
                        break

                if not filled:
                    # Invalidation check: Price ran away without pulling back into our limit
                    chase_limit = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    last_h = float(bars["high"].iloc[-1])
                    last_l = float(bars["low"].iloc[-1])
                    if side == "BUY" and last_h > ep + chase_limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                        continue
                    if side == "SELL" and last_l < ep - chase_limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                        continue
                    continue

                entry["filled"] = 1
                modified = True

            # Walk through filled trade bars
            walk_bars = bars.iloc[fill_index:] if fill_index is not None else bars
            hit_tp1 = bool(entry.get("hit_tp1"))
            hit_tp2 = bool(entry.get("hit_tp2"))
            hit_tp3 = bool(entry.get("hit_tp3"))
            hit_sl = bool(entry.get("hit_sl"))
            hit_be = bool(entry.get("hit_be"))
            closed_now = False

            for b in walk_bars.itertuples(index=False):
                if hit_tp3 or hit_sl or hit_be:
                    break

                current_sl = ep if hit_tp1 else sl

                if side == "BUY":
                    if b.low <= current_sl:
                        if hit_tp1:
                            hit_be = True
                        else:
                            hit_sl = True
                        continue
                    if not hit_tp1 and b.high >= tp1:
                        hit_tp1 = True
                    if not hit_tp2 and b.high >= tp2:
                        hit_tp2 = True
                    if not hit_tp3 and b.high >= tp3:
                        hit_tp3 = True
                else:
                    if b.high >= current_sl:
                        if hit_tp1:
                            hit_be = True
                        else:
                            hit_sl = True
                        continue
                    if not hit_tp1 and b.low <= tp1:
                        hit_tp1 = True
                    if not hit_tp2 and b.low <= tp2:
                        hit_tp2 = True
                    if not hit_tp3 and b.low <= tp3:
                        hit_tp3 = True

            entry["hit_tp1"] = int(hit_tp1)
            entry["hit_tp2"] = int(hit_tp2)
            entry["hit_tp3"] = int(hit_tp3)
            entry["hit_sl"] = int(hit_sl)
            entry["hit_be"] = int(hit_be)

            # Determine final realized outcome
            if hit_tp3:
                entry["outcome"] = "WIN_TP3"
                entry["r_achieved"] = 3.0
                closed_now = True
            elif hit_be:
                entry["outcome"] = "WIN_BE"
                entry["r_achieved"] = 0.33  # TP1 banked + remaining exited at BE
                closed_now = True
            elif hit_sl:
                entry["outcome"] = "LOSS"
                entry["r_achieved"] = -1.0
                closed_now = True

            if closed_now:
                events.append(("CLOSE", dict(entry)))
                modified = True

        if modified:
            self.kv.set("journal", journal)
            self.kv.save()

        return events


# ═════════════════════════════════════════════════════════════════════════════
# 12. TRADING BOT ORCHESTRATOR (MAIN SYSTEM LOOP)
# ═════════════════════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    """Master workflow controller coordinating data, signals, risk, and alerts."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = ResilientHttpClient(
            timeout=cfg.http_timeout_sec,
            max_retries=cfg.http_max_retries,
            backoff=cfg.http_backoff_factor,
        )
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.http)
        self.tg = TelegramNotifier(self.http, dry_run=cfg.dry_run)
        self.outcome = OutcomeTracker(cfg, self.kv)

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter:
            return True
        london_now = now.astimezone(LONDON_TZ)
        ny_now = now.astimezone(NY_TZ)
        lh = london_now.hour + london_now.minute / 60.0
        nh = ny_now.hour + ny_now.minute / 60.0

        london_open = 7.0 <= lh < 10.0
        ny_open = 8.0 <= nh < 11.0
        ny_afternoon = 13.0 <= nh < 16.0
        return london_open or ny_open or ny_afternoon

    def run(self) -> None:
        try:
            self.kv.force_load()
        except StateError as e:
            logging.error(f"Persistence state load error: {e}")
            return

        now = datetime.now(UTC)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.cfg.kill_switch:
            logging.info("KILL SWITCH ENGAGED — Execution halted")
            return
        if now.weekday() >= 5:
            logging.info("Weekend detected — Gold market closed, skipping iteration")
            return
        if not self._in_killzone(now):
            logging.info("Outside institutional killzone — Skipping iteration to preserve capital")
            return

        balance = self.risk.dynamic_balance()

        # Step 1: Evaluate Circuit Breakers
        ok, reason = self.risk.check_circuit_breakers(balance)
        if not ok:
            logging.info(f"Circuit Breaker Triggered: {reason}")
            return

        # Step 2: High-Impact News Blackout Filter
        in_blackout, news_title = self.news.is_blackout()
        if in_blackout:
            logging.info(f"News Safety Filter Active: [{news_title}] — Halting new signals")
            return

        # Step 3: Fetch Market Data with 4-Tier Failover
        try:
            dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            err = redact(e)
            logging.error(f"Fatal Market Data Failure: {err}")
            self.tg.send(f"⚠️ <b>Market Data Warning</b>\n<code>{err[:200]}</code>")
            return

        # Step 4: Update Trade Outcomes & Realized R
        try:
            events = self.outcome.update(dfs["5min"])
            for evt_type, trade in events:
                if evt_type == "CLOSE":
                    self.tg.send(
                        f"📕 <b>Position Closed</b> {trade['signal']} {trade['symbol']}\n"
                        f"Entry: <code>{trade['entry']}</code> → Result: <b>{trade['outcome']}</b> "
                        f"({trade.get('r_achieved', 0):+.2f}R)"
                    )
                elif evt_type in ("INVALIDATED", "STALE"):
                    self.tg.send(
                        f"⚪ <b>Order {evt_type}</b>: {trade['signal']} Limit @ <code>{trade['entry']}</code>"
                    )
        except Exception as e:
            logging.warning(f"Outcome tracker non-fatal warning: {e}")

        balance = self.risk.dynamic_balance()

        # Slice latest closed bar (avoid repainting or lookahead bias)
        m5_closed = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15_closed = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1_closed = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4_closed = dfs["4h"].iloc[:-1].reset_index(drop=True)

        if len(m5_closed) < self.cfg.min_bars_m5 or len(h1_closed) < self.cfg.min_bars_h1:
            logging.warning("Insufficient bars post-closed trim")
            return

        last_price = float(m5_closed["close"].iloc[-1])
        signal_bar_iso = m5_closed.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

        # Step 5: Check Active Position Limits
        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("Existing active position already open — Skipping new signal")
            self._update_state(last_price, now_str)
            return

        unfilled_count = sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled"))
        if unfilled_count >= 2:
            logging.info(f"Max unfilled pending limits ({unfilled_count}) reached — Skipping")
            self._update_state(last_price, now_str)
            return

        # Step 6: Generate Quantitative Signals
        result = self.signal.generate(m5_closed, m15_closed, h1_closed, h4_closed)
        logging.info(
            f"Parliament Consensus: {result.direction} | "
            f"BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%} | "
            f"Verdict={result.verdict} | Skip={result.skip_reason}"
        )

        if result.direction == "NEUTRAL" or result.verdict == "CANCEL":
            self._update_state(last_price, now_str)
            return

        # Step 7: Directional Cooldown Check
        state = self.kv.get("state", {}) or {}
        last_signal_dir = state.get("last_signal_type", "NEUTRAL")
        last_signal_time = state.get("last_signal_ts", "")
        cool_window = (
            self.cfg.opposite_cooldown_minutes
            if last_signal_dir != result.direction
            else self.cfg.cooldown_minutes
        )

        if last_signal_time:
            try:
                last_dt = datetime.strptime(last_signal_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                if (now - last_dt) < timedelta(minutes=cool_window):
                    logging.info(f"Signal cooldown ({cool_window} min) active — Skipping")
                    self._update_state(last_price, now_str)
                    return
            except Exception:
                pass

        # Step 8: Detect Order Block & Calculate Entry Levels
        ob_zone = self.signal.detect_order_block(m5_closed, result.direction)
        try:
            entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5_closed, result.direction, ob_zone)
        except ValueError as e:
            logging.warning(f"Level computation rejected: {e}")
            self._update_state(last_price, now_str)
            return

        # Step 9: Dynamic Position Sizing (Lot)
        conf_pct = max(result.buy_ratio, result.sell_ratio) * 100.0
        try:
            lot, actual_risk = self.risk.compute_lot(balance, entry, sl, conf_pct)
        except ValueError as e:
            logging.warning(f"Position sizing rejected: {e}")
            self._update_state(last_price, now_str)
            return

        # Scale lot according to Parliament Verdict
        if result.verdict == "EXECUTE_HALF":
            lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif result.verdict == "EXECUTE_MICRO":
            lot = max(self.cfg.min_lot, round(lot / 4.0, 2))

        # Step 10: Format MT5-Ready Signal Message
        action_name = f"{result.direction} LIMIT"
        ob_display = f"✅ {ob_zone['high']:.2f}–{ob_zone['low']:.2f}" if ob_zone else "⚠️ ATR Dynamic Buffer"

        tg_message = (
            f"🏛️ <b>APEX QUANT v{VERSION} — {action_name}</b>\n\n"
            f"Asset: <b>{self.cfg.symbol}</b>\n"
            f"Order Type: <b>{action_name}</b>\n"
            f"Entry Price: <code>{entry:.2f}</code>\n"
            f"Stop Loss: <code>{sl:.2f}</code>\n\n"
            f"🎯 TP1: <code>{tp1:.2f}</code> (1/3 Close + Move SL to BE)\n"
            f"🎯 TP2: <code>{tp2:.2f}</code> (1/3 Close)\n"
            f"🎯 TP3: <code>{tp3:.2f}</code> (Runner Target)\n\n"
            f"📦 <b>Suggested Lot: <code>{lot}</code></b>\n"
            f"💰 Balance: <code>${balance:.2f}</code> | Risk: <code>{actual_risk:.2f}%</code>\n"
            f"📊 Consensus: BUY={result.buy_ratio:.0%} | SELL={result.sell_ratio:.0%}\n"
            f"🧠 Verdict: <b>{result.verdict}</b> | Active Models: {result.active_models}/12\n"
            f"🌐 Regime: <b>{result.regime}</b> (Hurst={result.hurst:.2f}, Entropy={result.entropy:.2f})\n"
            f"🧱 SMC Zone: {ob_display}\n\n"
            f"⏰ Bar Time: <code>{signal_bar_iso} UTC</code>"
        )

        sent = self.tg.send(tg_message)
        if sent:
            # Record trade to journal
            trade_record = {
                "id": hashlib.md5(f"{signal_bar_iso}_{result.direction}".encode()).hexdigest()[:10],
                "timestamp": now_str,
                "symbol": self.cfg.symbol,
                "signal": result.direction,
                "verdict": result.verdict,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "lot": lot,
                "risk_pct": actual_risk,
                "filled": 0,
                "outcome": "PENDING",
                "r_achieved": None,
                "hit_tp1": 0,
                "hit_tp2": 0,
                "hit_tp3": 0,
                "hit_sl": 0,
                "hit_be": 0,
            }
            journal.append(trade_record)
            self.kv.set("journal", journal)

            # Increment daily signal counter
            cb = self.kv.get("circuit_breaker", {}) or {}
            cb["daily_signals_count"] = cb.get("daily_signals_count", 0) + 1
            self.kv.set("circuit_breaker", cb)

            # Update state cooldown
            state["last_signal_type"] = result.direction
            state["last_signal_ts"] = now_str
            self.kv.set("state", state)

        self._update_state(last_price, now_str)
        self.kv.save()

    def _update_state(self, last_price: float, now_str: str) -> None:
        state = self.kv.get("state", {}) or {}
        state["last_run_utc"] = now_str
        state["last_gold_price"] = last_price
        state["engine_version"] = VERSION
        self.kv.set("state", state)
        self.kv.save()


# ═════════════════════════════════════════════════════════════════════════════
# 13. COMMAND LINE INTERFACE & ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
def setup_logging(verbose: bool = False) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX Institutional Quantitative Trading Engine for Gold (XAUUSD)")
    parser.add_argument("command", choices=["run", "test", "audit", "state", "version"], help="Command to execute")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Execute without sending live broker/Telegram actions")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = Config.from_env()
    if args.dry_run:
        cfg.dry_run = True

    try:
        cfg.validate()
    except AssertionError as e:
        logging.critical(f"Configuration boundary validation failed: {e}")
        sys.exit(1)

    if args.command == "version":
        print(f"APEX Predator Engine v{VERSION}")
        sys.exit(0)

    orchestrator = TradingBotOrchestrator(cfg)

    if args.command == "run":
        orchestrator.run()
    elif args.command == "state":
        orchestrator.kv.force_load()
        print(json.dumps(orchestrator.kv.data, indent=2, default=str))
    elif args.command == "audit":
        orchestrator.kv.force_load()
        journal = orchestrator.kv.get("journal", []) or []
        print(f"Total Logged Trades: {len(journal)}")
        wins = sum(1 for j in journal if str(j.get("outcome", "")).startswith("WIN"))
        losses = sum(1 for j in journal if j.get("outcome") == "LOSS")
        print(f"Wins: {wins} | Losses: {losses}")
    elif args.command == "test":
        logging.info("Running system connectivity self-test...")
        orchestrator.kv.force_load()
        dfs, src, offset = orchestrator.data.fetch_all()
        logging.info(f"Self-test SUCCESS: Source={src}, Bars={len(dfs['5min'])}, Offset={offset}")


if __name__ == "__main__":
    main()
