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

try:
    from sklearn.ensemble import RandomForestClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "18.0-PRODUCTION"
NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
UTC = timezone.utc


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & RUNTIME VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v18.json"
    local_state_file: str = "apex_state_v18.json"

    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # 1 Standard Lot = 100 Troy Ounces Gold
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0

    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01

    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24
    max_daily_signals: int = 5
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120

    atr_period: int = 14
    atr_sl_mult: float = 1.8
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.4
    atr_tp3_mult: float = 4.0
    spread_buffer: float = 0.35

    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    m5_fetch_size: int = 800
    max_stale_mult: float = 3.0
    price_min: float = 500.0
    price_max: float = 10000.0
    price_max_jump_pct: float = 0.10

    parliament_min_active: int = 8
    parliament_min_ratio: float = 0.55
    parliament_half_ratio: float = 0.65
    parliament_full_ratio: float = 0.75
    adx_min: float = 22.0
    hurst_trend_min: float = 0.55
    hurst_meanrev_max: float = 0.45
    entropy_chaos_max: float = 0.85

    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_cache_sec_on_fail: int = 300
    news_max_consecutive_fail: int = 3

    dry_run: bool = True
    kill_switch: bool = False
    skip_killzone_filter: bool = False
    pending_max_age_hours: int = 24
    entry_pullback_atr: float = 0.30
    entry_max_chase_atr: float = 2.0
    http_timeout: tuple = (3, 10)
    max_retry: int = 3
    max_journal: int = 500
    h1_cache_minutes: int = 55
    h4_cache_minutes: int = 220
    td_min_interval_sec: float = 1.5
    enable_ml_models: bool = True
    ml_purge_bars: int = 10
    ml_min_val_acc: float = 0.56

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for name in cls.__dataclass_fields__:
            env_key = f"APEX_{name.upper()}"
            raw = os.getenv(env_key)
            if raw is None or not str(raw).strip():
                continue
            cur = getattr(cfg, name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(cur, float):
                    v = float(raw)
                    if not np.isfinite(v):
                        raise ValueError("Non-finite")
                    setattr(cfg, name, v)
                elif isinstance(cur, str):
                    setattr(cfg, name, str(raw).strip())
            except (ValueError, TypeError):
                logging.warning(f"Config: Abaikan env {env_key}={raw}")

        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        offset_env = os.getenv("APEX_MT5_OFFSET", "").strip()
        if offset_env:
            try:
                cfg.mt5_offset = float(offset_env)
            except ValueError:
                pass
        if os.getenv("APEX_KILL_SWITCH", "").strip().lower() in ("1", "true", "yes"):
            cfg.kill_switch = True

        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0 < self.parliament_min_ratio < self.parliament_half_ratio < self.parliament_full_ratio <= 1.0):
            raise ValueError("Rasio parlemen tidak valid")
        if not (self.max_lot >= self.min_lot > 0):
            raise ValueError("Batas lot tidak valid")
        if self.account_balance <= 0:
            raise ValueError("Saldo akun harus > 0")
        if not (0 < self.max_risk_pct_per_trade <= 10.0):
            raise ValueError("max_risk_pct_per_trade harus 0.1% s.d 10.0%")
        if not (0 < self.max_daily_loss_pct <= 25.0):
            raise ValueError("max_daily_loss_pct harus 0.1% s.d 25.0%")
        if self.contract_size <= 0:
            raise ValueError("contract_size harus > 0")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOGGING & RESILIENT HTTP CLIENT
# ═════════════════════════════════════════════════════════════════════════════
def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.addHandler(h)


def redact(text: Any) -> str:
    s = str(text).replace("\n", " ").replace("\r", " ")
    return re.sub(r"(token|apikey|api_key|bearer|secret)\s*[=:]\s*[^\s&'\"]+", r"\1=***", s, flags=re.I)


class DataFetchError(Exception):
    pass


class StateError(Exception):
    pass


class HTTPClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.cfg.http_timeout)
        last_error = "Unknown"
        for attempt in range(self.cfg.max_retry):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                    self._backoff(attempt)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_error = redact(e)
                self._backoff(attempt)
        raise DataFetchError(f"HTTP gagal: {last_error}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(0.5 * (2 ** attempt) + random.uniform(0.1, 0.4))


# ═════════════════════════════════════════════════════════════════════════════
# 3. PERSISTENT KEY-VALUE STORE
# ═════════════════════════════════════════════════════════════════════════════
class KVStore:
    def __init__(self, cfg: Config, http: HTTPClient):
        self.cfg = cfg
        self.http = http
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: Optional[dict] = None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg.gist_token}",
            "Accept": "application/vnd.github+json",
        }

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_gist:
            try:
                r = self.http.request("GET", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers())
                files = r.json().get("files", {})
                content = files.get(self.cfg.gist_state_file, {}).get("content", "").strip()
                self._cache = json.loads(content) if content else {}
            except Exception as e:
                raise StateError(f"Gist load: {redact(e)}")
        else:
            p = Path(self.cfg.local_state_file)
            try:
                self._cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except Exception as e:
                raise StateError(f"Local load: {e}")
        return self._cache

    def _save(self, data: dict) -> bool:
        if self.use_gist:
            try:
                payload = {"files": {self.cfg.gist_state_file: {"content": json.dumps(data, default=str, indent=2, sort_keys=True)}}}
                self.http.request("PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers(), json=payload)
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"Gist save: {redact(e)}")
                return False
        else:
            try:
                Path(self.cfg.local_state_file).write_text(json.dumps(data, default=str, indent=2, sort_keys=True), encoding="utf-8")
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"Local save: {e}")
                return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._load().get(key, default)
        except StateError:
            return default

    def set(self, key: str, value: Any) -> bool:
        try:
            data = dict(self._load())
        except StateError:
            data = {}
        data[key] = value
        return self._save(data)

    def force_load(self) -> None:
        self._cache = None
        self._load()


# ═════════════════════════════════════════════════════════════════════════════
# 4. QUANTITATIVE INDICATOR LIBRARY (NO LOOKAHEAD BIAS)
# ═════════════════════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        v = Indicators.atr_series(df, period).iloc[-1]
        return 0.0 if pd.isna(v) or v < 1e-6 else float(v)

    @staticmethod
    def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
        return 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < period * 2:
                return 0.0
            up, dn = df["high"].diff(), -df["low"].diff()
            pdm = up.where((up > dn) & (up > 0), 0.0)
            mdm = dn.where((dn > up) & (dn > 0), 0.0)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
            atr_s = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)
            pdi = 100.0 * pdm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s
            mdi = 100.0 * mdm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s
            dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100.0
            v = dx.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
            return 0.0 if pd.isna(v) or v < 0 else float(v)
        except Exception:
            return 0.0

    @staticmethod
    def hurst(close: pd.Series, max_lag: int = 20) -> float:
        try:
            ts = np.log(close.dropna().values)
            if len(ts) < max_lag * 2:
                return 0.5
            lags, tau = [], []
            for lag in range(2, max_lag):
                diff = ts[lag:] - ts[:-lag]
                std = float(np.std(diff))
                if std > 1e-12:
                    lags.append(lag)
                    tau.append(std)
            if len(tau) < 4:
                return 0.5
            return float(np.clip(np.polyfit(np.log(lags), np.log(tau), 1)[0], 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def shannon_entropy(close: pd.Series, bins: int = 10) -> float:
        try:
            returns = close.pct_change().dropna()
            if len(returns) < bins * 5:
                return 0.5
            hist, _ = np.histogram(returns, bins=bins)
            p = hist / hist.sum()
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)) / np.log(bins))
        except Exception:
            return 0.5

    @staticmethod
    def kalman_estimate(series: np.ndarray) -> np.ndarray:
        n = len(series)
        if n < 2:
            return series
        x, P, Q, R = series[0], 1.0, 1e-3, 1e-2
        est = np.empty(n)
        est[0] = x
        for i in range(1, n):
            P += Q
            K = P / (P + R)
            x += K * (series[i] - x)
            P *= (1.0 - K)
            est[i] = x
        return est


# ═════════════════════════════════════════════════════════════════════════════
# 5. MARKET DATA HANDLER (4-TIER MULTI-PROVIDER FAILOVER)
# ═════════════════════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg
        self.http = http
        self.kv = kv
        self._last_td_call = 0.0

    def _validate(self, df: pd.DataFrame, source_label: str, min_bars: int) -> None:
        if df is None or len(df) < min_bars:
            raise DataFetchError(f"{source_label}: Batang kurang ({0 if df is None else len(df)} < {min_bars})")
        if df["datetime"].isna().any() or df["datetime"].duplicated().any():
            raise DataFetchError(f"{source_label}: Masalah timestamp")
        last_price = float(df["close"].iloc[-1])
        if not (self.cfg.price_min <= last_price <= self.cfg.price_max):
            raise DataFetchError(f"{source_label}: Harga {last_price} di luar batas aman")
        if len(df) > 2:
            prev = float(df["close"].iloc[-2])
            jump = abs(last_price - prev) / max(prev, 1e-9)
            if jump > self.cfg.price_max_jump_pct:
                raise DataFetchError(f"{source_label}: Lonjakan anomali {jump:.1%}")

    def _check_staleness(self, df: pd.DataFrame, tf_minutes: int, source_label: str) -> None:
        if df.empty:
            return
        last_dt = df["datetime"].iloc[-1]
        if last_dt.tzinfo is None:
            last_dt = last_dt.tz_localize("UTC")
        age = (datetime.now(UTC) - last_dt.to_pydatetime()).total_seconds() / 60.0
        if age > self.cfg.max_stale_mult * tf_minutes:
            raise DataFetchError(f"{source_label}: Data basi ({age:.1f}m)")

    def _compute_offset(self) -> float:
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda_key and td_key):
            return self.cfg.mt5_offset
        try:
            r1 = self.http.request("GET", "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing", headers={"Authorization": f"Bearer {oanda_key}"})
            oanda_bid = float(r1.json()["prices"][0]["bids"][0]["price"])
            r2 = self.http.request("GET", "https://api.twelvedata.com/price", params={"symbol": "XAU/USD", "apikey": td_key})
            td_price = float(r2.json()["price"])
            if td_price <= 0:
                return self.cfg.mt5_offset
            diff = oanda_bid - td_price
            return round(diff, 3) if abs(diff) <= 2.0 else self.cfg.mt5_offset
        except Exception:
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
        return (
            df.set_index("datetime")
            .resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

    def _fetch_twelvedata(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("Tanpa TD key")

        def fetch(iv: str, size: int) -> pd.DataFrame:
            el = time.time() - self._last_td_call
            if el < self.cfg.td_min_interval_sec:
                time.sleep(self.cfg.td_min_interval_sec - el)
            self._last_td_call = time.time()
            r = self.http.request("GET", "https://api.twelvedata.com/time_series", params={"symbol": "XAU/USD", "interval": iv, "outputsize": size, "apikey": key, "timezone": "UTC"})
            data = r.json()
            if "values" not in data or not data["values"]:
                raise DataFetchError(f"TD {iv}: {data.get('message', 'Kosong')}")
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = 100.0
            return self._apply_offset(df.dropna().sort_values("datetime").reset_index(drop=True), offset)

        m5 = fetch("5min", min(self.cfg.m5_fetch_size, 800))
        h1 = fetch("1h", 800)
        h4 = fetch("4h", 800)
        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def _fetch_xaus(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        def fetch(iv: str, rng: str) -> pd.DataFrame:
            r = self.http.request("GET", "https://xaus.com/api/v1/chart", params={"symbol": "xau", "interval": iv, "range": rng}, headers={"Accept": "application/json"})
            pts = r.json().get("points")
            if not pts:
                raise DataFetchError(f"XAUS {iv}: Kosong")
            rows = [{"datetime": pd.to_datetime(int(p["t"]), unit="s", utc=True), "open": float(p["o"]), "high": float(p["h"]), "low": float(p["l"]), "close": float(p["c"]), "volume": float(p.get("v", 100.0) or 100.0)} for p in pts if all(k in p for k in ("t", "o", "h", "l", "c"))]
            df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5 = fetch("5m", "5d")
        h1 = fetch("1h", "1mo")
        h4 = self._apply_offset(self._resample(h1, "4h"), offset)
        self._validate(m5, "XAUS-M5", self.cfg.min_bars_m5)
        self._validate(h1, "XAUS-H1", self.cfg.min_bars_h1)
        self._validate(h4, "XAUS-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "XAUS-M5")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "XAUS-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "XAUS"

    def _fetch_biquote(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        h = {"User-Agent": "Mozilla/5.0 APEX-Bot/18.0", "Accept": "application/json"}
        def fetch(iv: str) -> pd.DataFrame:
            r = self.http.request("GET", "https://biquote.io/api/XAUUSD/ohlc", params={"interval": iv, "limit": 800}, headers=h)
            bars = r.json().get("bars")
            if not bars:
                raise DataFetchError(f"BQ {iv}: Kosong")
            rows = [{"datetime": pd.to_datetime(b["openTime"], utc=True), "open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"]), "volume": float(b.get("tickVolume", 100.0) or 100.0)} for b in bars]
            df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5, h1, h4 = fetch("5m"), fetch("1h"), fetch("4h")
        self._validate(m5, "BQ-M5", self.cfg.min_bars_m5)
        self._validate(h1, "BQ-H1", self.cfg.min_bars_h1)
        self._validate(h4, "BQ-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "BQ-M5")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "BQ-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "Biquote"

    def _fetch_yahoo(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        h = {"User-Agent": "Mozilla/5.0 APEX-Bot/18.0", "Accept": "application/json"}
        def fetch(iv: str, rng: str) -> pd.DataFrame:
            r = self.http.request("GET", "https://query1.finance.yahoo.com/v8/finance/chart/GC=F", params={"interval": iv, "range": rng}, headers=h)
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res:
                raise DataFetchError(f"Yahoo {iv}: Kosong")
            ts = res.get("timestamp", [])
            q = res.get("indicators", {}).get("quote", [{}])[0]
            n = min(len(ts), len(q.get("open", [])), len(q.get("high", [])), len(q.get("low", [])), len(q.get("close", [])))
            rows = [{"datetime": pd.to_datetime(int(ts[i]), unit="s", utc=True), "open": float(q["open"][i]), "high": float(q["high"][i]), "low": float(q["low"][i]), "close": float(q["close"][i]), "volume": 100.0} for i in range(n) if None not in (q["open"][i], q["high"][i], q["low"][i], q["close"][i])]
            df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5, m15, h1 = fetch("5m", "5d"), fetch("15m", "5d"), fetch("1h", "1mo")
        h4 = self._apply_offset(self._resample(h1, "4h"), offset)
        self._validate(m5, "YH-M5", self.cfg.min_bars_m5)
        self._validate(m15, "YH-M15", self.cfg.min_bars_m15)
        self._validate(h1, "YH-H1", self.cfg.min_bars_h1)
        self._validate(h4, "YH-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "YH-M5")
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "Yahoo Finance"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        offset = self._compute_offset()
        errors = []
        for name, fn in [("TwelveData", self._fetch_twelvedata), ("XAUS", self._fetch_xaus), ("Biquote", self._fetch_biquote), ("Yahoo", self._fetch_yahoo)]:
            try:
                dfs, src = fn(offset)
                logging.info(f"Data source: {src} | Offset: {offset:+.3f}")
                return dfs, src, offset
            except Exception as e:
                errors.append(f"{name}: {redact(e)}")
        raise DataFetchError("Semua data feed gagal: " + " | ".join(errors))


# ═════════════════════════════════════════════════════════════════════════════
# 6. SIGNAL GENERATOR (12-MODEL QUANT PARLIAMENT & SMC)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    buy_score: float = 0.0
    sell_score: float = 0.0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    active_models: int = 0
    verdict: str = "CANCEL"
    skip_reason: str = ""
    votes_detail: list = field(default_factory=list)
    regime: str = "UNKNOWN"
    adx: float = 0.0
    hurst: float = 0.5
    entropy: float = 0.5


class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._registry = self._build_model_registry()

    @staticmethod
    def _ema_cross(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 30: return 0, 0.0
        diff = Indicators.ema(df["close"], 9) - Indicators.ema(df["close"], 21)
        atr = Indicators.atr_last(df) + 1e-9
        if diff.iloc[-1] > 0 and diff.diff().iloc[-1] > 0: return 1, min(1.0, float(abs(diff.iloc[-1]) / atr))
        if diff.iloc[-1] < 0 and diff.diff().iloc[-1] < 0: return -1, min(1.0, float(abs(diff.iloc[-1]) / atr))
        return 0, 0.0

    @staticmethod
    def _ema_trend(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        c = df["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return 1, 0.9
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[int, float]:
        if len(df) < period + 5: return 0, 0.0
        atr = Indicators.atr_series(df, period)
        hl2 = (df["high"] + df["low"]) / 2.0
        upper = (hl2 + mult * atr).values
        lower = (hl2 - mult * atr).values
        close = df["close"].values
        trend = np.ones(len(df))
        for i in range(1, len(df)):
            if close[i] > upper[i - 1]: trend[i] = 1
            elif close[i] < lower[i - 1]: trend[i] = -1
            else: trend[i] = trend[i - 1]
        if trend[-1] == 1 and trend[-2] == 1: return 1, 0.85
        if trend[-1] == -1 and trend[-2] == -1: return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _donchian(df: pd.DataFrame, period: int = 20) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        hh, ll, c = df["high"].rolling(period).max().iloc[-2], df["low"].rolling(period).min().iloc[-2], df["close"].iloc[-1]
        if c > hh: return 1, 0.8
        if c < ll: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _kalman(df: pd.DataFrame) -> Tuple[int, float]:
        obs = df["close"].values
        if len(obs) < 15: return 0, 0.0
        est = Indicators.kalman_estimate(obs)
        slope, atr = est[-1] - est[-5], Indicators.atr_last(df) + 1e-9
        if slope > 0: return 1, min(1.0, float(abs(slope) / atr))
        if slope < 0: return -1, min(1.0, float(abs(slope) / atr))
        return 0, 0.0

    @staticmethod
    def _macd(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 35: return 0, 0.0
        macd = Indicators.ema(df["close"], 12) - Indicators.ema(df["close"], 26)
        sig = Indicators.ema(macd, 9)
        hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0: return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0: return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _rsi(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 16: return 0, 0.0
        rsi = Indicators.rsi_series(df["close"], 14)
        cur, prev = rsi.iloc[-1], rsi.iloc[-2]
        if cur < 30 and cur > prev: return 1, 0.85
        if cur > 70 and cur < prev: return -1, 0.85
        if 50 < cur < 65 and cur > prev: return 1, 0.6
        if 35 < cur < 50 and cur < prev: return -1, 0.6
        return 0, 0.0

    @staticmethod
    def _stochastic(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 20: return 0, 0.0
        low, high = df["low"].rolling(14).min(), df["high"].rolling(14).max()
        k = 100.0 * (df["close"] - low) / (high - low + 1e-9)
        d = k.rolling(3).mean()
        if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] < 25: return 1, 0.8
        if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] > 75: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _bollinger(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        ma, std, c = df["close"].rolling(period).mean(), df["close"].rolling(period).std(), df["close"].iloc[-1]
        if c < (ma - mult * std).iloc[-1]: return 1, 0.75
        if c > (ma + mult * std).iloc[-1]: return -1, 0.75
        return 0, 0.0

    @staticmethod
    def _ichimoku(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        hi, lo, c = df["high"], df["low"], df["close"]
        tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2.0
        kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2.0
        sa, sb = ((tenkan + kijun) / 2.0).shift(26), ((hi.rolling(52).max() + lo.rolling(52).min()) / 2.0).shift(26)
        if pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]): return 0, 0.0
        if c.iloc[-1] > max(sa.iloc[-1], sb.iloc[-1]) and tenkan.iloc[-1] > kijun.iloc[-1]: return 1, 0.85
        if c.iloc[-1] < min(sa.iloc[-1], sb.iloc[-1]) and tenkan.iloc[-1] < kijun.iloc[-1]: return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _smc_orderblock(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        tail = df.tail(60).reset_index(drop=True)
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        bodies = (tail["close"] - tail["open"]).abs()
        cands = bodies.iloc[3:-1]
        if cands.empty: return 0, 0.0
        imp = int(cands.idxmax())
        if imp < 1 or abs(tail["close"].iloc[imp] - tail["open"].iloc[imp]) < atr * 1.2: return 0, 0.0
        is_bull_imp = tail["close"].iloc[imp] > tail["open"].iloc[imp]
        ob = tail.iloc[imp - 1]
        if is_bull_imp == (ob["close"] > ob["open"]): return 0, 0.0
        last_c, ob_h, ob_l = tail["close"].iloc[-1], max(ob["open"], ob["close"]), min(ob["open"], ob["close"])
        if is_bull_imp and (ob_l - atr * 0.5 <= last_c <= ob_h + atr * 0.5): return 1, 0.85
        if (not is_bull_imp) and (ob_l - atr * 0.5 <= last_c <= ob_h + atr * 0.5): return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _wavelet(df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SKLEARN or len(df) < 100: return 0, 0.0
        try:
            b, a = butter(2, 0.15)
            smooth = pd.Series(filtfilt(b, a, df["close"].values))
            ef, es = smooth.ewm(span=10).mean(), smooth.ewm(span=30).mean()
            diff, slope, atr = ef.iloc[-1] - es.iloc[-1], ef.diff(3).iloc[-1], Indicators.atr_last(df)
            if atr <= 0: return 0, 0.0
            conf = min(1.0, float(abs(diff) / atr))
            if diff > 0 and slope > 0: return 1, conf
            if diff < 0 and slope < 0: return -1, conf
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _ml_ensemble(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models) or len(df) < 300: return 0, 0.0
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

            horizon = 20
            atr_v, closes, highs, lows = Indicators.atr_series(df).values, c.values, df["high"].values, df["low"].values
            n = len(closes)
            y_arr = np.zeros(n)
            for i in range(n - horizon):
                tp, sl = closes[i] + 1.0 * atr_v[i], closes[i] - 1.8 * atr_v[i]
                hit_tp = np.where(highs[i+1:i+horizon+1] >= tp)[0]
                hit_sl = np.where(lows[i+1:i+horizon+1] <= sl)[0]
                t_tp = hit_tp[0] if len(hit_tp) else 999
                t_sl = hit_sl[0] if len(hit_sl) else 999
                if t_tp < t_sl: y_arr[i] = 1
                elif t_sl < t_tp: y_arr[i] = 0

            y = pd.Series(y_arr[:n - horizon], index=df.index[:n - horizon])
            valid = features.index.intersection(y.index)
            X_clean, y_clean = features.loc[valid], y.loc[valid]
            split = int(len(X_clean) * 0.8)
            X_tr, y_tr = X_clean.iloc[:split], y_clean.iloc[:split]
            X_val, y_val = X_clean.iloc[split:], y_clean.iloc[split:]
            if len(X_tr) < 100 or len(X_val) < 20: return 0, 0.0
            clf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1)
            clf.fit(X_tr.values, y_tr.values)
            if clf.score(X_val.values, y_val.values) < self.cfg.ml_min_val_acc: return 0, 0.0
            prob = clf.predict_proba(features.iloc[-1:].values)[0]
            prob_up = float(prob[1]) if len(prob) > 1 else 0.5
            if prob_up > 0.62: return 1, min(1.0, (prob_up - 0.5) * 2.0)
            if prob_up < 0.38: return -1, min(1.0, (0.5 - prob_up) * 2.0)
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _build_model_registry(self) -> List[Tuple[str, Callable, float]]:
        reg = [
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
            reg.append(("ML_Ensemble", self._ml_ensemble, 1.0))
        return reg

    def _classify_regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        H = Indicators.hurst(h1["close"])
        E = Indicators.shannon_entropy(h1["close"])
        if H > self.cfg.hurst_trend_min and E < self.cfg.entropy_chaos_max:
            return "TRENDING", H, E
        if H < self.cfg.hurst_meanrev_max:
            return "MEAN_REVERTING", H, E
        return "CHAOTIC", H, E

    @staticmethod
    def _htf_bias(h4: pd.DataFrame) -> str:
        if len(h4) < 50: return "NEUTRAL"
        c = h4["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return "BEARISH"
        return "NEUTRAL"

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame) -> SignalResult:
        res = SignalResult()
        adx_val = Indicators.adx_last(m5)
        res.adx = adx_val
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min} (Konsolidasi)"
            return res

        regime, H, E = self._classify_regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E
        if regime == "CHAOTIC":
            res.skip_reason = f"Pasar Chaotic (Hurst={H:.2f}, Entropy={E:.2f})"
            return res

        h4_trend = self._htf_bias(h4)
        buy_w, sell_w, active = 0.0, 0.0, 0
        details = []

        for name, fn, weight in self._registry:
            try:
                vote, conf = fn(m5)
            except Exception:
                vote, conf = 0, 0.0
            if vote == 0:
                details.append({"model": name, "vote": "NEUTRAL", "weight": weight, "conf": round(conf, 2), "score": 0.0})
                continue
            active += 1
            score = weight * conf
            if vote > 0:
                buy_w += score
                v_label = "BUY"
            else:
                sell_w += score
                v_label = "SELL"
            details.append({"model": name, "vote": v_label, "weight": weight, "conf": round(conf, 2), "score": round(score, 2)})

        res.buy_score, res.sell_score, res.active_models, res.votes_detail = round(buy_w, 2), round(sell_w, 2), active, details
        tot = buy_w + sell_w
        if tot < 1e-6:
            res.skip_reason = "Tidak ada model aktif"
            return res

        res.buy_ratio, res.sell_ratio = buy_w / tot, sell_w / tot
        if active < self.cfg.parliament_min_active:
            res.skip_reason = f"Kuorum aktif {active} < {self.cfg.parliament_min_active}"
            return res

        dom = max(res.buy_ratio, res.sell_ratio)
        if dom >= self.cfg.parliament_full_ratio: res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio: res.verdict = "EXECUTE_HALF"
        elif dom >= self.cfg.parliament_min_ratio: res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = f"Konsensus {dom:.0%} < minimum {self.cfg.parliament_min_ratio:.0%}"
            return res

        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH":
                res.skip_reason = "BUY diblokir: H4 BEARISH"
                return res
            res.direction = "BUY"
        else:
            if h4_trend == "BULLISH":
                res.skip_reason = "SELL diblokir: H4 BULLISH"
                return res
            res.direction = "SELL"
        return res

    @staticmethod
    def detect_order_block(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[dict]:
        if len(df) < lookback + 5: return None
        tail, atr = df.tail(lookback + 5).reset_index(drop=True), Indicators.atr_last(df)
        if atr <= 0: return None
        for i in range(len(tail) - 2, 2, -1):
            if abs(tail["close"].iloc[i] - tail["open"].iloc[i]) < 1.2 * atr: continue
            is_bull = tail["close"].iloc[i] > tail["open"].iloc[i]
            if side == "BUY" and not is_bull: continue
            if side == "SELL" and is_bull: continue
            for j in range(i - 1, max(i - 10, 0), -1):
                if side == "BUY" and tail["close"].iloc[j] < tail["open"].iloc[j]:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
                if side == "SELL" and tail["close"].iloc[j] > tail["open"].iloc[j]:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
        return None

    def compute_levels(self, m5: pd.DataFrame, side: str, ob: Optional[dict]) -> Tuple[float, float, float, float, float]:
        atr = Indicators.atr_last(m5)
        if atr <= 0: raise ValueError("ATR <= 0")
        price, dec, buf = float(m5["close"].iloc[-1]), self.cfg.price_decimals, self.cfg.spread_buffer * 2.0
        ob_h, ob_l = (ob["high"], ob["low"]) if ob else (price, price)

        if side == "BUY":
            entry = ob_h - (ob_h - ob_l) * 0.5 if ob_h < price else price - atr * self.cfg.entry_pullback_atr
            entry = min(entry, price - atr * 0.05)
            if abs(price - entry) > self.cfg.entry_max_chase_atr * atr: raise ValueError("Entry terlalu jauh")
            sl = min(ob_l, entry) - buf
            if entry <= sl: raise ValueError("BUY entry <= sl")
            return round(entry, dec), round(sl, dec), round(entry + atr * self.cfg.atr_tp1_mult, dec), round(entry + atr * self.cfg.atr_tp2_mult, dec), round(entry + atr * self.cfg.atr_tp3_mult, dec)
        else:
            entry = ob_l + (ob_h - ob_l) * 0.5 if ob_l > price else price + atr * self.cfg.entry_pullback_atr
            entry = max(entry, price + atr * 0.05)
            if abs(entry - price) > self.cfg.entry_max_chase_atr * atr: raise ValueError("Entry terlalu jauh")
            sl = max(ob_h, entry) + buf
            if sl <= entry: raise ValueError("SELL sl <= entry")
            return round(entry, dec), round(sl, dec), round(entry - atr * self.cfg.atr_tp1_mult, dec), round(entry - atr * self.cfg.atr_tp2_mult, dec), round(entry - atr * self.cfg.atr_tp3_mult, dec)


# ═════════════════════════════════════════════════════════════════════════════
# 7. RISK MANAGEMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        base = self.cfg.account_balance
        journal = [j for j in (self.kv.get("journal", []) or []) if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS") and j.get("r_achieved") is not None]
        journal.sort(key=lambda x: x.get("timestamp", ""))
        for t in journal:
            try:
                base += float(t.get("r_achieved", 0.0)) * (base * (float(t.get("risk_pct", 1.0)) / 100.0))
            except Exception:
                continue
        return max(10.0, round(base, 2))

    def compute_lot(self, balance: float, entry: float, sl: float, conf_pct: float) -> Tuple[float, float]:
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-4: raise ValueError("Jarak SL terlalu kecil")
        risk_pct = max(0.1, self.cfg.max_risk_pct_per_trade * max(0.4, min(1.0, conf_pct / 100.0)))
        raw_lot = (balance * (risk_pct / 100.0)) / (sl_dist * self.cfg.contract_size)
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(raw_lot, 2)))
        notional = lot * self.cfg.contract_size * entry
        if balance > 0 and (notional / balance) > self.cfg.hard_leverage_cap:
            lot = max(self.cfg.min_lot, round((self.cfg.hard_leverage_cap * balance) / (self.cfg.contract_size * entry), 2))
        return lot, round((lot * self.cfg.contract_size * sl_dist) / balance * 100.0, 2)

    def _get_journal(self) -> list:
        return self.kv.get("journal", []) or []

    def _today_signals_filled(self) -> int:
        today = datetime.now(UTC).date().isoformat()
        return sum(1 for j in self._get_journal() if j.get("timestamp", "").startswith(today) and j.get("filled"))

    def _daily_loss_pct(self, balance: float) -> float:
        today = datetime.now(UTC).date().isoformat()
        tot = sum(abs(float(j.get("r_achieved", -1.0))) * (float(j.get("risk_pct", 1.0)) / 100.0) for j in self._get_journal() if j.get("timestamp", "").startswith(today) and j.get("outcome") == "LOSS")
        return tot * 100.0 if balance > 0 else 0.0

    def _consecutive_losses(self) -> int:
        closed = [j for j in self._get_journal() if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        count = 0
        for t in closed:
            if t.get("outcome") == "LOSS": count += 1
            else: break
        return count

    def _in_drawdown_pause(self) -> bool:
        last = self.kv.get("last_loss_ts", "")
        if not last: return False
        try:
            dt = datetime.fromisoformat(last)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)
            return (datetime.now(UTC) - dt) < timedelta(hours=self.cfg.dd_pause_hours)
        except Exception:
            return False

    def check_circuit_breakers(self, balance: float) -> Tuple[bool, str]:
        signals = self._today_signals_filled()
        if signals >= self.cfg.max_daily_signals: return False, f"MAX_DAILY_SIGNALS ({signals}/{self.cfg.max_daily_signals})"
        loss = self._daily_loss_pct(balance)
        if loss >= self.cfg.max_daily_loss_pct: return False, f"MAX_DAILY_LOSS_LOCK ({loss:.1f}% >= {self.cfg.max_daily_loss_pct}%)"
        consec = self._consecutive_losses()
        if consec >= self.cfg.max_consecutive_losses: return False, f"MAX_CONSECUTIVE_LOSS ({consec} >= {self.cfg.max_consecutive_losses})"
        if self._in_drawdown_pause(): return False, "DRAWDOWN_PAUSE (Cooling-off 24h)"
        return True, ""


# ═════════════════════════════════════════════════════════════════════════════
# 8. NEWS SAFETY FILTER
# ═════════════════════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg, self.kv, self.http = cfg, kv, http

    def is_blackout(self) -> Tuple[bool, str]:
        if self.cfg.news_blackout_min <= 0: return False, ""
        cached, now_ts = self.kv.get("news_cache", {}) or {}, time.time()
        ttl = self.cfg.news_cache_sec if cached.get("ok") else self.cfg.news_cache_sec_on_fail
        if now_ts - cached.get("ts", 0) < ttl: return bool(cached.get("blackout")), str(cached.get("title", ""))
        fails = int(self.kv.get("news_fail_count", 0))

        try:
            r = self.http.request("GET", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
            events, now = r.json(), datetime.now(UTC)
            in_blackout, title = False, ""
            for ev in events:
                if ev.get("country") != "USD" or ev.get("impact") != "High": continue
                try:
                    t = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                    if t.tzinfo is None: t = t.replace(tzinfo=UTC)
                except Exception:
                    continue
                if abs((t - now).total_seconds()) / 60.0 <= self.cfg.news_blackout_min:
                    in_blackout, title = True, ev.get("title", "High-Impact USD News")
                    break
            self.kv.set("news_cache", {"ts": now_ts, "ok": True, "blackout": in_blackout, "title": title})
            self.kv.set("news_fail_count", 0)
            return in_blackout, title
        except Exception as e:
            fails += 1
            self.kv.set("news_fail_count", fails)
            self.kv.set("news_cache", {"ts": now_ts, "ok": False, "blackout": False, "title": ""})
            if fails >= self.cfg.news_max_consecutive_fail: return True, "EMERGENCY_NEWS_FEED_FAIL_GUARD"
            return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# 9. TELEGRAM NOTIFIER
# ═════════════════════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg, self.kv, self.http = cfg, kv, http
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def send(self, message: str) -> bool:
        if self.cfg.dry_run: message = "🧪 <b>[DRY RUN — APEX v18.0]</b>\n" + message
        if not (self.token and self.chat_id):
            if self.cfg.dry_run: logging.info(f"[TG DRY]\n{redact(message)}")
            return True
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
        now = datetime.now(UTC)
        dedup = {k: v for k, v in (self.kv.get("tg_hashes", {}) or {}).items() if (now - datetime.fromisoformat(v)).total_seconds() < 3600}
        if digest in dedup: return True
        try:
            resp = self.http.request("POST", f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True})
            if resp.json().get("ok"):
                dedup[digest] = now.isoformat()
                self.kv.set("tg_hashes", dedup)
                return True
        except Exception as e:
            logging.error(f"TG kirim gagal: {redact(e)}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 10. OUTCOME TRACKER
# ═════════════════════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg, self.kv = cfg, kv

    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        events, now, journal, modified = [], datetime.now(UTC), (self.kv.get("journal", []) or []), False
        for entry in journal:
            if entry.get("outcome") not in ("PENDING", None): continue
            anchor = entry.get("signal_bar_dt") or entry.get("timestamp")
            try: anchor_ts = pd.Timestamp(anchor, tz="UTC")
            except Exception: continue
            if (now - anchor_ts.to_pydatetime()) > timedelta(hours=self.cfg.pending_max_age_hours):
                entry["outcome"], entry["r_achieved"] = "STALE", 0.0
                events.append(("STALE", dict(entry)))
                modified = True
                continue
            side, ep, sl, tp1, tp2, tp3 = entry["signal"], float(entry["entry"]), float(entry["sl"]), float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
            if ep <= 0 or sl <= 0: continue
            bars = m5[m5["datetime"] > anchor_ts].sort_values("datetime")
            if bars.empty: continue
            filled, fill_idx = bool(entry.get("filled")), None
            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep: filled, fill_idx = True, idx; break
                    if side == "SELL" and b.high >= ep: filled, fill_idx = True, idx; break
                if not filled:
                    chase = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    if (side == "BUY" and float(bars["high"].iloc[-1]) > ep + chase) or (side == "SELL" and float(bars["low"].iloc[-1]) < ep - chase):
                        entry["outcome"], entry["filled"], entry["r_achieved"] = "INVALIDATED", 0, 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                    continue
                entry["filled"] = 1
                modified = True

            walk = bars.iloc[fill_idx:] if fill_idx is not None else bars
            h1_, h2, h3, hs, hb = bool(entry.get("hit_tp1")), bool(entry.get("hit_tp2")), bool(entry.get("hit_tp3")), bool(entry.get("hit_sl")), bool(entry.get("hit_be"))
            closed = False
            for b in walk.itertuples(index=False):
                if h3 or hs or hb: break
                csl = ep if h1_ else sl
                if side == "BUY":
                    if b.low <= csl:
                        if h1_: hb = True
                        else: hs = True
                        continue
                    if not h1_ and b.high >= tp1: h1_ = True
                    if not h2 and b.high >= tp2: h2 = True
                    if not h3 and b.high >= tp3: h3 = True
                else:
                    if b.high >= csl:
                        if h1_: hb = True
                        else: hs = True
                        continue
                    if not h1_ and b.low <= tp1: h1_ = True
                    if not h2 and b.low <= tp2: h2 = True
                    if not h3 and b.low <= tp3: h3 = True

            entry["hit_tp1"], entry["hit_tp2"], entry["hit_tp3"], entry["hit_sl"], entry["hit_be"] = int(h1_), int(h2), int(h3), int(hs), int(hb)
            if entry.get("outcome") == "PENDING":
                if h3: entry["outcome"] = "WIN_TP3"
                elif h2 or hb: entry["outcome"] = "WIN_BE"
                elif hs: entry["outcome"] = "LOSS"
                if entry["outcome"] != "PENDING":
                    dist = abs(ep - sl)
                    r1, r2, r3 = abs(tp1 - ep) / dist, abs(tp2 - ep) / dist, abs(tp3 - ep) / dist
                    if h3: entry["r_achieved"] = round((r1 + r2 + r3) / 3.0, 3)
                    elif h2: entry["r_achieved"] = round((r1 + r2) / 3.0, 3)
                    elif h1_ and hb: entry["r_achieved"] = round(r1 / 3.0, 3)
                    elif hb: entry["r_achieved"] = 0.0
                    else: entry["r_achieved"] = -1.0
                    events.append(("CLOSE", dict(entry)))
                    modified, closed = True, True
                    if entry["outcome"] == "LOSS": self.kv.set("last_loss_ts", now.isoformat())
            if not closed and (h1_ or h2 or h3 or hs or hb): modified = True
        if modified: self.kv.set("journal", journal)
        return events


# ═════════════════════════════════════════════════════════════════════════════
# 11. ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HTTPClient(cfg)
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http, self.kv)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.kv, self.http)
        self.tg = TelegramNotifier(cfg, self.kv, self.http)
        self.outcome = OutcomeTracker(cfg, self.kv)

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter: return True
        ld, ny = now.astimezone(LONDON_TZ), now.astimezone(NY_TZ)
        lh, nh = ld.hour + ld.minute / 60.0, ny.hour + ny.minute / 60.0
        return (7.0 <= lh < 10.0) or (8.0 <= nh < 11.0) or (13.0 <= nh < 16.0)

    def run(self) -> None:
        try: self.kv.force_load()
        except StateError as e: logging.error(f"State load: {e}"); return
        now, now_str = datetime.now(UTC), datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        if self.cfg.kill_switch: logging.info("KILL SWITCH AKTIF"); return
        if now.weekday() >= 5: logging.info("Akhir pekan — pasar tutup"); return
        if not self._in_killzone(now): logging.info("Di luar sesi killzone"); return

        balance = self.risk.dynamic_balance()
        ok, reason = self.risk.check_circuit_breakers(balance)
        if not ok: logging.info(f"Circuit Breaker: {reason}"); return

        in_blackout, news_title = self.news.is_blackout()
        if in_blackout: logging.info(f"News Blackout: [{news_title}]"); return

        try: dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            err = redact(e)
            logging.error(f"Data error: {err}")
            self.tg.send(f"⚠️ <b>Market Data Error</b>\n<code>{err[:200]}</code>")
            return

        try:
            for evt_type, trade in self.outcome.update(dfs["5min"]):
                if evt_type == "CLOSE":
                    self.tg.send(f"📕 <b>Closed</b> {trade['signal']} {trade['symbol']}\nEntry: <code>{trade['entry']}</code> → Result: <b>{trade['outcome']}</b> ({trade.get('r_achieved', 0):+.2f}R)")
                elif evt_type in ("INVALIDATED", "STALE"):
                    self.tg.send(f"⚪ <b>Order {evt_type}</b>: {trade['signal']} Limit @ <code>{trade['entry']}</code>")
        except Exception as e: logging.warning(f"Tracker: {e}")

        balance = self.risk.dynamic_balance()
        m5_c = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15_c = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1_c = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4_c = dfs["4h"].iloc[:-1].reset_index(drop=True)

        if len(m5_c) < self.cfg.min_bars_m5 or len(h1_c) < self.cfg.min_bars_h1: return
        last_price = float(m5_c["close"].iloc[-1])
        sig_dt = m5_c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("Posisi terbuka sedang aktif — skip"); self._update_state(last_price, now_str); return
        if sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled")) >= 2:
            logging.info("Pending limit penuh"); self._update_state(last_price, now_str); return

        result = self.signal.generate(m5_c, m15_c, h1_c, h4_c)
        logging.info(f"Parlemen: {result.direction} BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%} Verdict={result.verdict} ({result.skip_reason})")
        if result.direction == "NEUTRAL" or result.verdict == "CANCEL": self._update_state(last_price, now_str); return

        state = self.kv.get("state", {}) or {}
        cool = self.cfg.opposite_cooldown_minutes if state.get("last_signal_type", "NEUTRAL") != result.direction else self.cfg.cooldown_minutes
        last_sig_time = state.get("last_signal_ts", "")
        if last_sig_time:
            try:
                if (now - datetime.strptime(last_sig_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)) < timedelta(minutes=cool):
                    logging.info(f"Cooldown aktif ({cool}m)"); self._update_state(last_price, now_str); return
            except Exception: pass

        ob = self.signal.detect_order_block(m5_c, result.direction)
        try: entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5_c, result.direction, ob)
        except ValueError as e: logging.warning(f"Levels: {e}"); self._update_state(last_price, now_str); return

        try: lot, risk_pct = self.risk.compute_lot(balance, entry, sl, max(result.buy_ratio, result.sell_ratio) * 100.0)
        except ValueError as e: logging.warning(f"Sizing: {e}"); self._update_state(last_price, now_str); return

        if result.verdict == "EXECUTE_HALF": lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif result.verdict == "EXECUTE_MICRO": lot = max(self.cfg.min_lot, round(lot / 4.0, 2))

        action_name = f"{result.direction} LIMIT"
        ob_disp = f"✅ {ob['high']:.2f}–{ob['low']:.2f}" if ob else "⚠️ Buffer Dinamis ATR"
        tg_msg = (
            f"🏛️ <b>APEX QUANT v{VERSION} — {action_name}</b>\n\n"
            f"Asset: <b>{self.cfg.symbol}</b>\n"
            f"Order Type: <b>{action_name}</b>\n"
            f"Entry Price: <code>{entry:.2f}</code>\n"
            f"Stop Loss: <code>{sl:.2f}</code>\n\n"
            f"🎯 TP1: <code>{tp1:.2f}</code> (1/3 Close + Move SL to BE)\n"
            f"🎯 TP2: <code>{tp2:.2f}</code> (1/3 Close)\n"
            f"🎯 TP3: <code>{tp3:.2f}</code> (Runner Target)\n\n"
            f"📦 <b>Suggested Lot: <code>{lot}</code></b>\n"
            f"💰 Balance: <code>${balance:.2f}</code> | Risk: <code>{risk_pct:.2f}%</code>\n"
            f"📊 Consensus: BUY={result.buy_ratio:.0%} | SELL={result.sell_ratio:.0%}\n"
            f"🧭 Regime: {result.regime} (Hurst={result.hurst:.2f}) | ADX: {result.adx:.1f}\n"
            f"🧱 Order Block: {ob_disp}\n\n"
            f"📱 <i>Salin harga di atas dan eksekusi Limit Order di MT5 Mobile.</i>"
        )

        record = {"id": f"{now_str.replace(' ', 'T')}_{result.direction}_{lot}", "timestamp": now_str, "signal_bar_dt": sig_dt, "symbol": self.cfg.symbol, "signal": result.direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "lot": lot, "verdict": result.verdict, "outcome": "PENDING", "filled": 0, "r_achieved": 0.0, "risk_pct": risk_pct}
        if self._append_journal(record):
            if self.tg.send(tg_msg):
                state["last_signal_type"] = result.direction
                state["last_signal_ts"] = now_str
                self.kv.set("state", state)
                logging.info(f"Sinyal terkirim: {action_name} @ {entry} Lot={lot}")

        self._update_state(last_price, now_str)

    def _update_state(self, last_price: float, now_str: str) -> None:
        st = self.kv.get("state", {}) or {}
        st["run_count"] = int(st.get("run_count", 0)) + 1
        st["last_price"], st["last_price_ts"] = last_price, now_str
        self.kv.set("state", st)

    def _append_journal(self, record: dict) -> bool:
        if self.cfg.dry_run:
            logging.info(f"[DRY] Journal: {record.get('id')} {record.get('signal')} @ {record.get('entry')} Lot={record.get('lot')}")
            return True
        j = [r for r in (self.kv.get("journal", []) or []) if r.get("id") != record.get("id")]
        j.append(record)
        j.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return self.kv.set("journal", j[: self.cfg.max_journal])


# ═════════════════════════════════════════════════════════════════════════════
# 12. CLI ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
def cmd_smoke(cfg: Config) -> None:
    print("=" * 65)
    print(f"APEX PREDATOR v{VERSION} — SMOKE TEST")
    print("=" * 65)
    http, kv = HTTPClient(cfg), KVStore(cfg, HTTPClient(cfg))
    data, sig = MarketDataHandler(cfg, http, kv), SignalGenerator(cfg)
    dfs, src, offset = data.fetch_all()
    m5_c, m15_c, h1_c, h4_c = dfs["5min"].iloc[:-1].reset_index(drop=True), dfs["15min"].iloc[:-1].reset_index(drop=True), dfs["1h"].iloc[:-1].reset_index(drop=True), dfs["4h"].iloc[:-1].reset_index(drop=True)
    print(f"Sumber: {src} | Offset MT5: {offset:+.3f}")
    print(f"Bar: M5={len(m5_c)}, M15={len(m15_c)}, H1={len(h1_c)}, H4={len(h4_c)}")
    print(f"Harga M5 Terakhir: {m5_c.iloc[-1]['datetime']} | Close: ${m5_c.iloc[-1]['close']:.2f}")
    res = sig.generate(m5_c, m15_c, h1_c, h4_c)
    print(f"Hasil: BUY={res.buy_ratio:.0%} SELL={res.sell_ratio:.0%} | Arah={res.direction} | Verdict={res.verdict}")
    if res.direction != "NEUTRAL":
        ob = sig.detect_order_block(m5_c, res.direction)
        e, s, t1, t2, t3 = sig.compute_levels(m5_c, res.direction, ob)
        print(f"Tingkat: Entry={e} SL={s} TP1={t1} TP2={t2} TP3={t3}")
    print("=" * 65)


def cmd_selftest(cfg: Config) -> None:
    errors = []
    if cfg.max_lot < cfg.min_lot: errors.append("max_lot < min_lot")
    if not (cfg.parliament_min_ratio < cfg.parliament_half_ratio < cfg.parliament_full_ratio): errors.append("Rasio parlemen salah")
    if cfg.account_balance <= 0: errors.append("Saldo <= 0")
    if errors:
        print("❌ GAGAL:\n - " + "\n - ".join(errors))
        sys.exit(1)
    print("✅ SELF-TEST BERHASIL: Semua parameter valid!")


def cmd_test_telegram(cfg: Config) -> None:
    tg = TelegramNotifier(cfg, KVStore(cfg, HTTPClient(cfg)), HTTPClient(cfg))
    ok = tg.send(f"🔔 <b>APEX v{VERSION} TELEGRAM TEST</b>\nStatus: <b>ONLINE & TERHUBUNG</b>")
    print("✅ TELEGRAM TERKIRIM" if ok else "❌ TELEGRAM GAGAL")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"APEX Predator Engine v{VERSION}")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "smoke", "selftest", "test-tg"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    if args.command == "selftest": cmd_selftest(cfg); return
    if args.command == "smoke": cmd_smoke(cfg); return
    if args.command == "test-tg": cmd_test_telegram(cfg); return

    try:
        TradingBotOrchestrator(cfg).run()
    except Exception as e:
        logging.exception("CRITICAL EXCEPTION")
        try: TelegramNotifier(cfg, KVStore(cfg, HTTPClient(cfg)), HTTPClient(cfg)).send(f"💥 <b>APEX CRASH</b>\n<code>{redact(e)[:350]}</code>")
        except Exception: pass
        sys.exit(1)


if __name__ == "__main__":
    main()
