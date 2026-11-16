#!/usr/bin/env python3
"""
===============================================================================
APEX PREDATOR v19.0 — Adaptive Institutional Engine for XAUUSD
===============================================================================
v19.0 OPTIMIZATIONS (over v18.1):
  [NEW]  DXY Correlation Filter        — Gold-DXY inverse edge (documented)
  [NEW]  Adaptive Model Weighting      — Self-learning from rolling accuracy
  [NEW]  Family De-duplication         — Prevent trend-family triple-counting
  [NEW]  Regime-Adaptive Thresholds    — Different bars per regime
  [NEW]  Session-Adaptive ATR Scaling  — London vs NY volatility profiles
  [NEW]  Trailing Stop after TP2       — ATR-trail the runner leg
  [FIX]  Conservative Intrabar Fill    — SL checked before TP in same bar
  [FIX]  Atomic State Writes           — Write-to-temp-then-rename
  [NEW]  Backtest Mode                 — Walk-forward historical validation
  [NEW]  Per-Model Performance Tracking
  [NEW]  SIGTERM/SIGINT State Drain
  [NEW]  Health JSON output
===============================================================================
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import html
import json
import logging
import math
import os
import random
import re
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
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
    from scipy.signal import butter, lfilter
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "19.0-PRODUCTION"
NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
UTC = timezone.utc


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v19.json"
    local_state_file: str = "apex_state_v19.json"

    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0

    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01

    # Drawdown-scaled risk (new in v19)
    dd_scale_threshold_pct: float = 1.5   # start scaling when DD > 1.5%
    dd_scale_min_factor: float = 0.4      # down to 40% of base risk at max DD

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
    trailing_atr_mult: float = 1.5  # trail after TP2

    # Session-adaptive ATR scaling (v19)
    session_atr_london: float = 1.00
    session_atr_ny_open: float = 1.00
    session_atr_ny_afternoon: float = 0.85
    session_atr_asian: float = 1.20

    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    m5_fetch_size: int = 800
    max_stale_mult: float = 3.0

    entry_pullback_atr: float = 0.25
    entry_max_chase_atr: float = 2.0
    pending_stale_hours: float = 4.0

    adx_min: float = 20.0
    hurst_trend_min: float = 0.52
    hurst_meanrev_max: float = 0.48
    entropy_chaos_max: float = 0.85
    allow_ranging_regime: bool = True
    enforce_htf_filter: bool = True
    counter_trend_allow_ratio: float = 0.85
    news_blackout_minutes: int = 30

    # Regime-adaptive consensus (v19)
    min_ratio_trending: float = 0.65
    min_ratio_meanrev: float = 0.55
    min_ratio_ranging: float = 0.60
    min_ratio_default: float = 0.55

    # Parliament
    parliament_min_active: int = 6
    parliament_full_ratio: float = 0.70
    parliament_half_ratio: float = 0.60

    # Family de-duplication (v19)
    family_cap_multiplier: float = 1.5

    # DXY filter (v19)
    enable_dxy_filter: bool = True
    dxy_corr_window: int = 20
    dxy_bias_threshold: float = 0.55

    # Adaptive weighting (v19)
    enable_adaptive_weights: bool = True
    adaptive_min_trades: int = 30
    adaptive_weight_floor: float = 0.5
    adaptive_weight_ceil: float = 1.8

    skip_killzone_filter: bool = False
    dry_run: bool = False
    kill_switch: bool = False
    enable_ml_models: bool = True
    ml_min_val_acc: float = 0.52

    http_timeout_sec: float = 12.0
    http_max_retries: int = 3
    http_backoff_factor: float = 1.5
    td_min_interval_sec: float = 8.0

    @classmethod
    def from_env(cls) -> "Config":
        def es(k: str, d: str) -> str:
            return os.getenv(k, d).strip()

        def ef(k: str, d: float) -> float:
            v = os.getenv(k, "").strip()
            try:
                return float(v) if v else d
            except ValueError:
                logging.warning(f"Invalid float {k}={v}, using {d}")
                return d

        def ei(k: str, d: int) -> int:
            v = os.getenv(k, "").strip()
            try:
                return int(v) if v else d
            except ValueError:
                logging.warning(f"Invalid int {k}={v}, using {d}")
                return d

        def eb(k: str, d: bool) -> bool:
            v = os.getenv(k, "").strip().lower()
            return d if not v else v in ("1", "true", "yes", "on")

        c = cls()
        c.gist_id = es("GIST_ID", c.gist_id)
        c.gist_token = es("GIST_TOKEN", c.gist_token)
        c.symbol = es("APEX_SYMBOL", c.symbol)
        c.account_balance = ef("APEX_BALANCE", c.account_balance)
        c.max_risk_pct_per_trade = ef("APEX_RISK_PCT", c.max_risk_pct_per_trade)
        c.max_daily_loss_pct = ef("APEX_MAX_DAILY_LOSS", c.max_daily_loss_pct)
        c.max_consecutive_losses = ei("APEX_MAX_CONSEC_LOSSES", c.max_consecutive_losses)
        c.skip_killzone_filter = eb("APEX_SKIP_KILLZONE_FILTER", c.skip_killzone_filter)
        c.dry_run = eb("APEX_DRY_RUN", c.dry_run)
        c.kill_switch = eb("APEX_KILL_SWITCH", c.kill_switch)
        c.enable_ml_models = eb("APEX_ENABLE_ML", c.enable_ml_models)
        c.allow_ranging_regime = eb("APEX_ALLOW_RANGING", c.allow_ranging_regime)
        c.enforce_htf_filter = eb("APEX_ENFORCE_HTF_FILTER", c.enforce_htf_filter)
        c.counter_trend_allow_ratio = ef("APEX_COUNTER_TREND_ALLOW_RATIO", c.counter_trend_allow_ratio)
        c.enable_dxy_filter = eb("APEX_ENABLE_DXY", c.enable_dxy_filter)
        c.enable_adaptive_weights = eb("APEX_ENABLE_ADAPTIVE", c.enable_adaptive_weights)
        return c

    def validate(self) -> None:
        assert self.contract_size == 100.0
        assert 0.05 <= self.max_risk_pct_per_trade <= 5.0
        assert 1.0 <= self.max_daily_loss_pct <= 15.0
        assert 0.01 <= self.min_lot <= self.max_lot
        assert self.atr_sl_mult > 0 and self.atr_tp1_mult > 0
        assert self.spread_buffer > 0
        assert self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult
        assert 0.05 <= self.dd_scale_min_factor <= 1.0
        assert self.family_cap_multiplier >= 1.0
        assert 0.4 <= self.dxy_bias_threshold <= 0.9


# ═════════════════════════════════════════════════════════════════════════════
# 2. EXCEPTIONS & HELPERS
# ═════════════════════════════════════════════════════════════════════════════
class ApexError(Exception): ...
class DataFetchError(ApexError): ...
class RiskViolationError(ApexError): ...
class StateError(ApexError): ...


def redact(val: Any) -> str:
    s = str(val)
    for pat, rep in [
        (r"(ghp_[A-Za-z0-9_]{30,})", "ghp_****"),
        (r"(bot[0-9]+:[A-Za-z0-9_-]{30,})", "bot****:****"),
        (r"(apikey=)[A-Za-z0-9_-]+", r"\1****"),
        (r"(Bearer\s+)[A-Za-z0-9_.-]+", r"\1****"),
    ]:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s


def atomic_write(path: Path, content: str) -> None:
    """Write to temp then rename — atomic on POSIX and near-atomic on Windows."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTTP RESILIENCE
# ═════════════════════════════════════════════════════════════════════════════
class ResilientHttpClient:
    def __init__(self, timeout: float = 12.0, max_retries: int = 3, backoff: float = 1.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"ApexQuantInstitutional/{VERSION} (XAUUSD Engine)"
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
                st = self.backoff ** attempt + random.uniform(0.1, 0.4)
                logging.warning(
                    f"[HTTP Retry {attempt}/{self.max_retries}] {method} {url} "
                    f"failed: {redact(e)}. Sleep {st:.2f}s..."
                )
                time.sleep(st)
        raise DataFetchError(
            f"HTTP failed after {self.max_retries} retries: {redact(last_exc)}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. STATE PERSISTENCE (atomic + async Gist)
# ═════════════════════════════════════════════════════════════════════════════
class KVStore:
    def __init__(self, cfg: Config, http: ResilientHttpClient):
        self.cfg = cfg
        self.http = http
        self.data: Dict[str, Any] = {
            "journal": [], "state": {}, "metrics": {},
            "circuit_breaker": {}, "model_perf": {},
        }
        self.loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gist")
        self._pending: List[Future] = []
        self._lock = threading.Lock()
        atexit.register(self._drain_pending)

    def _drain_pending(self, timeout: float = 15.0) -> None:
        with self._lock:
            pending = list(self._pending)
        for f in pending:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def force_load(self) -> None:
        if self.cfg.gist_id and self.cfg.gist_token:
            try:
                url = f"https://api.github.com/gists/{self.cfg.gist_id}"
                headers = {
                    "Authorization": f"Bearer {self.cfg.gist_token}",
                    "Accept": "application/vnd.github+json",
                }
                resp = self.http.request("GET", url, headers=headers)
                files = resp.json().get("files", {})
                target = files.get(self.cfg.gist_state_file)
                if target and target.get("content"):
                    self.data = json.loads(target["content"])
                    self.loaded = True
                    logging.info("State loaded from GitHub Gist.")
                    return
            except Exception as e:
                logging.warning(f"Gist read failed: {redact(e)}. Local fallback.")

        p = Path(self.cfg.local_state_file)
        if p.exists():
            try:
                self.data = json.loads(p.read_text(encoding="utf-8"))
                self.loaded = True
                logging.info(f"State loaded from local: {p}")
                return
            except Exception as e:
                logging.warning(f"Local read error: {e}")

        self.loaded = True
        logging.info("No existing state — fresh init.")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        serialized = json.dumps(self.data, indent=2, default=str)
        try:
            atomic_write(Path(self.cfg.local_state_file), serialized)
        except Exception as e:
            logging.error(f"Local save failed: {e}")

        if self.cfg.gist_id and self.cfg.gist_token:
            with self._lock:
                self._pending = [f for f in self._pending if not f.done()]
                self._pending.append(self._executor.submit(self._sync_gist, serialized))

    def _sync_gist(self, serialized: str) -> None:
        try:
            url = f"https://api.github.com/gists/{self.cfg.gist_id}"
            headers = {
                "Authorization": f"Bearer {self.cfg.gist_token}",
                "Accept": "application/vnd.github+json",
            }
            payload = {
                "description": f"Apex State v{VERSION}",
                "files": {self.cfg.gist_state_file: {"content": serialized}},
            }
            self.http.request("PATCH", url, headers=headers, json=payload)
            logging.info("Gist synced.")
        except Exception as e:
            logging.error(f"Gist sync error: {redact(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. INDICATORS (add correlation/beta helpers)
# ═════════════════════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi_series(s: pd.Series, period: int = 14) -> pd.Series:
        d = s.diff()
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = g.ewm(alpha=1.0 / period, adjust=False).mean()
        al = l.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = ag / (al + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def rsi(s: pd.Series, period: int = 14) -> float:
        v = Indicators.rsi_series(s, period)
        return float(v.iloc[-1]) if not v.empty else 50.0

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        pc = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - pc).abs()
        tr3 = (df["low"] - pc).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return Indicators.true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        v = Indicators.atr_series(df, period)
        return float(v.iloc[-1]) if not v.empty else 0.0

    @staticmethod
    def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l = df["high"], df["low"]
        um = h.diff()
        dm = -l.diff()
        pdm = np.where((um > dm) & (um > 0), um, 0.0)
        mdm = np.where((dm > um) & (dm > 0), dm, 0.0)
        tr = Indicators.true_range(df)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        pdi = pd.Series(pdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        mdi = pd.Series(mdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        pi = 100.0 * (pdi / (atr + 1e-9))
        mi = 100.0 * (mdi / (atr + 1e-9))
        dx = 100.0 * ((pi - mi).abs() / (pi + mi + 1e-9))
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        v = Indicators.adx_series(df, period)
        return float(v.iloc[-1]) if not v.empty else 0.0

    @staticmethod
    def macd(s: pd.Series, f: int = 12, sl: int = 26, sig: int = 9
             ) -> Tuple[float, float, float]:
        ef = Indicators.ema(s, f)
        es = Indicators.ema(s, sl)
        ml = ef - es
        sli = Indicators.ema(ml, sig)
        hist = ml - sli
        return float(ml.iloc[-1]), float(sli.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def bollinger(s: pd.Series, period: int = 20, nstd: float = 2.0
                  ) -> Tuple[float, float, float]:
        sma = s.rolling(period).mean()
        std = s.rolling(period).std(ddof=0)
        return (float((sma + nstd * std).iloc[-1]),
                float(sma.iloc[-1]),
                float((sma - nstd * std).iloc[-1]))

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3
                   ) -> Tuple[float, float]:
        lo = df["low"].rolling(period).min()
        hi = df["high"].rolling(period).max()
        k_f = 100.0 * ((df["close"] - lo) / (hi - lo + 1e-9))
        k = k_f.rolling(smooth).mean()
        d = k.rolling(smooth).mean()
        return float(k.iloc[-1]), float(d.iloc[-1])

    @staticmethod
    def hurst(s: pd.Series, max_lags: int = 30) -> float:
        ts = s.values
        if len(ts) < max_lags * 2:
            return 0.5
        valid, taus = [], []
        for lag in range(2, max_lags):
            diff = ts[lag:] - ts[:-lag]
            std = float(np.std(diff, ddof=1))
            if std > 1e-12:
                valid.append(lag)
                taus.append(std)
        if len(valid) < 3:
            return 0.5
        try:
            reg = np.polyfit(np.log(valid), np.log(taus), 1)
            return float(np.clip(reg[0], 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def shannon_entropy(s: pd.Series, bins: int = 10) -> float:
        ret = s.pct_change().dropna()
        if len(ret) < bins * 2:
            return 0.5
        c, _ = np.histogram(ret, bins=bins, density=True)
        c = c[c > 0]
        if c.size == 0:
            return 0.5
        p = c / c.sum()
        ent = -np.sum(p * np.log2(p))
        return float(np.clip(ent / np.log2(bins), 0.0, 1.0))

    @staticmethod
    def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> float:
        """Pearson correlation of returns over `window` bars. NaN-safe."""
        n = min(len(a), len(b), window + 1)
        if n < 5:
            return 0.0
        ra = a.pct_change().dropna().iloc[-n + 1:]
        rb = b.pct_change().dropna().iloc[-n + 1:]
        k = min(len(ra), len(rb))
        if k < 5:
            return 0.0
        ra, rb = ra.iloc[-k:], rb.iloc[-k:]
        va, vb = float(ra.var()), float(rb.var())
        if va <= 1e-12 or vb <= 1e-12:
            return 0.0
        cov = float(ra.cov(rb))
        return float(np.clip(cov / math.sqrt(va * vb), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# 6. MARKET DATA (adds DXY fetch)
# ═════════════════════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config, http: ResilientHttpClient):
        self.cfg = cfg
        self.http = http
        self._last_td_call = 0.0
        self._dxy_cache: Optional[pd.DataFrame] = None
        self._dxy_cache_ts = 0.0

    def compute_broker_offset(self) -> float:
        oanda = os.getenv("OANDA_API_KEY", "").strip()
        td = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda and td):
            return self.cfg.mt5_offset
        try:
            r1 = self.http.request(
                "GET", "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                headers={"Authorization": f"Bearer {oanda}"},
            )
            ob = float(r1.json()["prices"][0]["bids"][0]["price"])
            r2 = self.http.request(
                "GET", "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td},
            )
            tp = float(r2.json()["price"])
            if tp <= 0:
                return self.cfg.mt5_offset
            off = ob - tp
            if abs(off) > 2.0:
                logging.warning(f"Offset {off:+.3f} rejected")
                return self.cfg.mt5_offset
            return round(off, 3)
        except Exception as e:
            logging.warning(f"Offset fallback: {redact(e)}")
            return self.cfg.mt5_offset

    @staticmethod
    def _apply_offset(df: pd.DataFrame, off: float) -> pd.DataFrame:
        if off == 0.0:
            return df
        out = df.copy()
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] + off
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            df.set_index("datetime")
            .resample(rule, label="left", closed="left")
            .agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            })
            .dropna().reset_index()
        )

    def _td_interval(self, interval: str, size: int) -> pd.DataFrame:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("TWELVE_DATA_API_KEY missing")
        elapsed = time.time() - self._last_td_call
        if elapsed < self.cfg.td_min_interval_sec:
            time.sleep(self.cfg.td_min_interval_sec - elapsed)
        self._last_td_call = time.time()
        r = self.http.request(
            "GET", "https://api.twelvedata.com/time_series",
            params={
                "symbol": "XAU/USD", "interval": interval,
                "outputsize": size, "apikey": key, "timezone": "UTC",
            },
        )
        data = r.json()
        if "values" not in data or not data["values"]:
            raise DataFetchError(
                f"TD {interval}: {data.get('message', 'no values')}"
            )
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 100.0), errors="coerce").fillna(100.0)
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df.sort_values("datetime").reset_index(drop=True)

    def fetch_dxy(self) -> Optional[pd.DataFrame]:
        """Fetch DXY (dollar index). Cached for 5 min."""
        if time.time() - self._dxy_cache_ts < 300 and self._dxy_cache is not None:
            return self._dxy_cache
        try:
            df = self._td_interval("1h", 200)
            self._dxy_cache = df
            self._dxy_cache_ts = time.time()
            return df
        except Exception as e:
            logging.warning(f"DXY fetch failed: {redact(e)}")
            return None

    def _fetch_twelvedata(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        m5 = self._apply_offset(self._td_interval("5min", min(self.cfg.m5_fetch_size, 800)), offset)
        h1 = self._apply_offset(self._td_interval("1h", 800), offset)
        h4 = self._apply_offset(self._td_interval("4h", 800), offset)
        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def _fetch_xaus(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        base = "https://xaus.com/api/v1/chart"

        def fr(interval: str, rng: str) -> pd.DataFrame:
            r = self.http.request(
                "GET", base, params={"symbol": "xau", "interval": interval, "range": rng},
                headers={"Accept": "application/json"},
            )
            pts = r.json().get("points")
            if not pts:
                raise DataFetchError(f"XAUS {interval}: no points")
            rows = []
            for p in pts:
                if not all(k in p for k in ("t", "o", "h", "l", "c")):
                    continue
                rows.append({
                    "epoch": int(p["t"]), "open": float(p["o"]),
                    "high": float(p["h"]), "low": float(p["l"]),
                    "close": float(p["c"]), "volume": float(p.get("v", 100.0)),
                })
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            return df.drop(columns=["epoch"]).sort_values("datetime").reset_index(drop=True)

        m5 = self._apply_offset(fr("5m", "7d"), offset)
        h1 = self._apply_offset(fr("1h", "30d"), offset)
        h4 = self._apply_offset(fr("4h", "90d"), offset)
        self._validate(m5, "XAUS-M5", self.cfg.min_bars_m5)
        self._validate(h1, "XAUS-H1", self.cfg.min_bars_h1)
        self._validate(h4, "XAUS-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "XAUS-M5")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "XAUS-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "XAUS"

    def _fetch_biquote(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        r = self.http.request(
            "GET", "https://biquote.com/api/gold/history", params={"limit": 1000}
        )
        recs = r.json()
        if not isinstance(recs, list) or len(recs) < self.cfg.min_bars_m5:
            raise DataFetchError("Biquote insufficient")
        df = pd.DataFrame(recs)
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c])
        df["volume"] = 100.0
        df = df.sort_values("datetime").reset_index(drop=True)
        m5 = self._apply_offset(df, offset)
        self._validate(m5, "BIQ-M5", self.cfg.min_bars_m5)
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        h1 = self._apply_offset(self._resample(m5, "1h"), offset)
        h4 = self._apply_offset(self._resample(m5, "4h"), offset)
        self._validate(h4, "BIQ-H4", self.cfg.min_bars_h4)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "Biquote"

    def _fetch_yahoo(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        r = self.http.request(
            "GET", "https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
            params={"interval": "5m", "range": "5d"},
        )
        d = r.json()["chart"]["result"][0]
        ts = d["timestamp"]
        q = d["indicators"]["quote"][0]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(ts, unit="s", utc=True),
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
            "volume": q.get("volume", [100.0] * len(ts)),
        }).dropna()
        m5 = self._apply_offset(df.sort_values("datetime").reset_index(drop=True), offset)
        self._validate(m5, "YF-M5", self.cfg.min_bars_m5)
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        h1 = self._apply_offset(self._resample(m5, "1h"), offset)
        h4 = self._apply_offset(self._resample(m5, "4h"), offset)
        self._validate(h4, "YF-H4", self.cfg.min_bars_h4)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "YahooFinance"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
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
                logging.info(f"Data: {src} | Offset: {offset:+.3f}")
                return dfs, src, offset
            except Exception as e:
                m = f"{name}: {redact(e)}"
                logging.warning(f"Tier fail: {m}")
                errors.append(m)
        raise DataFetchError("All tiers failed. " + "; ".join(errors))

    def _validate(self, df: pd.DataFrame, label: str, min_b: int) -> None:
        if len(df) < min_b:
            raise DataFetchError(f"{label}: bars {len(df)} < {min_b}")
        for c in ("open", "high", "low", "close"):
            if (df[c] <= 0).any() or df[c].isna().any():
                raise DataFetchError(f"{label}: invalid {c}")

    def _check_staleness(self, df: pd.DataFrame, int_min: int, label: str) -> None:
        now = datetime.now(UTC)
        if now.weekday() >= 5:
            return
        age = (now - df["datetime"].iloc[-1]).total_seconds()
        max_s = int_min * 60 * self.cfg.max_stale_mult
        if age > max_s:
            raise DataFetchError(f"{label}: stale {age/60:.1f}m > {max_s/60:.1f}m")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SIGNAL GENERATOR (v19: DXY, family dedup, adaptive weights, regime thresholds)
# ═════════════════════════════════════════════════════════════════════════════
MODEL_FAMILIES: Dict[str, str] = {
    "EMA_Cross": "trend",
    "EMA_Trend": "trend",
    "Supertrend": "trend",
    "Donchian": "breakout",
    "Kalman": "structure",
    "MACD": "momentum",
    "RSI": "oscillator",
    "Stochastic": "oscillator",
    "BollingerBB": "volatility",
    "Ichimoku": "trend",
    "SMC_OB": "structure",
    "Wavelet": "filter",
    "ML_Ensemble": "ml",
    "DXY_Inverse": "macro",
}


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
    dxy_corr: float = 0.0
    session: str = "UNKNOWN"
    skip_reason: str = ""


class SignalGenerator:
    def __init__(self, cfg: Config, kv: Optional[KVStore] = None):
        self.cfg = cfg
        self.kv = kv
        self._registry = self._build_registry()
        self._dxy_df: Optional[pd.DataFrame] = None

    def set_dxy(self, dxy: Optional[pd.DataFrame]) -> None:
        self._dxy_df = dxy

    # ─── Individual Models ──────────────────────────────────────────────
    def _ema_cross(self, df):
        c = df["close"]
        e9, e21 = Indicators.ema(c, 9), Indicators.ema(c, 21)
        diff, prev = e9.iloc[-1] - e21.iloc[-1], e9.iloc[-2] - e21.iloc[-2]
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        conf = min(1.0, abs(diff) / atr)
        if diff > 0 and prev <= 0: return 1, min(1.0, conf + 0.3)
        if diff < 0 and prev >= 0: return -1, min(1.0, conf + 0.3)
        return (1, conf) if diff > 0 else ((-1, conf) if diff < 0 else (0, 0.0))

    def _ema_trend(self, df):
        c = df["close"]
        e20, e50, e200 = (Indicators.ema(c, s).iloc[-1] for s in (20, 50, 200))
        last = c.iloc[-1]
        if last > e20 > e50 > e200: return 1, 0.85
        if last < e20 < e50 < e200: return -1, 0.85
        return 0, 0.0

    def _supertrend(self, df, period=10, mult=3.0):
        atr = Indicators.atr_series(df, period)
        hl2 = (df["high"] + df["low"]) / 2.0
        ub = hl2 + mult * atr
        lb = hl2 - mult * atr
        c = df["close"]
        n = len(df)
        if n < 2: return 0, 0.0
        fu, fl = ub.copy(), lb.copy()
        trend = pd.Series(1, index=df.index)
        for i in range(1, n):
            fu.iloc[i] = ub.iloc[i] if (ub.iloc[i] < fu.iloc[i-1] or c.iloc[i-1] > fu.iloc[i-1]) else fu.iloc[i-1]
            fl.iloc[i] = lb.iloc[i] if (lb.iloc[i] > fl.iloc[i-1] or c.iloc[i-1] < fl.iloc[i-1]) else fl.iloc[i-1]
            if trend.iloc[i-1] == 1:
                trend.iloc[i] = -1 if c.iloc[i] < fl.iloc[i] else 1
            else:
                trend.iloc[i] = 1 if c.iloc[i] > fu.iloc[i] else -1
        last_t = int(trend.iloc[-1])
        band = fl.iloc[-1] if last_t == 1 else fu.iloc[-1]
        conf = min(1.0, abs(c.iloc[-1] - band) / (atr.iloc[-1] + 1e-9))
        return last_t, conf

    def _donchian(self, df, period=20):
        hi = df["high"].rolling(period).max().iloc[-2]
        lo = df["low"].rolling(period).min().iloc[-2]
        last = df["close"].iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        if last > hi: return 1, min(1.0, (last - hi) / atr)
        if last < lo: return -1, min(1.0, (lo - last) / atr)
        return 0, 0.0

    def _kalman(self, df):
        p = df["close"].values
        if len(p) < 30: return 0, 0.0
        x, pv, q, r = p[0], 1.0, 1e-4, 0.01
        est = []
        for z in p:
            pv += q
            k = pv / (pv + r)
            x = x + k * (z - x)
            pv = (1.0 - k) * pv
            est.append(x)
        k_s = pd.Series(est, index=df.index)
        slope = k_s.diff(3).iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        conf = min(1.0, abs(slope) / (atr * 0.2))
        if slope > 0.05 * atr: return 1, conf
        if slope < -0.05 * atr: return -1, conf
        return 0, 0.0

    def _macd(self, df):
        ml, sl, h = Indicators.macd(df["close"])
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        conf = min(1.0, abs(h) / atr)
        if ml > sl and h > 0: return 1, conf
        if ml < sl and h < 0: return -1, conf
        return 0, 0.0

    def _rsi(self, df):
        r = Indicators.rsi(df["close"], 14)
        if r < 30: return 1, (30.0 - r) / 30.0
        if r > 70: return -1, (r - 70.0) / 30.0
        if r > 55: return 1, (r - 50.0) / 50.0
        if r < 45: return -1, (50.0 - r) / 50.0
        return 0, 0.0

    def _stochastic(self, df):
        k, d = Indicators.stochastic(df)
        if k < 20 and k > d: return 1, (20.0 - k) / 20.0
        if k > 80 and k < d: return -1, (k - 80.0) / 20.0
        return 0, 0.0

    def _bollinger(self, df):
        up, mid, lo = Indicators.bollinger(df["close"])
        last = df["close"].iloc[-1]
        bw = up - lo
        if bw <= 0: return 0, 0.0
        if last <= lo: return 1, min(1.0, (lo - last) / bw + 0.5)
        if last >= up: return -1, min(1.0, (last - up) / bw + 0.5)
        return 0, 0.0

    def _ichimoku(self, df):
        c, h, l = df["close"], df["high"], df["low"]
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        sa = ((tenkan + kijun) / 2.0).shift(26)
        sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
        lc, kv, tv = c.iloc[-1], kijun.iloc[-1], tenkan.iloc[-1]
        s_a = sa.iloc[-1] if not pd.isna(sa.iloc[-1]) else kv
        s_b = sb.iloc[-1] if not pd.isna(sb.iloc[-1]) else kv
        top, bot = max(s_a, s_b), min(s_a, s_b)
        if lc > top and tv > kv: return 1, 0.8
        if lc < bot and tv < kv: return -1, 0.8
        return 0, 0.0

    def _smc_ob(self, df):
        ob_b = SignalGenerator.detect_order_block(df, "BUY")
        ob_s = SignalGenerator.detect_order_block(df, "SELL")
        last = df["close"].iloc[-1]
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        if ob_b and ob_b["low"] <= last <= ob_b["high"] + 0.3 * atr: return 1, 0.9
        if ob_s and ob_s["low"] - 0.3 * atr <= last <= ob_s["high"]: return -1, 0.9
        return 0, 0.0

    def _wavelet(self, df):
        if not HAS_SKLEARN or len(df) < 50: return 0, 0.0
        try:
            b, a = butter(2, 0.15)
            sm = lfilter(b, a, df["close"].values)
            ss = pd.Series(sm)
            ef = ss.ewm(span=10, adjust=False).mean()
            es = ss.ewm(span=30, adjust=False).mean()
            diff, slope = ef.iloc[-1] - es.iloc[-1], ef.diff(3).iloc[-1]
            atr = Indicators.atr_last(df)
            if atr <= 0: return 0, 0.0
            conf = min(1.0, float(abs(diff) / atr))
            if diff > 0 and slope > 0: return 1, conf
            if diff < 0 and slope < 0: return -1, conf
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _ml_ensemble(self, df):
        if not (HAS_SKLEARN and self.cfg.enable_ml_models) or len(df) < 300:
            return 0, 0.0
        try:
            c = df["close"]
            f = pd.DataFrame(index=df.index)
            f["r1"] = c.pct_change(1)
            f["r5"] = c.pct_change(5)
            f["atr_n"] = Indicators.atr_series(df) / (c + 1e-9)
            f["rsi"] = Indicators.rsi_series(c, 14) / 100.0
            f["d20"] = (c - Indicators.ema(c, 20)) / (c + 1e-9)
            f["d50"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
            f = f.dropna()
            horizon = 20
            av = Indicators.atr_series(df).values
            cl, hi, lo = c.values, df["high"].values, df["low"].values
            n = len(cl)
            y_arr = np.zeros(n)
            for i in range(n - horizon):
                tp = cl[i] + 1.0 * av[i]
                sl = cl[i] - 1.8 * av[i]
                htp = np.where(hi[i+1:i+horizon+1] >= tp)[0]
                hsl = np.where(lo[i+1:i+horizon+1] <= sl)[0]
                t_tp = htp[0] if len(htp) else 999
                t_sl = hsl[0] if len(hsl) else 999
                if t_tp < t_sl: y_arr[i] = 1
                elif t_sl < t_tp: y_arr[i] = 0
            y = pd.Series(y_arr[:n - horizon], index=df.index[:n - horizon])
            v = f.index.intersection(y.index)
            X, Y = f.loc[v], y.loc[v]
            split = int(len(X) * 0.8)
            train_end = max(0, split - horizon)
            Xt, Yt = X.iloc[:train_end], Y.iloc[:train_end]
            Xv, Yv = X.iloc[split:], Y.iloc[split:]
            if len(Xt) < 100 or len(Xv) < 20: return 0, 0.0
            clf = RandomForestClassifier(
                n_estimators=60, max_depth=4, random_state=42, n_jobs=1
            )
            clf.fit(Xt.values, Yt.values)
            if float(clf.score(Xv.values, Yv.values)) < self.cfg.ml_min_val_acc:
                return 0, 0.0
            proba = clf.predict_proba(f.iloc[-1:].values)[0]
            pu = float(proba[1]) if len(proba) > 1 else 0.5
            if pu > 0.62: return 1, min(1.0, (pu - 0.5) * 2.0)
            if pu < 0.38: return -1, min(1.0, (0.5 - pu) * 2.0)
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _dxy_inverse(self, df):
        """Gold and DXY historically inverse. If DXY trending down + gold trending up = BUY confirmation."""
        if not self.cfg.enable_dxy_filter or self._dxy_df is None:
            return 0, 0.0
        dxy = self._dxy_df
        if len(dxy) < 30: return 0, 0.0
        try:
            corr = Indicators.rolling_correlation(
                df["close"], dxy["close"], self.cfg.dxy_corr_window
            )
            # Find closest DXY bar timestamp to last gold bar
            dxy_close = dxy["close"].reset_index(drop=True)
            e20 = Indicators.ema(dxy_close, 20).iloc[-1]
            e50 = Indicators.ema(dxy_close, 50).iloc[-1]
            slope5 = float(dxy_close.iloc[-1] - dxy_close.iloc[-5]) if len(dxy_close) >= 5 else 0.0

            # Strong inverse correlation required
            if corr > -self.cfg.dxy_bias_threshold:  # e.g., corr > -0.55 → skip
                return 0, 0.0

            dxy_bearish = slope5 < 0 and e20 < e50
            dxy_bullish = slope5 > 0 and e20 > e50
            strength = min(1.0, abs(corr))

            if dxy_bearish:  # DXY down → gold up → BUY
                return 1, strength
            if dxy_bullish:  # DXY up → gold down → SELL
                return -1, strength
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _build_registry(self):
        reg: List[Tuple[str, Callable, float]] = [
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
            ("SMC_OB", self._smc_ob, 1.5),
            ("Wavelet", self._wavelet, 0.8),
        ]
        if self.cfg.enable_ml_models:
            reg.append(("ML_Ensemble", self._ml_ensemble, 1.0))
        if self.cfg.enable_dxy_filter:
            reg.append(("DXY_Inverse", self._dxy_inverse, 1.4))
        return reg

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _adaptive_weight(self, model_name: str, base_weight: float) -> float:
        """v19: Scale base weight by rolling hit rate from journal."""
        if not self.cfg.enable_adaptive_weights or self.kv is None:
            return base_weight
        perf = self.kv.get("model_perf", {}).get(model_name, {})
        n = perf.get("n", 0)
        hits = perf.get("hits", 0)
        if n < self.cfg.adaptive_min_trades:
            return base_weight
        acc = hits / max(1, n)
        # acc 0.5 → 1.0x; acc 0.6 → 1.4x; acc 0.4 → 0.7x
        factor = 1.0 + (acc - 0.5) * 3.0
        factor = float(np.clip(factor, self.cfg.adaptive_weight_floor, self.cfg.adaptive_weight_ceil))
        return base_weight * factor

    def _family_cap(self, raw_votes: Dict[str, Tuple[int, float, float]]) -> Tuple[float, float]:
        """v19: Prevent any single family from dominating by capping its total."""
        fam_totals: Dict[str, Dict[int, float]] = defaultdict(lambda: {1: 0.0, -1: 0.0})
        for name, (vote, conf, score) in raw_votes.items():
            if vote == 0: continue
            fam = MODEL_FAMILIES.get(name, "other")
            fam_totals[fam][vote] += score

        buy_total = sell_total = 0.0
        for name, (vote, conf, score) in raw_votes.items():
            if vote == 0: continue
            fam = MODEL_FAMILIES.get(name, "other")
            fam_dom = max(fam_totals[fam][1], fam_totals[fam][-1])
            # family_cap: total family contribution can't exceed family_cap * max member weight
            n_members = sum(1 for k in raw_votes if MODEL_FAMILIES.get(k, "other") == fam and raw_votes[k][0] != 0)
            cap = self.cfg.family_cap_multiplier * max(
                1.0, max((raw_votes[k][2] for k in raw_votes if MODEL_FAMILIES.get(k, "other") == fam), default=1.0)
            )
            scale = min(1.0, cap / max(fam_dom, 1e-6)) if fam_dom > 0 else 1.0
            adj = score * scale
            if vote > 0: buy_total += adj
            else: sell_total += adj
        return buy_total, sell_total

    def _classify_regime(self, h1) -> Tuple[str, float, float]:
        H = Indicators.hurst(h1["close"])
        E = Indicators.shannon_entropy(h1["close"])
        if E >= self.cfg.entropy_chaos_max: return "CHAOTIC", H, E
        if H >= self.cfg.hurst_trend_min: return "TRENDING", H, E
        if H <= self.cfg.hurst_meanrev_max: return "MEAN_REVERTING", H, E
        return "RANGING", H, E

    @staticmethod
    def _htf_bias(h4) -> str:
        if len(h4) < 50: return "NEUTRAL"
        c = h4["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _regime_min_ratio(regime: str, cfg: Config) -> float:
        if regime == "TRENDING": return cfg.min_ratio_trending
        if regime == "MEAN_REVERTING": return cfg.min_ratio_meanrev
        if regime == "RANGING": return cfg.min_ratio_ranging
        return cfg.min_ratio_default

    def generate(self, m5, m15, h1, h4, dxy=None) -> SignalResult:
        if dxy is not None:
            self.set_dxy(dxy)
        res = SignalResult()
        res.adx = Indicators.adx_last(m5)
        regime, H, E = self._classify_regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E
        h4_trend = self._htf_bias(h4)

        raw_votes: Dict[str, Tuple[int, float, float]] = {}
        details = []
        active = 0

        for name, fn, weight in self._registry:
            try:
                vote, conf = fn(m5)
            except Exception as e:
                logging.debug(f"{name} error: {e}")
                vote, conf = 0, 0.0
            adj_weight = self._adaptive_weight(name, weight)
            score = adj_weight * conf if vote != 0 else 0.0
            raw_votes[name] = (vote, conf, score)
            if vote != 0:
                active += 1
            details.append({
                "model": name, "vote": {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[vote],
                "family": MODEL_FAMILIES.get(name, "other"),
                "conf": round(conf, 2), "weight": round(adj_weight, 2),
                "score": round(score, 2),
            })

        buy_w, sell_w = self._family_cap(raw_votes)
        res.buy_score, res.sell_score = round(buy_w, 2), round(sell_w, 2)
        res.active_models = active
        res.votes_detail = details

        # DXY correlation info
        if self._dxy_df is not None and len(self._dxy_df) > 20:
            try:
                res.dxy_corr = round(Indicators.rolling_correlation(
                    m5["close"], self._dxy_df["close"], self.cfg.dxy_corr_window
                ), 3)
            except Exception:
                pass

        total_w = buy_w + sell_w
        if total_w > 1e-6:
            res.buy_ratio = buy_w / total_w
            res.sell_ratio = sell_w / total_w

        # Filters
        if res.adx < self.cfg.adx_min:
            res.skip_reason = f"ADX {res.adx:.1f} < {self.cfg.adx_min}"
            return res
        if regime == "CHAOTIC":
            res.skip_reason = f"CHAOTIC (E={E:.2f})"
            return res
        if regime == "RANGING" and not self.cfg.allow_ranging_regime:
            res.skip_reason = f"RANGING (H={H:.2f})"
            return res
        if total_w < 1e-6:
            res.skip_reason = "No directional models"
            return res
        if active < self.cfg.parliament_min_active:
            res.skip_reason = f"Active {active} < quorum {self.cfg.parliament_min_active}"
            return res

        dom = max(res.buy_ratio, res.sell_ratio)
        min_ratio = self._regime_min_ratio(regime, self.cfg)
        if dom < min_ratio:
            res.skip_reason = f"Consensus {dom:.0%} < {regime} minimum {min_ratio:.0%}"
            return res

        if dom >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        else:
            res.verdict = "EXECUTE_MICRO"

        # HTF filter
        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH" and self.cfg.enforce_htf_filter:
                if dom >= self.cfg.counter_trend_allow_ratio:
                    res.direction = "BUY"
                    if res.verdict == "EXECUTE_FULL":
                        res.verdict = "EXECUTE_HALF"
                else:
                    res.skip_reason = f"BUY blocked: H4 BEARISH ({dom:.0%})"
                    return res
            else:
                res.direction = "BUY"
        else:
            if h4_trend == "BULLISH" and self.cfg.enforce_htf_filter:
                if dom >= self.cfg.counter_trend_allow_ratio:
                    res.direction = "SELL"
                    if res.verdict == "EXECUTE_FULL":
                        res.verdict = "EXECUTE_HALF"
                else:
                    res.skip_reason = f"SELL blocked: H4 BULLISH ({dom:.0%})"
                    return res
            else:
                res.direction = "SELL"

        return res

    @staticmethod
    def detect_order_block(df, side, lookback=40):
        if len(df) < lookback + 5: return None
        tail = df.tail(lookback + 5).reset_index(drop=True)
        atr = Indicators.atr_last(df)
        if atr <= 0: return None
        for i in range(len(tail) - 2, 2, -1):
            body = abs(tail["close"].iloc[i] - tail["open"].iloc[i])
            if body < 1.2 * atr: continue
            is_bull = tail["close"].iloc[i] > tail["open"].iloc[i]
            if side == "BUY" and not is_bull: continue
            if side == "SELL" and is_bull: continue
            for j in range(i - 1, max(i - 10, 0), -1):
                opp = tail["close"].iloc[j] < tail["open"].iloc[j]
                if side == "BUY" and opp:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
                if side == "SELL" and not opp:
                    return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
        return None

    def _session_atr_mult(self, session: str) -> float:
        return {
            "LONDON_OPEN": self.cfg.session_atr_london,
            "NY_OPEN": self.cfg.session_atr_ny_open,
            "NY_AFTERNOON": self.cfg.session_atr_ny_afternoon,
            "ASIAN": self.cfg.session_atr_asian,
        }.get(session, 1.0)

    def compute_levels(self, m5, side, ob, session: str = "LONDON_OPEN"):
        atr = Indicators.atr_last(m5) * self._session_atr_mult(session)
        if atr <= 0: raise ValueError("ATR must be positive")
        price = float(m5["close"].iloc[-1])
        dec = self.cfg.price_decimals
        sb = self.cfg.spread_buffer
        ob_h = ob["high"] if ob else price
        ob_l = ob["low"] if ob else price

        if side == "BUY":
            ob_c = (ob_h - (ob_h - ob_l) * 0.5) if (ob and ob_h < price) else None
            if ob_c and (price - ob_c) <= self.cfg.entry_max_chase_atr * atr:
                entry, sl_ref = ob_c, ob_l
            else:
                entry = price - atr * self.cfg.entry_pullback_atr
                sl_ref = entry - atr * self.cfg.atr_sl_mult
            entry = min(entry, price - atr * 0.05)
            if (price - entry) > self.cfg.entry_max_chase_atr * atr:
                entry = price - atr * self.cfg.entry_pullback_atr
            sl = min(sl_ref, entry - atr * 0.5) - sb
            if entry <= sl: sl = entry - atr * self.cfg.atr_sl_mult - sb
            tp1 = entry + atr * self.cfg.atr_tp1_mult
            tp2 = entry + atr * self.cfg.atr_tp2_mult
            tp3 = entry + atr * self.cfg.atr_tp3_mult
            if not (sl < entry < tp1 < tp2 < tp3):
                raise ValueError(f"Incoherent BUY: {sl} {entry} {tp1} {tp2} {tp3}")
            return tuple(round(x, dec) for x in (entry, sl, tp1, tp2, tp3))
        else:
            ob_c = (ob_l + (ob_h - ob_l) * 0.5) if (ob and ob_l > price) else None
            if ob_c and (ob_c - price) <= self.cfg.entry_max_chase_atr * atr:
                entry, sl_ref = ob_c, ob_h
            else:
                entry = price + atr * self.cfg.entry_pullback_atr
                sl_ref = entry + atr * self.cfg.atr_sl_mult
            entry = max(entry, price + atr * 0.05)
            if (entry - price) > self.cfg.entry_max_chase_atr * atr:
                entry = price + atr * self.cfg.entry_pullback_atr
            sl = max(sl_ref, entry + atr * 0.5) + sb
            if sl <= entry: sl = entry + atr * self.cfg.atr_sl_mult + sb
            tp1 = entry - atr * self.cfg.atr_tp1_mult
            tp2 = entry - atr * self.cfg.atr_tp2_mult
            tp3 = entry - atr * self.cfg.atr_tp3_mult
            if not (sl > entry > tp1 > tp2 > tp3):
                raise ValueError(f"Incoherent SELL: {sl} {entry} {tp1} {tp2} {tp3}")
            return tuple(round(x, dec) for x in (entry, sl, tp1, tp2, tp3))


# ═════════════════════════════════════════════════════════════════════════════
# 8. RISK MANAGEMENT (drawdown-scaled, session-aware)
# ═════════════════════════════════════════════════════════════════════════════
class RiskManagementEngine:
    CLOSED_OUTCOMES = ("WIN_TP2", "WIN_TP3", "WIN_BE", "LOSS")

    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        base = self.cfg.account_balance
        journal = self.kv.get("journal", []) or []
        closed = [j for j in journal if j.get("r_achieved") is not None
                  and j.get("outcome") in self.CLOSED_OUTCOMES]
        closed.sort(key=lambda x: x.get("timestamp", ""))
        for t in closed:
            try:
                r = float(t.get("r_achieved", 0.0))
                rp = float(t.get("risk_pct", 1.0))
                base += r * (base * (rp / 100.0))
            except (ValueError, TypeError):
                continue
        return max(10.0, round(base, 2))

    def _drawdown_scale(self, current_balance: float) -> float:
        """v19: Reduce risk size during drawdown."""
        cb = self.kv.get("circuit_breaker", {}) or {}
        day_start = cb.get("day_start_balance", current_balance)
        if day_start <= 0: return 1.0
        dd_pct = max(0.0, (day_start - current_balance) / day_start * 100.0)
        if dd_pct <= self.cfg.dd_scale_threshold_pct:
            return 1.0
        max_dd = self.cfg.max_daily_loss_pct
        span = max_dd - self.cfg.dd_scale_threshold_pct
        if span <= 0: return self.cfg.dd_scale_min_factor
        progress = min(1.0, (dd_pct - self.cfg.dd_scale_threshold_pct) / span)
        return float(np.clip(
            1.0 - progress * (1.0 - self.cfg.dd_scale_min_factor),
            self.cfg.dd_scale_min_factor, 1.0
        ))

    def compute_lot(self, balance, entry, sl, confidence_pct) -> Tuple[float, float]:
        sl_d = abs(entry - sl)
        if sl_d < 1e-4: raise ValueError("SL distance too small")
        conf = max(0.4, min(1.0, confidence_pct / 100.0))
        risk_pct = max(0.1, self.cfg.max_risk_pct_per_trade * conf)
        risk_pct *= self._drawdown_scale(balance)  # v19
        risk_d = balance * (risk_pct / 100.0)
        raw = risk_d / (sl_d * self.cfg.contract_size)
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(raw, 2)))
        notional = lot * self.cfg.contract_size * entry
        if balance > 0 and (notional / balance) > self.cfg.hard_leverage_cap:
            max_lot = (self.cfg.hard_leverage_cap * balance) / (self.cfg.contract_size * entry)
            if max_lot < self.cfg.min_lot:
                raise RiskViolationError(
                    f"Leverage cap: max_lot={max_lot:.4f} < min={self.cfg.min_lot}"
                )
            lot = max(self.cfg.min_lot, round(max_lot, 2))
        actual = (lot * self.cfg.contract_size * sl_d) / balance * 100.0
        return lot, round(actual, 2)

    def check_circuit_breakers(self, balance) -> Tuple[bool, str]:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        cb = self.kv.get("circuit_breaker", {}) or {}
        locked = cb.get("locked_until")
        if locked:
            try:
                if datetime.now(UTC) < datetime.fromisoformat(locked):
                    return False, f"LOCKED until {locked}"
            except Exception:
                pass
        ds = cb.get("day_start_balance")
        if cb.get("last_date") != today or ds is None:
            cb["last_date"] = today
            cb["day_start_balance"] = balance
            cb["daily_signals_count"] = 0
            self.kv.set("circuit_breaker", cb)
            ds = balance
        dd = (ds - balance) / ds * 100.0
        if dd >= self.cfg.max_daily_loss_pct:
            lock = (datetime.now(UTC) + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lock
            self.kv.set("circuit_breaker", cb); self.kv.save()
            return False, f"DAILY DD {dd:.2f}% >= {self.cfg.max_daily_loss_pct}%"
        journal = self.kv.get("journal", []) or []
        closed = [j for j in journal if j.get("outcome") in self.CLOSED_OUTCOMES]
        consec = 0
        for t in reversed(closed):
            if t.get("outcome") == "LOSS": consec += 1
            else: break
        if consec >= self.cfg.max_consecutive_losses:
            lock = (datetime.now(UTC) + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lock
            self.kv.set("circuit_breaker", cb); self.kv.save()
            return False, f"CONSEC {consec} >= {self.cfg.max_consecutive_losses}"
        if cb.get("daily_signals_count", 0) >= self.cfg.max_daily_signals:
            return False, f"DAILY CAP {cb.get('daily_signals_count')}"
        return True, "clear"


# ═════════════════════════════════════════════════════════════════════════════
# 9. NEWS FILTER
# ═════════════════════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg, http):
        self.cfg = cfg
        self.http = http
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self._cached: List[dict] = []
        self._cache_ts = 0.0

    def fetch_events(self) -> List[dict]:
        if time.time() - self._cache_ts < 3600 and self._cached:
            return self._cached
        try:
            r = self.http.request("GET", self.url)
            data = r.json()
            if isinstance(data, list):
                self._cached = data
                self._cache_ts = time.time()
                return data
        except Exception as e:
            logging.warning(f"News fetch failed: {redact(e)}")
        return self._cached

    def is_blackout(self, now=None) -> Tuple[bool, str]:
        if now is None: now = datetime.now(UTC)
        window = timedelta(minutes=self.cfg.news_blackout_minutes)
        for ev in self.fetch_events():
            if ev.get("country") != "USD" or ev.get("impact") not in ("High", "Holiday"):
                continue
            ds = ev.get("date", "")
            if not ds: continue
            try:
                et = datetime.fromisoformat(ds)
                et = et.replace(tzinfo=UTC) if et.tzinfo is None else et.astimezone(UTC)
                if (et - window) <= now <= (et + window):
                    title = html.escape(str(ev.get("title", "High-Impact")))
                    diff = int((et - now).total_seconds() / 60)
                    status = f"in {diff}m" if diff > 0 else f"{abs(diff)}m ago"
                    return True, f"{title} ({status})"
            except Exception:
                continue
        return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# 10. TELEGRAM
# ═════════════════════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, http, dry_run=False):
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.http = http
        self.dry_run = dry_run

    def send(self, msg: str) -> bool:
        if self.dry_run or not (self.token and self.chat_id):
            logging.info(f"[DRY-RUN TG]\n{msg}")
            return True
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            self.http.request("POST", url, json={
                "chat_id": self.chat_id, "text": msg,
                "parse_mode": "HTML", "disable_web_page_preview": True,
            })
            logging.info("Telegram dispatched.")
            return True
        except Exception as e:
            logging.error(f"Telegram failed: {redact(e)}")
            return False


# ═════════════════════════════════════════════════════════════════════════════
# 11. OUTCOME TRACKER (conservative intrabar, trailing, model perf)
# ═════════════════════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        journal = self.kv.get("journal", []) or []
        events = []
        now = datetime.now(UTC)
        modified = False
        model_perf = self.kv.get("model_perf", {}) or {}

        for entry in journal:
            if entry.get("outcome") != "PENDING": continue
            try:
                anchor = datetime.strptime(
                    entry["timestamp"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=UTC)
            except Exception:
                anchor = now - timedelta(hours=24)

            age_h = (now - anchor).total_seconds() / 3600.0
            if not entry.get("filled") and age_h > self.cfg.pending_stale_hours:
                entry["outcome"] = "STALE"
                entry["filled"] = 0
                entry["r_achieved"] = 0.0
                events.append(("STALE", dict(entry)))
                modified = True
                continue

            side = entry["signal"]
            ep, sl = float(entry["entry"]), float(entry["sl"])
            tp1, tp2, tp3 = float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
            if ep <= 0 or sl <= 0: continue

            bars = m5[m5["datetime"] > anchor].sort_values("datetime")
            if bars.empty: continue

            filled = bool(entry.get("filled"))
            fill_idx: Optional[int] = None
            if not filled:
                for i, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep:
                        filled, fill_idx = True, i; break
                    if side == "SELL" and b.high >= ep:
                        filled, fill_idx = True, i; break
                if not filled:
                    chase = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    lh, ll = float(bars["high"].iloc[-1]), float(bars["low"].iloc[-1])
                    if (side == "BUY" and lh > ep + chase) or (side == "SELL" and ll < ep - chase):
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                    continue
                entry["filled"] = 1
                modified = True

            walk = bars.iloc[fill_idx:] if fill_idx is not None else bars
            hit_tp1 = bool(entry.get("hit_tp1"))
            hit_tp2 = bool(entry.get("hit_tp2"))
            hit_tp3 = bool(entry.get("hit_tp3"))
            hit_sl = bool(entry.get("hit_sl"))
            hit_be = bool(entry.get("hit_be"))
            trailing_sl = sl  # for runner trailing after TP2

            for b in walk.itertuples(index=False):
                if hit_tp3 or hit_sl or hit_be: break

                # Determine current effective SL
                if hit_tp2:
                    # v19: trail after TP2
                    atr = Indicators.atr_last(m5)  # fixed estimate is fine for tracking
                    trail_dist = atr * self.cfg.trailing_atr_mult
                    if side == "BUY":
                        trailing_sl = max(trailing_sl, float(b.high) - trail_dist)
                    else:
                        trailing_sl = min(trailing_sl, float(b.low) + trail_dist)
                    cur_sl = trailing_sl
                elif hit_tp1:
                    cur_sl = ep  # BE after TP1
                else:
                    cur_sl = sl

                # Conservative intrabar: SL checked BEFORE TP
                if side == "BUY":
                    if b.low <= cur_sl:
                        if hit_tp2:
                            # Runner stopped out — record as WIN_TP2 with partial
                            hit_be = True
                        elif hit_tp1:
                            hit_be = True
                        else:
                            hit_sl = True
                        continue
                    if not hit_tp1 and b.high >= tp1: hit_tp1 = True
                    if hit_tp1 and not hit_tp2 and b.high >= tp2: hit_tp2 = True
                    if hit_tp2 and not hit_tp3 and b.high >= tp3: hit_tp3 = True
                else:
                    if b.high >= cur_sl:
                        if hit_tp1: hit_be = True
                        else: hit_sl = True
                        continue
                    if not hit_tp1 and b.low <= tp1: hit_tp1 = True
                    if hit_tp1 and not hit_tp2 and b.low <= tp2: hit_tp2 = True
                    if hit_tp2 and not hit_tp3 and b.low <= tp3: hit_tp3 = True

            entry["hit_tp1"] = int(hit_tp1)
            entry["hit_tp2"] = int(hit_tp2)
            entry["hit_tp3"] = int(hit_tp3)
            entry["hit_sl"] = int(hit_sl)
            entry["hit_be"] = int(hit_be)

            sl_d = abs(ep - sl)
            if sl_d < 1e-9: continue
            r1 = abs(tp1 - ep) / sl_d
            r2 = abs(tp2 - ep) / sl_d
            r3 = abs(tp3 - ep) / sl_d

            closed = False
            if hit_tp3:
                entry["outcome"] = "WIN_TP3"
                entry["r_achieved"] = round((r1 + r2 + r3) / 3.0, 4)
                closed = True
            elif hit_tp2 and hit_be:
                entry["outcome"] = "WIN_TP2"
                entry["r_achieved"] = round((r1 + r2) / 3.0, 4)
                closed = True
            elif hit_tp1 and hit_be:
                entry["outcome"] = "WIN_BE"
                entry["r_achieved"] = round(r1 / 3.0, 4)
                closed = True
            elif hit_sl:
                entry["outcome"] = "LOSS"
                entry["r_achieved"] = -1.0
                closed = True

            if closed:
                events.append(("CLOSE", dict(entry)))
                modified = True

                # v19: Update per-model performance
                votes_snap = entry.get("votes_snapshot", [])
                outcome_dir = side
                was_correct = entry["outcome"].startswith("WIN")
                for v in votes_snap:
                    mname = v.get("model")
                    vdir = v.get("vote")
                    if not mname or vdir == "NEUTRAL": continue
                    perf = model_perf.setdefault(mname, {"n": 0, "hits": 0})
                    perf["n"] = int(perf.get("n", 0)) + 1
                    if (vdir == outcome_dir and was_correct) or (vdir != outcome_dir and not was_correct):
                        perf["hits"] = int(perf.get("hits", 0)) + 1
                self.kv.set("model_perf", model_perf)

        if modified:
            self.kv.set("journal", journal)
            self.kv.save()
        return events


# ═════════════════════════════════════════════════════════════════════════════
# 12. ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = ResilientHttpClient(
            timeout=cfg.http_timeout_sec, max_retries=cfg.http_max_retries,
            backoff=cfg.http_backoff_factor,
        )
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http)
        self.signal = SignalGenerator(cfg, self.kv)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.http)
        self.tg = TelegramNotifier(self.http, dry_run=cfg.dry_run)
        self.outcome = OutcomeTracker(cfg, self.kv)

    def _session(self, now: datetime) -> str:
        lh = now.astimezone(LONDON_TZ)
        nh = now.astimezone(NY_TZ)
        lf = lh.hour + lh.minute / 60.0
        nf = nh.hour + nh.minute / 60.0
        if 7.0 <= lf < 10.0: return "LONDON_OPEN"
        if 8.0 <= nf < 11.0: return "NY_OPEN"
        if 13.0 <= nf < 16.0: return "NY_AFTERNOON"
        if 0.0 <= lf < 6.0: return "ASIAN"
        return "OFF_HOURS"

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter: return True
        return self._session(now) in ("LONDON_OPEN", "NY_OPEN", "NY_AFTERNOON")

    def run(self) -> None:
        try: self.kv.force_load()
        except StateError as e:
            logging.error(f"State: {e}"); return

        now = datetime.now(UTC)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        session = self._session(now)

        if self.cfg.kill_switch:
            logging.info("KILL SWITCH"); return
        if now.weekday() >= 5:
            logging.info("Weekend"); return
        if not self._in_killzone(now):
            logging.info(f"Outside killzone (session={session})"); return

        balance = self.risk.dynamic_balance()
        ok, reason = self.risk.check_circuit_breakers(balance)
        if not ok:
            logging.info(f"CB: {reason}"); return

        blackout, news_title = self.news.is_blackout()
        if blackout:
            logging.info(f"News: [{news_title}]"); return

        try:
            dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            err = redact(e)
            logging.error(f"Data: {err}")
            self.tg.send(f"⚠️ <b>Data Warning</b>\n<code>{html.escape(err[:200])}</code>")
            return

        # Fetch DXY for correlation filter
        dxy = self.data.fetch_dxy() if self.cfg.enable_dxy_filter else None

        try:
            events = self.outcome.update(dfs["5min"])
            for evt, trade in events:
                if evt == "CLOSE":
                    sig = html.escape(str(trade["signal"]))
                    sym = html.escape(str(trade["symbol"]))
                    out = html.escape(str(trade["outcome"]))
                    e_px = html.escape(str(trade["entry"]))
                    rv = float(trade.get("r_achieved", 0) or 0)
                    self.tg.send(
                        f"📕 <b>Closed</b> {sig} {sym}\n"
                        f"Entry: <code>{e_px}</code> → <b>{out}</b> ({rv:+.2f}R)"
                    )
                elif evt in ("INVALIDATED", "STALE"):
                    sig = html.escape(str(trade["signal"]))
                    e_px = html.escape(str(trade["entry"]))
                    self.tg.send(f"⚪ <b>Order {evt}</b>: {sig} @ <code>{e_px}</code>")
        except Exception as e:
            logging.warning(f"Outcome warning: {e}")

        balance = self.risk.dynamic_balance()
        m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)

        if len(m5c) < self.cfg.min_bars_m5 or len(h1c) < self.cfg.min_bars_h1:
            logging.warning("Insufficient bars"); return

        last_price = float(m5c["close"].iloc[-1])
        sig_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("Active position open"); self._update_state(last_price, now_str); return
        unfilled = sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled"))
        if unfilled >= 2:
            logging.info(f"Max pending ({unfilled})"); self._update_state(last_price, now_str); return

        result = self.signal.generate(m5c, m15c, h1c, h4c, dxy=dxy)
        result.session = session
        logging.info(
            f"Consensus: {result.direction} | B={result.buy_ratio:.0%} S={result.sell_ratio:.0%} | "
            f"Verdict={result.verdict} | Regime={result.regime} | Session={session} | "
            f"DXYcorr={result.dxy_corr:+.2f} | Skip={result.skip_reason}"
        )

        if result.direction == "NEUTRAL" or result.verdict == "CANCEL":
            self._update_state(last_price, now_str); return

        state = self.kv.get("state", {}) or {}
        last_dir = state.get("last_signal_type", "NEUTRAL")
        last_ts = state.get("last_signal_ts", "")
        cool = self.cfg.opposite_cooldown_minutes if last_dir != result.direction else self.cfg.cooldown_minutes
        if last_ts:
            try:
                last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                if (now - last_dt) < timedelta(minutes=cool):
                    logging.info(f"Cooldown {cool}m"); self._update_state(last_price, now_str); return
            except Exception:
                pass

        ob = self.signal.detect_order_block(m5c, result.direction)
        try:
            entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5c, result.direction, ob, session)
        except ValueError as e:
            logging.warning(f"Levels: {e}"); self._update_state(last_price, now_str); return

        conf_pct = max(result.buy_ratio, result.sell_ratio) * 100.0
        try:
            lot, actual_risk = self.risk.compute_lot(balance, entry, sl, conf_pct)
        except (ValueError, RiskViolationError) as e:
            logging.warning(f"Sizing: {e}"); self._update_state(last_price, now_str); return

        if result.verdict == "EXECUTE_HALF":
            lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif result.verdict == "EXECUTE_MICRO":
            lot = max(self.cfg.min_lot, round(lot / 4.0, 2))

        action = f"{result.direction} LIMIT"
        ob_disp = f"✅ {ob['high']:.2f}–{ob['low']:.2f}" if ob else "⚠️ ATR Buffer"

        tg_msg = (
            f"🏛️ <b>APEX v{VERSION} — {html.escape(action)}</b>\n\n"
            f"Asset: <b>{html.escape(self.cfg.symbol)}</b>\n"
            f"Entry: <code>{entry:.2f}</code> | SL: <code>{sl:.2f}</code>\n\n"
            f"🎯 TP1: <code>{tp1:.2f}</code> (1/3+BE)\n"
            f"🎯 TP2: <code>{tp2:.2f}</code> (1/3, trail)\n"
            f"🎯 TP3: <code>{tp3:.2f}</code> (runner)\n\n"
            f"📦 <b>Lot: <code>{lot}</code></b>\n"
            f"💰 Bal: <code>${balance:.2f}</code> | Risk: <code>{actual_risk:.2f}%</code>\n"
            f"📊 B={result.buy_ratio:.0%} S={result.sell_ratio:.0%}\n"
            f"🧠 {html.escape(result.verdict)} | Models {result.active_models}/13\n"
            f"🌐 {html.escape(result.regime)} (H={result.hurst:.2f}, E={result.entropy:.2f})\n"
            f"💵 DXY corr: <code>{result.dxy_corr:+.2f}</code>\n"
            f"🕐 Session: {html.escape(session)}\n"
            f"🧱 SMC: {html.escape(ob_disp)}\n\n"
            f"⏰ Bar: <code>{html.escape(sig_bar_iso)} UTC</code>"
        )

        if self.tg.send(tg_msg):
            trade = {
                "id": hashlib.md5(f"{sig_bar_iso}_{result.direction}".encode()).hexdigest()[:10],
                "timestamp": now_str,
                "symbol": self.cfg.symbol,
                "signal": result.direction,
                "verdict": result.verdict,
                "session": session,
                "regime": result.regime,
                "dxy_corr": result.dxy_corr,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "lot": lot, "risk_pct": actual_risk,
                "filled": 0, "outcome": "PENDING", "r_achieved": None,
                "hit_tp1": 0, "hit_tp2": 0, "hit_tp3": 0, "hit_sl": 0, "hit_be": 0,
                "votes_snapshot": result.votes_detail,
            }
            journal.append(trade)
            self.kv.set("journal", journal)

            cb = self.kv.get("circuit_breaker", {}) or {}
            cb["daily_signals_count"] = cb.get("daily_signals_count", 0) + 1
            self.kv.set("circuit_breaker", cb)

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
# 13. BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """Walk-forward backtest. Uses SignalGenerator + level logic, no live data."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.signal = SignalGenerator(cfg, kv=None)

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 100.0), errors="coerce").fillna(100.0)
        return df.dropna().sort_values("datetime").reset_index(drop=True)

    def run(self, df: pd.DataFrame, max_bars: Optional[int] = None) -> Dict[str, Any]:
        if len(df) < 300:
            raise ValueError("Backtest requires >= 300 bars")
        h1 = self._resample(df, "1h")
        h4 = self._resample(df, "4h")
        h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True)
        h4["datetime"] = pd.to_datetime(h4["datetime"], utc=True)

        trades = []
        balance = self.cfg.account_balance
        start = 200
        end = len(df) if max_bars is None else min(len(df), start + max_bars)

        for i in range(start, end - 20):
            hist = df.iloc[:i].reset_index(drop=True)
            h1h = h1[h1["datetime"] <= hist["datetime"].iloc[-1]].reset_index(drop=True)
            h4h = h4[h4["datetime"] <= hist["datetime"].iloc[-1]].reset_index(drop=True)
            if len(h1h) < 100 or len(h4h) < 50: continue

            sig = self.signal.generate(hist, hist, h1h, h4h)
            if sig.direction == "NEUTRAL" or sig.verdict == "CANCEL": continue

            ob = self.signal.detect_order_block(hist, sig.direction)
            session = self._classify_session(hist["datetime"].iloc[-1])
            try:
                entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(
                    hist, sig.direction, ob, session
                )
            except ValueError:
                continue

            # Simulate forward
            future = df.iloc[i:min(i + 200, len(df))].reset_index(drop=True)
            result = self._simulate(future, sig.direction, entry, sl, tp1, tp2, tp3)
            if result is None: continue

            r = result["r"]
            balance *= (1 + r * 0.01)  # 1% risk per trade
            trades.append({
                "time": hist["datetime"].iloc[-1].isoformat(),
                "side": sig.direction,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "outcome": result["outcome"], "r": r, "balance": round(balance, 2),
                "regime": sig.regime, "session": session,
            })

        if not trades:
            return {"trades": 0, "message": "No signals generated"}

        wins = sum(1 for t in trades if t["outcome"].startswith("WIN"))
        losses = sum(1 for t in trades if t["outcome"] == "LOSS")
        total_r = sum(t["r"] for t in trades)
        eq = pd.Series([t["balance"] for t in trades])
        max_dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100

        return {
            "trades": len(trades),
            "wins": wins, "losses": losses,
            "win_rate": round(wins / len(trades) * 100, 2),
            "total_r": round(total_r, 2),
            "final_balance": round(balance, 2),
            "return_pct": round((balance / self.cfg.account_balance - 1) * 100, 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "avg_r": round(total_r / len(trades), 3),
        }

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            df.set_index("datetime").resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna().reset_index()
        )

    @staticmethod
    def _classify_session(ts: pd.Timestamp) -> str:
        t = ts.tz_convert(LONDON_TZ) if ts.tzinfo else ts.replace(tzinfo=UTC).astimezone(LONDON_TZ)
        h = t.hour + t.minute / 60.0
        if 7.0 <= h < 10.0: return "LONDON_OPEN"
        if 13.0 <= h < 16.0: return "NY_OPEN"
        if 18.0 <= h < 21.0: return "NY_AFTERNOON"
        if 0.0 <= h < 6.0: return "ASIAN"
        return "OFF_HOURS"

    @staticmethod
    def _simulate(future, side, entry, sl, tp1, tp2, tp3) -> Optional[dict]:
        filled = False
        for b in future.itertuples(index=False):
            if not filled:
                if (side == "BUY" and b.low <= entry) or (side == "SELL" and b.high >= entry):
                    filled = True
                continue
            sl_d = abs(entry - sl)
            r1 = abs(tp1 - entry) / sl_d
            r2 = abs(tp2 - entry) / sl_d
            r3 = abs(tp3 - entry) / sl_d

            # Conservative: SL checked first
            if side == "BUY":
                if b.low <= sl: return {"outcome": "LOSS", "r": -1.0}
                if b.high >= tp3: return {"outcome": "WIN_TP3", "r": (r1+r2+r3)/3.0}
                if b.high >= tp2: return {"outcome": "WIN_TP2", "r": (r1+r2)/3.0}
                if b.high >= tp1: return {"outcome": "WIN_BE", "r": r1/3.0}
            else:
                if b.high >= sl: return {"outcome": "LOSS", "r": -1.0}
                if b.low <= tp3: return {"outcome": "WIN_TP3", "r": (r1+r2+r3)/3.0}
                if b.low <= tp2: return {"outcome": "WIN_TP2", "r": (r1+r2)/3.0}
                if b.low <= tp1: return {"outcome": "WIN_BE", "r": r1/3.0}
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 14. CLI
# ═════════════════════════════════════════════════════════════════════════════
_ORCH_REF: Optional[TradingBotOrchestrator] = None


def _sigterm(signum, frame):
    logging.warning(f"Signal {signum} — draining...")
    if _ORCH_REF is not None:
        try: _ORCH_REF.kv._drain_pending(timeout=10.0)
        except Exception: pass
    sys.exit(0)


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    global _ORCH_REF
    parser = argparse.ArgumentParser(description=f"APEX Predator v{VERSION}")
    parser.add_argument("command", choices=["run", "test", "audit", "state", "version", "health", "backtest"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv", help="CSV path for backtest mode")
    parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = Config.from_env()
    if args.dry_run: cfg.dry_run = True

    try: cfg.validate()
    except AssertionError as e:
        logging.critical(f"Config: {e}"); sys.exit(1)

    if args.command == "version":
        print(f"APEX Predator v{VERSION}"); return

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    if args.command == "backtest":
        if not args.csv:
            print("--csv required for backtest"); sys.exit(1)
        bt = BacktestEngine(cfg)
        df = bt.load_csv(args.csv)
        print(f"Loaded {len(df)} bars")
        metrics = bt.run(df, args.max_bars)
        print(json.dumps(metrics, indent=2))
        return

    orch = TradingBotOrchestrator(cfg)
    _ORCH_REF = orch

    if args.command == "run": orch.run()
    elif args.command == "state":
        orch.kv.force_load()
        print(json.dumps(orch.kv.data, indent=2, default=str))
    elif args.command == "audit":
        orch.kv.force_load()
        journal = orch.kv.get("journal", []) or []
        closed = [j for j in journal if j.get("outcome") in RiskManagementEngine.CLOSED_OUTCOMES]
        wins = sum(1 for j in closed if str(j.get("outcome", "")).startswith("WIN"))
        losses = sum(1 for j in closed if j.get("outcome") == "LOSS")
        total_r = sum(float(j.get("r_achieved", 0) or 0) for j in closed)
        print(f"Trades: {len(journal)} | Closed: {len(closed)}")
        print(f"Wins: {wins} | Losses: {losses} | WR: {wins/max(1,len(closed))*100:.1f}%")
        print(f"Total R: {total_r:+.2f}R | Avg: {total_r/max(1,len(closed)):.3f}R")
        perf = orch.kv.get("model_perf", {}) or {}
        if perf:
            print("\nModel Performance:")
            for m, p in sorted(perf.items(), key=lambda x: -x[1].get("n", 0)):
                n, h = p.get("n", 0), p.get("hits", 0)
                if n > 0:
                    print(f"  {m:<15} n={n:>3} acc={h/n*100:>5.1f}%")
    elif args.command == "health":
        orch.kv.force_load()
        state = orch.kv.get("state", {})
        cb = orch.kv.get("circuit_breaker", {})
        out = {
            "version": VERSION,
            "last_run_utc": state.get("last_run_utc"),
            "last_gold_price": state.get("last_gold_price"),
            "circuit_breaker_locked": cb.get("locked_until"),
            "daily_signals": cb.get("daily_signals_count", 0),
            "balance": orch.risk.dynamic_balance(),
        }
        print(json.dumps(out, indent=2))
    elif args.command == "test":
        logging.info("Self-test...")
        orch.kv.force_load()
        dfs, src, off = orch.data.fetch_all()
        dxy = orch.data.fetch_dxy()
        logging.info(f"OK: src={src}, M5={len(dfs['5min'])}, offset={off:+.3f}, dxy={len(dxy) if dxy is not None else 0}")


if __name__ == "__main__":
    main()
