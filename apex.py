#!/usr/bin/env python3
"""
APEX PREDATOR v19.4 — Institutional XAUUSD Quant Engine
Single-file, production-ready, mathematically optimized.
"""

from __future__ import annotations

import subprocess
import sys

def _ensure_deps():
    req = ["numpy>=1.24,<3.0", "pandas>=2.0,<3.0", "requests>=2.31,<3.0"]
    opt = ["scikit-learn>=1.3,<2.0", "scipy>=1.11,<2.0"]
    missing = []
    for mod, pkg in [("numpy", req[0]), ("pandas", req[1]), ("requests", req[2])]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    missing_opt = []
    for mod, pkg in [("sklearn", opt[0]), ("scipy", opt[1])]:
        try:
            __import__(mod)
        except ImportError:
            missing_opt.append(pkg)
    if missing_opt:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_opt])
        except subprocess.CalledProcessError:
            print("⚠️  Optional deps failed — reduced model set", flush=True)

_ensure_deps()

import argparse, atexit, hashlib, html, json, logging, math, os, random, re, signal, threading, time
from collections import defaultdict
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

VERSION = "19.4-PRODUCTION"
NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
UTC = timezone.utc

# ═══════════════════ CONFIG (MATHEMATICALLY OPTIMIZED) ═══════════════════
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v18.json"
    local_state_file: str = "apex_state_v18.json"
    
    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    price_decimals: int = 2
    mt5_offset: float = 0.0

    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01

    dd_scale_threshold_pct: float = 1.5
    dd_scale_min_factor: float = 0.4
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24
    max_daily_signals: int = 5
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120

    # --- PERBAIKAN MATEMATIKA RR ---
    atr_period: int = 14
    atr_sl_mult: float = 1.5     # SL 1.5 ATR
    atr_tp1_mult: float = 1.5    # TP1 1.5 ATR (1:1 RR)
    atr_tp2_mult: float = 3.0    # TP2 3.0 ATR (1:2 RR)
    atr_tp3_mult: float = 5.0    # TP3 5.0 ATR (1:3.33 RR)
    
    spread_buffer: float = 0.35
    trailing_atr_mult: float = 1.5

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

    min_ratio_trending: float = 0.65
    min_ratio_meanrev: float = 0.55
    min_ratio_ranging: float = 0.60
    min_ratio_default: float = 0.55

    parliament_min_active: int = 6
    parliament_full_ratio: float = 0.70
    parliament_half_ratio: float = 0.60
    family_cap_multiplier: float = 1.5

    enable_dxy_filter: bool = True
    dxy_corr_window: int = 20
    dxy_bias_threshold: float = 0.55

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
    def from_env(cls):
        def es(k, d): return os.getenv(k, d).strip()
        def ef(k, d):
            v = os.getenv(k, "").strip()
            try: return float(v) if v else d
            except ValueError: return d
        def ei(k, d):
            v = os.getenv(k, "").strip()
            try: return int(v) if v else d
            except ValueError: return d
        def eb(k, d):
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
        c.parliament_min_active = ei("APEX_MIN_ACTIVE", c.parliament_min_active)
        c.adx_min = ef("APEX_ADX_MIN", c.adx_min)
        return c

    def validate(self):
        assert self.contract_size == 100.0
        assert 0.05 <= self.max_risk_pct_per_trade <= 5.0
        assert 1.0 <= self.max_daily_loss_pct <= 15.0
        assert 0.01 <= self.min_lot <= self.max_lot
        assert self.atr_sl_mult > 0 and self.atr_tp1_mult > 0
        assert self.spread_buffer > 0
        assert self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult

class ApexError(Exception): ...
class DataFetchError(ApexError): ...
class RiskViolationError(ApexError): ...
class StateError(ApexError): ...

def redact(v):
    s = str(v)
    for pat, rep in [
        (r"(ghp_[A-Za-z0-9_]{30,})", "ghp_****"),
        (r"(bot[0-9]+:[A-Za-z0-9_-]{30,})", "bot****:****"),
        (r"(apikey=)[A-Za-z0-9_-]+", r"\1****"),
        (r"(Bearer\s+)[A-Za-z0-9_.-]+", r"\1****"),
    ]:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

def atomic_write(p: Path, content: str):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)

class ResilientHttpClient:
    def __init__(self, timeout=12.0, max_retries=3, backoff=1.5):
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"ApexQuant/{VERSION}"})

    def request(self, method, url, **kw):
        kw.setdefault("timeout", self.timeout)
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.request(method, url, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                return r
            except Exception as e:
                last = e
                if attempt == self.max_retries: break
                st = self.backoff ** attempt + random.uniform(0.1, 0.4)
                time.sleep(st)
        raise DataFetchError(f"HTTP failed: {redact(last)}")

class KVStore:
    def __init__(self, cfg, http):
        self.cfg, self.http = cfg, http
        self.data = {"journal": [], "state": {}, "metrics": {}, "circuit_breaker": {}, "model_perf": {}}
        self.loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gist")
        self._pending: List[Future] = []
        self._lock = threading.Lock()
        atexit.register(self._drain)

    def _drain(self, timeout=15.0):
        with self._lock: p = list(self._pending)
        for f in p:
            try: f.result(timeout=timeout)
            except Exception: pass

    def force_load(self):
        if self.cfg.gist_id and self.cfg.gist_token:
            try:
                h = {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}
                r = self.http.request("GET", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=h)
                t = r.json().get("files", {}).get(self.cfg.gist_state_file)
                if t and t.get("content"):
                    self.data = json.loads(t["content"])
                    self.loaded = True
                    logging.info("State loaded from Gist.")
                    return
            except Exception as e:
                logging.warning(f"Gist read: {redact(e)}")
        p = Path(self.cfg.local_state_file)
        if p.exists():
            try:
                self.data = json.loads(p.read_text(encoding="utf-8"))
                self.loaded = True
                logging.info(f"State loaded from {p}")
                return
            except Exception: pass
        self.loaded = True

    def get(self, k, d=None): return self.data.get(k, d)
    def set(self, k, v): self.data[k] = v

    def save(self):
        s = json.dumps(self.data, indent=2, default=str)
        if not (self.cfg.gist_id and self.cfg.gist_token):
            try: atomic_write(Path(self.cfg.local_state_file), s)
            except Exception: pass
        else:
            with self._lock:
                self._pending = [f for f in self._pending if not f.done()]
                self._pending.append(self._executor.submit(self._sync_gist, s))

    def _sync_gist(self, s):
        try:
            h = {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}
            self.http.request("PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=h,
                json={"description": f"Apex State v{VERSION}", "files": {self.cfg.gist_state_file: {"content": s}}})
        except Exception: pass

class Indicators:
    @staticmethod
    def ema(s, span): return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi_series(s, period=14):
        d = s.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
        ag = g.ewm(alpha=1.0 / period, adjust=False).mean()
        al = l.ewm(alpha=1.0 / period, adjust=False).mean()
        return 100.0 - (100.0 / (1.0 + ag / (al + 1e-9)))

    @staticmethod
    def rsi(s, period=14):
        v = Indicators.rsi_series(s, period)
        return float(v.iloc[-1]) if not v.empty else 50.0

    @staticmethod
    def true_range(df):
        pc = df["close"].shift(1)
        return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)

    @staticmethod
    def atr_series(df, period=14):
        return Indicators.true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df, period=14):
        v = Indicators.atr_series(df, period)
        return float(v.iloc[-1]) if not v.empty else 0.0

    @staticmethod
    def adx_series(df, period=14):
        h, l = df["high"], df["low"]
        um, dm = h.diff(), -l.diff()
        pdm = np.where((um > dm) & (um > 0), um, 0.0)
        mdm = np.where((dm > um) & (dm > 0), dm, 0.0)
        atr = Indicators.true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()
        pdi = pd.Series(pdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        mdi = pd.Series(mdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        pi, mi = 100 * (pdi / (atr + 1e-9)), 100 * (mdi / (atr + 1e-9))
        dx = 100 * ((pi - mi).abs() / (pi + mi + 1e-9))
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def adx_last(df, period=14):
        v = Indicators.adx_series(df, period)
        return float(v.iloc[-1]) if not v.empty else 0.0

    @staticmethod
    def macd(s, f=12, sl=26, sig=9):
        ml = Indicators.ema(s, f) - Indicators.ema(s, sl)
        sli = Indicators.ema(ml, sig)
        return float(ml.iloc[-1]), float(sli.iloc[-1]), float((ml - sli).iloc[-1])

    @staticmethod
    def bollinger(s, period=20, nstd=2.0):
        sma = s.rolling(period).mean(); std = s.rolling(period).std(ddof=0)
        return float((sma + nstd * std).iloc[-1]), float(sma.iloc[-1]), float((sma - nstd * std).iloc[-1])

    @staticmethod
    def stochastic(df, period=14, smooth=3):
        lo = df["low"].rolling(period).min(); hi = df["high"].rolling(period).max()
        k_f = 100.0 * ((df["close"] - lo) / (hi - lo + 1e-9))
        k = k_f.rolling(smooth).mean(); d = k.rolling(smooth).mean()
        return float(k.iloc[-1]), float(d.iloc[-1])

    @staticmethod
    def hurst(s, max_lags=30):
        ts = s.values
        if len(ts) < max_lags * 2: return 0.5
        vl, taus = [], []
        for lag in range(2, max_lags):
            diff = ts[lag:] - ts[:-lag]
            std = float(np.std(diff, ddof=1))
            if std > 1e-12: vl.append(lag); taus.append(std)
        if len(vl) < 3: return 0.5
        try: return float(np.clip(np.polyfit(np.log(vl), np.log(taus), 1)[0], 0.0, 1.0))
        except Exception: return 0.5

    @staticmethod
    def shannon_entropy(s, bins=10):
        r = s.pct_change().dropna()
        if len(r) < bins * 2: return 0.5
        c, _ = np.histogram(r, bins=bins, density=True); c = c[c > 0]
        if c.size == 0: return 0.5
        p = c / c.sum(); e = -np.sum(p * np.log2(p))
        return float(np.clip(e / np.log2(bins), 0.0, 1.0))

    @staticmethod
    def rolling_correlation(a, b, window):
        n = min(len(a), len(b), window + 1)
        if n < 5: return 0.0
        ra = a.pct_change().dropna().iloc[-n + 1:]
        rb = b.pct_change().dropna().iloc[-n + 1:]
        k = min(len(ra), len(rb))
        if k < 5: return 0.0
        ra, rb = ra.iloc[-k:], rb.iloc[-k:]
        try:
            va, vb = float(ra.var()), float(rb.var())
            if not (math.isfinite(va) and math.isfinite(vb)): return 0.0
            if va <= 1e-12 or vb <= 1e-12: return 0.0
            cov = float(ra.cov(rb))
            if not math.isfinite(cov): return 0.0
            rho = cov / math.sqrt(va * vb)
            return float(np.clip(rho, -1.0, 1.0)) if math.isfinite(rho) else 0.0
        except Exception: return 0.0

class MarketDataHandler:
    def __init__(self, cfg, http):
        self.cfg, self.http = cfg, http
        self._td_ts = 0.0; self._dxy = None; self._dxy_ts = 0.0

    def compute_offset(self): return self.cfg.mt5_offset

    @staticmethod
    def _off(df, off):
        if off == 0.0: return df
        o = df.copy()
        for c in ("open", "high", "low", "close"): o[c] += off
        return o

    @staticmethod
    def _rs(df, rule):
        return (df.set_index("datetime").resample(rule, label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna().reset_index())

    def _td(self, intv, size):
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key: raise DataFetchError("TWELVE_DATA_API_KEY missing")
        el = time.time() - self._td_ts
        if el < self.cfg.td_min_interval_sec: time.sleep(self.cfg.td_min_interval_sec - el)
        self._td_ts = time.time()
        r = self.http.request("GET", "https://api.twelvedata.com/time_series",
            params={"symbol": "XAU/USD", "interval": intv, "outputsize": size, "apikey": key, "timezone": "UTC"})
        d = r.json()
        if "values" not in d or not d["values"]: raise DataFetchError(f"TD {intv}")
        df = pd.DataFrame(d["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns: df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(100.0)
        else: df["volume"] = 100.0
        return df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)

    def fetch_dxy(self):
        if time.time() - self._dxy_ts < 300 and self._dxy is not None: return self._dxy
        try:
            df = self._td("1h", 200); self._dxy, self._dxy_ts = df, time.time(); return df
        except Exception: return None

    def _t_td(self, off):
        m5 = self._off(self._td("5min", 800), off)
        h1 = self._off(self._td("1h", 800), off)
        h4 = self._off(self._td("4h", 800), off)
        m15 = self._off(self._rs(m5, "15min"), off)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def fetch_all(self):
        try:
            dfs, src = self._t_td(0.0)
            return dfs, src, 0.0
        except Exception as e: raise DataFetchError(f"Data failure: {redact(e)}")

MODEL_FAMILIES = {"EMA_Cross":"trend","EMA_Trend":"trend","Supertrend":"trend","Ichimoku":"trend",
    "Donchian":"breakout","Kalman":"structure","SMC_OB":"structure","MACD":"momentum",
    "RSI":"oscillator","Stochastic":"oscillator","BollingerBB":"volatility","Wavelet":"filter",
    "ML_Ensemble":"ml","DXY_Inverse":"macro"}

@dataclass
class SignalResult:
    direction: str = "NEUTRAL"; verdict: str = "CANCEL"
    buy_ratio: float = 0.0; sell_ratio: float = 0.0
    buy_score: float = 0.0; sell_score: float = 0.0
    active_models: int = 0; total_models: int = 0
    votes_detail: List[Dict[str, Any]] = field(default_factory=list)
    regime: str = "UNKNOWN"; hurst: float = 0.5; entropy: float = 0.5
    adx: float = 0.0; dxy_corr: float = 0.0; session: str = "UNKNOWN"; skip_reason: str = ""

class SignalGenerator:
    def __init__(self, cfg, kv=None):
        self.cfg, self.kv = cfg, kv
        self._registry = self._build(); self._dxy = None

    def set_dxy(self, d): self._dxy = d

    def _ema_cross(self, df):
        c = df["close"]; e9, e21 = Indicators.ema(c,9), Indicators.ema(c,21)
        d, p = e9.iloc[-1]-e21.iloc[-1], e9.iloc[-2]-e21.iloc[-2]
        atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        cf = min(1.0, abs(d)/atr)
        if d > 0 and p <= 0: return 1, min(1.0, cf+0.3)
        if d < 0 and p >= 0: return -1, min(1.0, cf+0.3)
        return (1, cf) if d > 0 else ((-1, cf) if d < 0 else (0, 0.0))

    def _ema_trend(self, df):
        c = df["close"]; e20, e50, e200 = (Indicators.ema(c,s).iloc[-1] for s in (20,50,200)); l = c.iloc[-1]
        if l > e20 > e50 > e200: return 1, 0.85
        if l < e20 < e50 < e200: return -1, 0.85
        return 0, 0.0

    def _supertrend(self, df, period=10, mult=3.0):
        atr = Indicators.atr_series(df, period); hl2 = (df["high"] + df["low"]) / 2
        ub, lb = hl2 + mult*atr, hl2 - mult*atr; c = df["close"]; n = len(df)
        if n < 2: return 0, 0.0
        fu, fl = ub.copy(), lb.copy(); trend = pd.Series(1, index=df.index)
        for i in range(1, n):
            fu.iloc[i] = ub.iloc[i] if (ub.iloc[i] < fu.iloc[i-1] or c.iloc[i-1] > fu.iloc[i-1]) else fu.iloc[i-1]
            fl.iloc[i] = lb.iloc[i] if (lb.iloc[i] > fl.iloc[i-1] or c.iloc[i-1] < fl.iloc[i-1]) else fl.iloc[i-1]
            if trend.iloc[i-1] == 1: trend.iloc[i] = -1 if c.iloc[i] < fl.iloc[i] else 1
            else: trend.iloc[i] = 1 if c.iloc[i] > fu.iloc[i] else -1
        lt = int(trend.iloc[-1]); band = fl.iloc[-1] if lt == 1 else fu.iloc[-1]
        return lt, min(1.0, abs(c.iloc[-1]-band)/(atr.iloc[-1]+1e-9))

    def _donchian(self, df, period=20):
        hi, lo = df["high"].rolling(period).max().iloc[-2], df["low"].rolling(period).min().iloc[-2]
        last = df["close"].iloc[-1]; atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        if last > hi: return 1, min(1.0, (last-hi)/atr)
        if last < lo: return -1, min(1.0, (lo-last)/atr)
        return 0, 0.0

    def _kalman(self, df):
        p = df["close"].values
        if len(p) < 30: return 0, 0.0
        x, pv, q, r = p[0], 1.0, 1e-4, 0.01; est = []
        for z in p:
            pv += q; k = pv/(pv+r); x += k*(z-x); pv = (1-k)*pv; est.append(x)
        s = pd.Series(est, index=df.index); slope = s.diff(3).iloc[-1]; atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        cf = min(1.0, abs(slope)/(atr*0.2))
        if slope > 0.05*atr: return 1, cf
        if slope < -0.05*atr: return -1, cf
        return 0, 0.0

    def _macd(self, df):
        ml, sl, h = Indicators.macd(df["close"]); atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        cf = min(1.0, abs(h)/atr)
        if ml > sl and h > 0: return 1, cf
        if ml < sl and h < 0: return -1, cf
        return 0, 0.0

    def _rsi(self, df):
        r = Indicators.rsi(df["close"], 14)
        if r < 30: return 1, (30-r)/30
        if r > 70: return -1, (r-70)/30
        if r > 55: return 1, (r-50)/50
        if r < 45: return -1, (50-r)/50
        return 0, 0.0

    def _stoch(self, df):
        k, d = Indicators.stochastic(df)
        if k < 20 and k > d: return 1, (20-k)/20
        if k > 80 and k < d: return -1, (k-80)/20
        return 0, 0.0

    def _bollinger(self, df):
        up, mid, lo = Indicators.bollinger(df["close"]); last = df["close"].iloc[-1]; bw = up - lo
        if bw <= 0: return 0, 0.0
        if last <= lo: return 1, min(1.0, (lo-last)/bw+0.5)
        if last >= up: return -1, min(1.0, (last-up)/bw+0.5)
        return 0, 0.0

    def _ichimoku(self, df):
        c, h, l = df["close"], df["high"], df["low"]
        t = (h.rolling(9).max() + l.rolling(9).min()) / 2; k = (h.rolling(26).max() + l.rolling(26).min()) / 2
        sa = ((t+k)/2).shift(26); sb = ((h.rolling(52).max() + l.rolling(52).min())/2).shift(26)
        lc, kv, tv = c.iloc[-1], k.iloc[-1], t.iloc[-1]
        s_a = sa.iloc[-1] if not pd.isna(sa.iloc[-1]) else kv; s_b = sb.iloc[-1] if not pd.isna(sb.iloc[-1]) else kv
        top, bot = max(s_a, s_b), min(s_a, s_b)
        if lc > top and tv > kv: return 1, 0.8
        if lc < bot and tv < kv: return -1, 0.8
        return 0, 0.0

    def _smc_ob(self, df):
        ob_b, ob_s = SignalGenerator.detect_ob(df, "BUY"), SignalGenerator.detect_ob(df, "SELL")
        last = df["close"].iloc[-1]; atr = Indicators.atr_last(df)
        if atr <= 0: return 0, 0.0
        if ob_b and ob_b["low"] <= last <= ob_b["high"] + 0.3*atr: return 1, 0.9
        if ob_s and ob_s["low"] - 0.3*atr <= last <= ob_s["high"]: return -1, 0.9
        return 0, 0.0

    def _wavelet(self, df):
        if not HAS_SKLEARN or len(df) < 50: return 0, 0.0
        try:
            b, a = butter(2, 0.15); sm = lfilter(b, a, df["close"].values)
            ss = pd.Series(sm); ef = ss.ewm(span=10, adjust=False).mean(); es = ss.ewm(span=30, adjust=False).mean()
            diff, slope = ef.iloc[-1]-es.iloc[-1], ef.diff(3).iloc[-1]; atr = Indicators.atr_last(df)
            if atr <= 0: return 0, 0.0
            cf = min(1.0, float(abs(diff)/atr))
            if diff > 0 and slope > 0: return 1, cf
            if diff < 0 and slope < 0: return -1, cf
            return 0, 0.0
        except Exception: return 0, 0.0

    def _ml(self, df):
        if not (HAS_SKLEARN and self.cfg.enable_ml_models) or len(df) < 300: return 0, 0.0
        try:
            c = df["close"]; f = pd.DataFrame(index=df.index)
            f["r1"] = c.pct_change(1); f["r5"] = c.pct_change(5)
            f["atr_n"] = Indicators.atr_series(df) / (c + 1e-9); f["rsi"] = Indicators.rsi_series(c, 14) / 100
            f["d20"] = (c - Indicators.ema(c, 20)) / (c + 1e-9); f["d50"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
            f = f.dropna(); h = 20; av = Indicators.atr_series(df).values
            cl, hi, lo = c.values, df["high"].values, df["low"].values; n = len(cl); y = np.zeros(n)
            for i in range(n - h):
                tp = cl[i] + 1.0*av[i]; sl = cl[i] - 1.8*av[i]
                htp = np.where(hi[i+1:i+h+1] >= tp)[0]; hsl = np.where(lo[i+1:i+h+1] <= sl)[0]
                t_tp = htp[0] if len(htp) else 999; t_sl = hsl[0] if len(hsl) else 999
                if t_tp < t_sl: y[i] = 1
                elif t_sl < t_tp: y[i] = 0
            ys = pd.Series(y[:n-h], index=df.index[:n-h]); v = f.index.intersection(ys.index)
            X, Y = f.loc[v], ys.loc[v]; split = int(len(X)*0.8); te = max(0, split - h)
            Xt, Yt = X.iloc[:te], Y.iloc[:te]; Xv, Yv = X.iloc[split:], Y.iloc[split:]
            if len(Xt) < 100 or len(Xv) < 20: return 0, 0.0
            clf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1)
            clf.fit(Xt.values, Yt.values)
            if float(clf.score(Xv.values, Yv.values)) < self.cfg.ml_min_val_acc: return 0, 0.0
            pr = clf.predict_proba(f.iloc[-1:].values)[0]; pu = float(pr[1]) if len(pr) > 1 else 0.5
            if pu > 0.62: return 1, min(1.0, (pu-0.5)*2)
            if pu < 0.38: return -1, min(1.0, (0.5-pu)*2)
            return 0, 0.0
        except Exception: return 0, 0.0

    def _dxy(self, df):
        if not self.cfg.enable_dxy_filter or self._dxy is None: return 0, 0.0
        if len(self._dxy) < 30: return 0, 0.0
        try:
            if "datetime" in df.columns:
                gold_h1 = (df.set_index("datetime").resample("1h", label="left", closed="left").agg({"close": "last"}).dropna().reset_index())
            else: gold_h1 = df
            if len(gold_h1) < 10: return 0, 0.0
            corr = Indicators.rolling_correlation(gold_h1["close"], self._dxy["close"], self.cfg.dxy_corr_window)
            dc = self._dxy["close"].reset_index(drop=True)
            e20, e50 = Indicators.ema(dc, 20).iloc[-1], Indicators.ema(dc, 50).iloc[-1]
            slope = float(dc.iloc[-1] - dc.iloc[-5]) if len(dc) >= 5 else 0.0
            if corr > -self.cfg.dxy_bias_threshold: return 0, 0.0
            st = min(1.0, abs(corr))
            if slope < 0 and e20 < e50: return 1, st
            if slope > 0 and e20 > e50: return -1, st
            return 0, 0.0
        except Exception: return 0, 0.0

    def _build(self):
        r = [("EMA_Cross", self._ema_cross, 1.0), ("EMA_Trend", self._ema_trend, 1.0),
             ("Supertrend", self._supertrend, 1.2), ("Donchian", self._donchian, 0.8),
             ("Kalman", self._kalman, 1.0), ("MACD", self._macd, 1.2),
             ("RSI", self._rsi, 1.0), ("Stochastic", self._stoch, 0.9),
             ("BollingerBB", self._bollinger, 0.9), ("Ichimoku", self._ichimoku, 1.3),
             ("SMC_OB", self._smc_ob, 1.5), ("Wavelet", self._wavelet, 0.8)]
        if self.cfg.enable_ml_models and HAS_SKLEARN: r.append(("ML_Ensemble", self._ml, 1.0))
        if self.cfg.enable_dxy_filter: r.append(("DXY_Inverse", self._dxy, 1.4))
        return r

    def _aw(self, name, base):
        if not self.cfg.enable_adaptive_weights or self.kv is None: return base
        p = self.kv.get("model_perf", {}).get(name, {}); n, h = p.get("n", 0), p.get("hits", 0)
        if n < self.cfg.adaptive_min_trades: return base
        acc = h / max(1, n); f = float(np.clip(1.0 + (acc - 0.5) * 3.0, self.cfg.adaptive_weight_floor, self.cfg.adaptive_weight_ceil))
        return base * f

    def _fam_cap(self, raw):
        ft = defaultdict(lambda: {1: 0.0, -1: 0.0})
        for name, (v, cf, sc) in raw.items():
            if v != 0: ft[MODEL_FAMILIES.get(name, "other")][v] += sc
        b = s = 0.0
        for name, (v, cf, sc) in raw.items():
            if v == 0: continue
            fam = MODEL_FAMILIES.get(name, "other"); fd = max(ft[fam][1], ft[fam][-1])
            mx = max((raw[k][2] for k in raw if MODEL_FAMILIES.get(k, "other") == fam), default=1.0)
            cap = self.cfg.family_cap_multiplier * max(1.0, mx)
            sc2 = min(1.0, cap / max(fd, 1e-6)) if fd > 0 else 1.0; adj = sc * sc2
            if v > 0: b += adj
            else: s += adj
        return b, s

    def _regime(self, h1):
        H, E = Indicators.hurst(h1["close"]), Indicators.shannon_entropy(h1["close"])
        if E >= self.cfg.entropy_chaos_max: return "CHAOTIC", H, E
        if H >= self.cfg.hurst_trend_min: return "TRENDING", H, E
        if H <= self.cfg.hurst_meanrev_max: return "MEAN_REVERTING", H, E
        return "RANGING", H, E

    @staticmethod
    def _htf(h4):
        if len(h4) < 50: return "NEUTRAL"
        c = h4["close"]; e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return "BEARISH"
        return "NEUTRAL"

    def _mr(self, reg):
        return {"TRENDING": self.cfg.min_ratio_trending, "MEAN_REVERTING": self.cfg.min_ratio_meanrev,
                "RANGING": self.cfg.min_ratio_ranging}.get(reg, self.cfg.min_ratio_default)

    def generate(self, m5, m15, h1, h4, dxy=None):
        if dxy is not None: self.set_dxy(dxy)
        res = SignalResult(); res.total_models = len(self._registry); res.adx = Indicators.adx_last(m5)
        reg, H, E = self._regime(h1); res.regime, res.hurst, res.entropy = reg, H, E
        htf = self._htf(h4); raw, details, active = {}, [], 0
        for name, fn, w in self._registry:
            try: v, cf = fn(m5)
            except Exception: v, cf = 0, 0.0
            aw = self._aw(name, w); sc = aw * cf if v != 0 else 0.0
            raw[name] = (v, cf, sc)
            if v != 0: active += 1
            details.append({"model": name, "vote": {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[v],
                "family": MODEL_FAMILIES.get(name, "other"), "conf": round(cf, 2), "weight": round(aw, 2), "score": round(sc, 2)})
        bw, sw = self._fam_cap(raw); res.buy_score, res.sell_score = round(bw, 2), round(sw, 2)
        res.active_models, res.votes_detail = active, details
        if self._dxy is not None and len(self._dxy) > 20:
            try: res.dxy_corr = round(Indicators.rolling_correlation(m5["close"], self._dxy["close"], self.cfg.dxy_corr_window), 3)
            except Exception: res.dxy_corr = 0.0
        tw = bw + sw
        if tw > 1e-6: res.buy_ratio, res.sell_ratio = bw / tw, sw / tw
        if res.adx < self.cfg.adx_min: res.skip_reason = f"ADX {res.adx:.1f} < {self.cfg.adx_min}"; return res
        if reg == "CHAOTIC": res.skip_reason = f"CHAOTIC E={E:.2f}"; return res
        if reg == "RANGING" and not self.cfg.allow_ranging_regime: res.skip_reason = f"RANGING H={H:.2f}"; return res
        if tw < 1e-6: res.skip_reason = "no directional"; return res
        if active < self.cfg.parliament_min_active: res.skip_reason = f"active {active} < quorum {self.cfg.parliament_min_active}"; return res
        dom = max(res.buy_ratio, res.sell_ratio); mr = self._mr(reg)
        if dom < mr: res.skip_reason = f"consensus {dom:.0%} < {reg} min {mr:.0%}"; return res
        if dom >= self.cfg.parliament_full_ratio: res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio: res.verdict = "EXECUTE_HALF"
        else: res.verdict = "EXECUTE_MICRO"
        if res.buy_ratio > res.sell_ratio:
            if htf == "BEARISH" and self.cfg.enforce_htf_filter:
                if dom >= self.cfg.counter_trend_allow_ratio:
                    res.direction = "BUY"
                    if res.verdict == "EXECUTE_FULL": res.verdict = "EXECUTE_HALF"
                else: res.skip_reason = f"BUY blocked H4 BEARISH ({dom:.0%})"; return res
            else: res.direction = "BUY"
        else:
            if htf == "BULLISH" and self.cfg.enforce_htf_filter:
                if dom >= self.cfg.counter_trend_allow_ratio:
                    res.direction = "SELL"
                    if res.verdict == "EXECUTE_FULL": res.verdict = "EXECUTE_HALF"
                else: res.skip_reason = f"SELL blocked H4 BULLISH ({dom:.0%})"; return res
            else: res.direction = "SELL"
        return res

    @staticmethod
    def detect_ob(df, side, lookback=40):
        if len(df) < lookback + 5: return None
        tail = df.tail(lookback + 5).reset_index(drop=True); atr = Indicators.atr_last(df)
        if atr <= 0: return None
        for i in range(len(tail)-2, 2, -1):
            body = abs(tail["close"].iloc[i] - tail["open"].iloc[i])
            if body < 1.2*atr: continue
            bull = tail["close"].iloc[i] > tail["open"].iloc[i]
            if side == "BUY" and not bull: continue
            if side == "SELL" and bull: continue
            for j in range(i-1, max(i-10, 0), -1):
                opp = tail["close"].iloc[j] < tail["open"].iloc[j]
                if side == "BUY" and opp: return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
                if side == "SELL" and not opp: return {"high": float(tail["high"].iloc[j]), "low": float(tail["low"].iloc[j])}
        return None

    def _sam(self, s):
        return {"LONDON_OPEN": self.cfg.session_atr_london, "NY_OPEN": self.cfg.session_atr_ny_open,
                "NY_AFTERNOON": self.cfg.session_atr_ny_afternoon, "ASIAN": self.cfg.session_atr_asian}.get(s, 1.0)

    def compute_levels(self, m5, side, ob, session="LONDON_OPEN"):
        atr = Indicators.atr_last(m5) * self._sam(session)
        if atr <= 0: raise ValueError("ATR must be positive")
        px = float(m5["close"].iloc[-1]); dec = self.cfg.price_decimals; sb = self.cfg.spread_buffer
        obh = ob["high"] if ob else px; obl = ob["low"] if ob else px
        if side == "BUY":
            obc = (obh - (obh-obl)*0.5) if (ob and obh < px) else None
            if obc and (px-obc) <= self.cfg.entry_max_chase_atr*atr: entry, slr = obc, obl
            else: entry, slr = px - atr*self.cfg.entry_pullback_atr, px - atr*self.cfg.entry_pullback_atr - atr*self.cfg.atr_sl_mult
            entry = min(entry, px - atr*0.05)
            if (px-entry) > self.cfg.entry_max_chase_atr*atr: entry = px - atr*self.cfg.entry_pullback_atr
            sl = min(slr, entry-atr*0.5) - sb
            if entry <= sl: sl = entry - atr*self.cfg.atr_sl_mult - sb
            tp1, tp2, tp3 = entry+atr*self.cfg.atr_tp1_mult, entry+atr*self.cfg.atr_tp2_mult, entry+atr*self.cfg.atr_tp3_mult
            if not (sl < entry < tp1 < tp2 < tp3): raise ValueError("BUY incoherent")
            return tuple(round(x, dec) for x in (entry, sl, tp1, tp2, tp3))
        else:
            obc = (obl + (obh-obl)*0.5) if (ob and obl > px) else None
            if obc and (obc-px) <= self.cfg.entry_max_chase_atr*atr: entry, slr = obc, obh
            else: entry, slr = px + atr*self.cfg.entry_pullback_atr, px + atr*self.cfg.entry_pullback_atr + atr*self.cfg.atr_sl_mult
            entry = max(entry, px + atr*0.05)
            if (entry-px) > self.cfg.entry_max_chase_atr*atr: entry = px + atr*self.cfg.entry_pullback_atr
            sl = max(slr, entry+atr*0.5) + sb
            if sl <= entry: sl = entry + atr*self.cfg.atr_sl_mult + sb
            tp1, tp2, tp3 = entry-atr*self.cfg.atr_tp1_mult, entry-atr*self.cfg.atr_tp2_mult, entry-atr*self.cfg.atr_tp3_mult
            if not (sl > entry > tp1 > tp2 > tp3): raise ValueError("SELL incoherent")
            return tuple(round(x, dec) for x in (entry, sl, tp1, tp2, tp3))

class RiskManagementEngine:
    CLOSED = ("WIN_TP2", "WIN_TP3", "WIN_BE", "LOSS")
    def __init__(self, cfg, kv): self.cfg, self.kv = cfg, kv

    def dynamic_balance(self):
        base = self.cfg.account_balance
        j = [x for x in (self.kv.get("journal", []) or []) if x.get("r_achieved") is not None and x.get("outcome") in self.CLOSED]
        j.sort(key=lambda x: x.get("timestamp", ""))
        for t in j:
            try: base += float(t.get("r_achieved", 0.0)) * (base * (float(t.get("risk_pct", 1.0)) / 100))
            except (ValueError, TypeError): pass
        return max(10.0, round(base, 2))

    def _dds(self, bal):
        cb = self.kv.get("circuit_breaker", {}) or {}; ds = cb.get("day_start_balance", bal)
        if ds <= 0: return 1.0
        dd = max(0.0, (ds-bal)/ds*100)
        if dd <= self.cfg.dd_scale_threshold_pct: return 1.0
        span = self.cfg.max_daily_loss_pct - self.cfg.dd_scale_threshold_pct
        if span <= 0: return self.cfg.dd_scale_min_factor
        prog = min(1.0, (dd-self.cfg.dd_scale_threshold_pct)/span)
        return float(np.clip(1.0 - prog*(1.0-self.cfg.dd_scale_min_factor), self.cfg.dd_scale_min_factor, 1.0))

    def compute_lot(self, bal, entry, sl, conf):
        d = abs(entry-sl)
        if d < 1e-4: raise ValueError("SL too small")
        cf = max(0.4, min(1.0, conf/100)); rp = max(0.1, self.cfg.max_risk_pct_per_trade * cf) * self._dds(bal)
        rd = bal * (rp/100); lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(rd/(d*self.cfg.contract_size), 2)))
        nv = lot * self.cfg.contract_size * entry
        if bal > 0 and (nv/bal) > self.cfg.hard_leverage_cap:
            ml = (self.cfg.hard_leverage_cap*bal)/(self.cfg.contract_size*entry)
            if ml < self.cfg.min_lot: raise RiskViolationError("Leverage cap exceeded")
            lot = max(self.cfg.min_lot, round(ml, 2))
        return lot, round((lot*self.cfg.contract_size*d)/bal*100, 2)

    def check_cb(self, bal):
        today = datetime.now(UTC).strftime("%Y-%m-%d"); cb = self.kv.get("circuit_breaker", {}) or {}
        lk = cb.get("locked_until")
        if lk:
            try:
                if datetime.now(UTC) < datetime.fromisoformat(lk): return False, f"LOCKED until {lk}"
                else: del cb["locked_until"]; self.kv.set("circuit_breaker", cb)
            except Exception: pass
        ds = cb.get("day_start_balance")
        if cb.get("last_date") != today or ds is None:
            cb["last_date"] = today; cb["day_start_balance"] = bal; cb["daily_signals_count"] = 0
            self.kv.set("circuit_breaker", cb); ds = bal
        dd = (ds-bal)/ds*100
        if dd >= self.cfg.max_daily_loss_pct:
            lkt = (datetime.now(UTC)+timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lkt; self.kv.set("circuit_breaker", cb); self.kv.save()
            return False, f"DAILY DD {dd:.2f}%"
        j = [x for x in (self.kv.get("journal", []) or []) if x.get("outcome") in self.CLOSED]
        c = 0
        for t in reversed(j):
            if t.get("outcome") == "LOSS": c += 1
            else: break
        if c >= self.cfg.max_consecutive_losses:
            lkt = (datetime.now(UTC)+timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lkt; self.kv.set("circuit_breaker", cb); self.kv.save()
            return False, f"CONSEC {c}"
        if cb.get("daily_signals_count", 0) >= self.cfg.max_daily_signals: return False, "DAILY CAP"
        return True, "clear"

class NewsFilter:
    def __init__(self, cfg, http):
        self.cfg, self.http = cfg, http; self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self._c, self._ts = [], 0.0
    def _fetch(self):
        if time.time() - self._ts < 3600 and self._c: return self._c
        try:
            d = self.http.request("GET", self.url).json()
            if isinstance(d, list): self._c, self._ts = d, time.time(); return d
        except Exception: pass
        return self._c
    def blackout(self, now=None):
        if now is None: now = datetime.now(UTC)
        w = timedelta(minutes=self.cfg.news_blackout_minutes)
        for ev in self._fetch():
            if ev.get("country") != "USD" or ev.get("impact") not in ("High", "Holiday"): continue
            ds = ev.get("date", "")
            if not ds: continue
            try:
                et = datetime.fromisoformat(ds)
                et = et.replace(tzinfo=UTC) if et.tzinfo is None else et.astimezone(UTC)
                if (et-w) <= now <= (et+w): return True, f"{html.escape(str(ev.get('title', 'High-Impact')))} ({int((et-now).total_seconds()/60)}m)"
            except Exception: continue
        return False, ""

class TelegramNotifier:
    def __init__(self, http, dry=False):
        self.token, self.chat, self.http, self.dry = os.getenv("TELEGRAM_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip(), http, dry
    def send(self, msg):
        if self.dry or not (self.token and self.chat): logging.info(f"[DRY-RUN TG]\n{msg}"); return True
        try:
            self.http.request("POST", f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True})
            return True
        except Exception as e: logging.error(f"TG fail: {redact(e)}"); return False

class OutcomeTracker:
    def __init__(self, cfg, kv): self.cfg, self.kv = cfg, kv

    def update(self, m5):
        journal = self.kv.get("journal", []) or []; events = []; now = datetime.now(UTC); mod = False; perf = self.kv.get("model_perf", {}) or {}
        for e in journal:
            if e.get("outcome") != "PENDING": continue
            try: anchor = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except Exception: anchor = now - timedelta(hours=24)
            if not e.get("filled") and (now-anchor).total_seconds()/3600 > self.cfg.pending_stale_hours:
                e["outcome"] = "STALE"; e["filled"] = 0; e["r_achieved"] = 0.0; events.append(("STALE", dict(e))); mod = True; continue
            side = e["signal"]; ep, sl = float(e["entry"]), float(e["sl"]); tp1, tp2, tp3 = float(e["tp1"]), float(e["tp2"]), float(e["tp3"])
            if ep <= 0 or sl <= 0: continue
            bars = m5[m5["datetime"] > anchor].sort_values("datetime")
            if bars.empty: continue
            filled = bool(e.get("filled")); fi = None
            if not filled:
                for i, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep: filled, fi = True, i; break
                    if side == "SELL" and b.high >= ep: filled, fi = True, i; break
                if not filled:
                    ch = self.cfg.entry_max_chase_atr * abs(ep-sl); lh, ll = float(bars["high"].iloc[-1]), float(bars["low"].iloc[-1])
                    if (side == "BUY" and lh > ep+ch) or (side == "SELL" and ll < ep-ch):
                        e["outcome"] = "INVALIDATED"; e["filled"] = 0; e["r_achieved"] = 0.0; events.append(("INVALIDATED", dict(e))); mod = True
                    continue
                e["filled"] = 1; mod = True
            walk = bars.iloc[fi:] if fi is not None else bars
            h1, h2, h3 = bool(e.get("hit_tp1")), bool(e.get("hit_tp2")), bool(e.get("hit_tp3"))
            hs, hb = bool(e.get("hit_sl")), bool(e.get("hit_be")); trail = sl; prev_high = float(walk["high"].iloc[0]); prev_low = float(walk["low"].iloc[0])
            for b in walk.itertuples(index=False):
                if h3 or hs or hb: break
                if h2:
                    atr = Indicators.atr_last(m5); td = atr * self.cfg.trailing_atr_mult
                    if side == "BUY": trail = max(trail, prev_high - td)
                    else: trail = min(trail, prev_low + td)
                    cs = trail
                elif h1: cs = ep
                else: cs = sl
                if side == "BUY":
                    if b.low <= cs:
                        if h1: hb = True
                        else: hs = True
                        continue
                    if not h1 and b.high >= tp1: h1 = True
                    if h1 and not h2 and b.high >= tp2: h2 = True
                    if h2 and not h3 and b.high >= tp3: h3 = True
                else:
                    if b.high >= cs:
                        if h1: hb = True
                        else: hs = True
                        continue
                    if not h1 and b.low <= tp1: h1 = True
                    if h1 and not h2 and b.low <= tp2: h2 = True
                    if h2 and not h3 and b.low <= tp3: h3 = True
                prev_high, prev_low = float(b.high), float(b.low)
            e["hit_tp1"], e["hit_tp2"], e["hit_tp3"], e["hit_sl"], e["hit_be"] = int(h1), int(h2), int(h3), int(hs), int(hb)
            d = abs(ep-sl)
            if d < 1e-9: continue
            r1, r2, r3 = abs(tp1-ep)/d, abs(tp2-ep)/d, abs(tp3-ep)/d
            closed = False
            if h3: e["outcome"] = "WIN_TP3"; e["r_achieved"] = round((r1+r2+r3)/3, 4); closed = True
            elif h2 and hb: e["outcome"] = "WIN_TP2"; e["r_achieved"] = round((r1+r2)/3, 4); closed = True
            elif h1 and hb: e["outcome"] = "WIN_BE"; e["r_achieved"] = round(r1/3, 4); closed = True
            elif hs: e["outcome"] = "LOSS"; e["r_achieved"] = -1.0; closed = True
            if closed:
                events.append(("CLOSE", dict(e))); mod = True; vs = e.get("votes_snapshot", []); corr = e["outcome"].startswith("WIN")
                for v in vs:
                    mn, vd = v.get("model"), v.get("vote")
                    if not mn or vd == "NEUTRAL": continue
                    p = perf.setdefault(mn, {"n": 0, "hits": 0}); p["n"] = int(p.get("n", 0)) + 1
                    if (vd == side and corr) or (vd != side and not corr): p["hits"] = int(p.get("hits", 0)) + 1
                self.kv.set("model_perf", perf)
        if mod: self.kv.set("journal", journal); self.kv.save()
        return events

class Orchestrator:
    def __init__(self, cfg):
        self.cfg = cfg; self.http = ResilientHttpClient(cfg.http_timeout_sec, cfg.http_max_retries, cfg.http_backoff_factor)
        self.kv = KVStore(cfg, self.http); self.data = MarketDataHandler(cfg, self.http)
        self.signal = SignalGenerator(cfg, self.kv); self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsFilter(cfg, self.http); self.tg = TelegramNotifier(self.http, cfg.dry_run)
        self.outcome = OutcomeTracker(cfg, self.kv)

    def _sess(self, now):
        lf, nf = now.astimezone(LONDON_TZ), now.astimezone(NY_TZ)
        lh, nh = lf.hour+lf.minute/60, nf.hour+nf.minute/60
        if 7 <= lh < 10: return "LONDON_OPEN"
        if 8 <= nh < 11: return "NY_OPEN"
        if 13 <= nh < 16: return "NY_AFTERNOON"
        if 0 <= lh < 6: return "ASIAN"
        return "OFF_HOURS"
    def _in_kz(self, now): return True if self.cfg.skip_killzone_filter else self._sess(now) in ("LONDON_OPEN","NY_OPEN","NY_AFTERNOON")

    def run(self):
        try: self.kv.force_load()
        except Exception: pass
        now = datetime.now(UTC); now_str = now.strftime("%Y-%m-%d %H:%M:%S"); sess = self._sess(now)
        if self.cfg.kill_switch or now.weekday() >= 5 or not self._in_kz(now): return
        bal = self.risk.dynamic_balance(); ok, rsn = self.risk.check_cb(bal)
        if not ok: return
        try: dfs, src, off = self.data.fetch_all()
        except Exception as e: self.tg.send(f"⚠️ Data Err: {redact(e)}"); return
        dxy = self.data.fetch_dxy() if self.cfg.enable_dxy_filter else None
        
        try:
            for evt, tr in self.outcome.update(dfs["5min"]):
                if evt == "CLOSE": self.tg.send(f"📕 <b>Closed</b> {html.escape(str(tr['signal']))} → <b>{html.escape(str(tr['outcome']))}</b> ({float(tr.get('r_achieved',0) or 0):+.2f}R)")
                elif evt in ("INVALIDATED","STALE"): self.tg.send(f"⚪ <b>{evt}</b> {html.escape(str(tr['signal']))} @ {html.escape(str(tr['entry']))}")
        except Exception: pass
        
        bal = self.risk.dynamic_balance()
        m5 = dfs["5min"].iloc[:-1].reset_index(drop=True); m15 = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1 = dfs["1h"].iloc[:-1].reset_index(drop=True); h4 = dfs["4h"].iloc[:-1].reset_index(drop=True)
        if len(m5) < 100 or len(h1) < 100: return
        lpx = float(m5["close"].iloc[-1]); biso = m5.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        journal = self.kv.get("journal", []) or []
        
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal): return
        if sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled")) >= 2: return
        
        res = self.signal.generate(m5, m15, h1, h4, dxy=dxy); res.session = sess
        if res.direction == "NEUTRAL" or res.verdict == "CANCEL": return
        
        st = self.kv.get("state", {}) or {}; ld, lt = st.get("last_signal_type", "NEUTRAL"), st.get("last_signal_ts", "")
        cool = self.cfg.opposite_cooldown_minutes if ld != res.direction else self.cfg.cooldown_minutes
        if lt:
            try:
                if (now - datetime.strptime(lt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)) < timedelta(minutes=cool): return
            except Exception: pass
            
        ob = self.signal.detect_ob(m5, res.direction)
        try: entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5, res.direction, ob, sess)
        except ValueError: return
        
        conf = max(res.buy_ratio, res.sell_ratio) * 100
        try: lot, ar = self.risk.compute_lot(bal, entry, sl, conf)
        except Exception: return
        
        if res.verdict == "EXECUTE_HALF": lot = max(self.cfg.min_lot, round(lot / 2, 2))
        elif res.verdict == "EXECUTE_MICRO": lot = max(self.cfg.min_lot, round(lot / 4, 2))
        
        ob_d = f"✅ {ob['high']:.2f}–{ob['low']:.2f}" if ob else "⚠️ ATR"
        msg = (f"🏛️ <b>APEX v{VERSION} — {html.escape(res.direction)} LIMIT</b>\n\n"
               f"Entry: <code>{entry:.2f}</code> | SL: <code>{sl:.2f}</code>\n\n"
               f"🎯 TP1: <code>{tp1:.2f}</code>\n🎯 TP2: <code>{tp2:.2f}</code>\n🎯 TP3: <code>{tp3:.2f}</code>\n\n"
               f"📦 <b>Lot: <code>{lot}</code></b>\n"
               f"💰 Bal: ${bal:.2f} | Risk: {ar:.2f}%\n"
               f"📊 B={res.buy_ratio:.0%} S={res.sell_ratio:.0%}\n"
               f"🧠 {html.escape(res.verdict)} | Models {res.active_models}/{res.total_models}\n"
               f"🌐 {html.escape(res.regime)} (H={res.hurst:.2f}, E={res.entropy:.2f})\n"
               f"💵 DXY: {res.dxy_corr:+.2f}\n🕐 {html.escape(sess)}\n"
               f"🧱 SMC: {html.escape(ob_d)}\n\n⏰ Bar: <code>{html.escape(biso)} UTC</code>")
               
        if self.tg.send(msg):
            journal.append({
                "id": hashlib.md5(f"{biso}_{res.direction}".encode()).hexdigest()[:10],
                "timestamp": now_str, "symbol": self.cfg.symbol, "signal": res.direction,
                "verdict": res.verdict, "session": sess, "regime": res.regime, "dxy_corr": res.dxy_corr,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "lot": lot, "risk_pct": ar,
                "filled": 0, "outcome": "PENDING", "r_achieved": None,
                "hit_tp1": 0, "hit_tp2": 0, "hit_tp3": 0, "hit_sl": 0, "hit_be": 0, "votes_snapshot": res.votes_detail
            })
            self.kv.set("journal", journal); cb = self.kv.get("circuit_breaker", {}) or {}
            cb["daily_signals_count"] = cb.get("daily_signals_count", 0) + 1; self.kv.set("circuit_breaker", cb)
            st["last_signal_type"] = res.direction; st["last_signal_ts"] = now_str; self.kv.set("state", st)
        self._st(lpx, now_str)

    def _st(self, px, ts):
        s = self.kv.get("state", {}) or {}; s["last_run_utc"] = ts; s["last_gold_price"] = px; s["engine_version"] = VERSION
        self.kv.set("state", s); self.kv.save()

class BacktestEngine:
    def __init__(self, cfg): self.cfg = cfg; self.signal = SignalGenerator(cfg, kv=None)
    def load_csv(self, path):
        df = pd.read_csv(path); df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 100.0), errors="coerce").fillna(100.0)
        return df.dropna().sort_values("datetime").reset_index(drop=True)
    def _rs(self, df, rule):
        return (df.set_index("datetime").resample(rule, label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index())

    def run(self, df, max_bars=None):
        if len(df) < 300: raise ValueError("Need >= 300 bars")
        h1 = self._rs(df, "1h"); h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True)
        h4 = self._rs(df, "4h"); h4["datetime"] = pd.to_datetime(h4["datetime"], utc=True)
        trades = []; bal = self.cfg.account_balance; start = 200; end = len(df) if max_bars is None else min(len(df), start + max_bars)
        active_trade = None
        for i in range(start, end):
            ts = df.iloc[i]["datetime"]
            if active_trade:
                if not active_trade["filled"]:
                    if (active_trade["side"] == "BUY" and df.iloc[i]["low"] <= active_trade["entry"]) or \
                       (active_trade["side"] == "SELL" and df.iloc[i]["high"] >= active_trade["entry"]):
                        active_trade["filled"] = True
                    elif (ts - active_trade["signal_time"]).total_seconds() > self.cfg.pending_stale_hours * 3600:
                        active_trade = None; continue
                    else: continue
                if active_trade["filled"]:
                    res = self._sim_bar(active_trade, df.iloc[i])
                    if res:
                        bal *= (1 + res["r"] * 0.01)
                        trades.append({"outcome": res["outcome"], "r": res["r"], "balance": round(bal, 2)})
                        active_trade = None
                continue
            hist = df.iloc[:i].reset_index(drop=True)
            h1h = h1[h1["datetime"] <= ts].reset_index(drop=True); h4h = h4[h4["datetime"] <= ts].reset_index(drop=True)
            if len(h1h) < 100 or len(h4h) < 50: continue
            sig = self.signal.generate(hist, hist, h1h, h4h)
            if sig.direction == "NEUTRAL" or sig.verdict == "CANCEL": continue
            ob = self.signal.detect_ob(hist, sig.direction)
            try: entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(hist, sig.direction, ob)
            except ValueError: continue
            active_trade = {"side": sig.direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                            "signal_time": ts, "filled": False, "hit_tp1": False, "hit_tp2": False}
        if not trades: return {"trades": 0, "message": "No signals"}
        w = sum(1 for t in trades if t["outcome"].startswith("WIN")); l = sum(1 for t in trades if t["outcome"] == "LOSS")
        tr = sum(t["r"] for t in trades); eq = pd.Series([t["balance"] for t in trades])
        mdd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
        return {"trades": len(trades), "wins": w, "losses": l, "win_rate": round(w / len(trades) * 100, 2),
                "total_r": round(tr, 2), "final_balance": round(bal, 2), "return_pct": round((bal / self.cfg.account_balance - 1) * 100, 2),
                "max_drawdown_pct": round(mdd, 2), "avg_r": round(tr / len(trades), 3)}

    @staticmethod
    def _sim_bar(t, bar):
        side = t["side"]; sl, entry = t["sl"], t["entry"]; tp1, tp2, tp3 = t["tp1"], t["tp2"], t["tp3"]
        d = abs(entry - sl); r1, r2, r3 = abs(tp1 - entry)/d, abs(tp2 - entry)/d, abs(tp3 - entry)/d
        if side == "BUY":
            if bar["low"] <= sl: return {"outcome": "LOSS", "r": -1.0}
            if bar["high"] >= tp3: return {"outcome": "WIN_TP3", "r": (r1+r2+r3)/3.0}
            if not t["hit_tp1"] and bar["high"] >= tp1: t["hit_tp1"] = True
            if t["hit_tp1"] and not t["hit_tp2"] and bar["high"] >= tp2: t["hit_tp2"] = True
            if t["hit_tp1"] and bar["low"] <= entry:
                 if t["hit_tp2"]: return {"outcome": "WIN_TP2", "r": (r1+r2)/3.0}
                 else: return {"outcome": "WIN_BE", "r": r1/3.0}
        else:
            if bar["high"] >= sl: return {"outcome": "LOSS", "r": -1.0}
            if bar["low"] <= tp3: return {"outcome": "WIN_TP3", "r": (r1+r2+r3)/3.0}
            if not t["hit_tp1"] and bar["low"] <= tp1: t["hit_tp1"] = True
            if t["hit_tp1"] and not t["hit_tp2"] and bar["low"] <= tp2: t["hit_tp2"] = True
            if t["hit_tp1"] and bar["high"] >= entry:
                 if t["hit_tp2"]: return {"outcome": "WIN_TP2", "r": (r1+r2)/3.0}
                 else: return {"outcome": "WIN_BE", "r": r1/3.0}
        return None

_ORCH: Optional[Orchestrator] = None
def _sig(signum, frame):
    if _ORCH is not None:
        try: _ORCH.kv._drain(timeout=10.0)
        except Exception: pass
    sys.exit(0)
def _log(v=False): logging.basicConfig(level=logging.DEBUG if v else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    global _ORCH
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run", choices=["run","test","audit","state","version","health","backtest"])
    parser.add_argument("-v", "--verbose", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv"); parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args(); _log(args.verbose)
    if args.command == "version": print(f"APEX v{VERSION}"); return
    
    cfg = Config.from_env()
    if args.dry_run: cfg.dry_run = True
    cfg.validate()
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    
    if args.command == "backtest":
        if not args.csv: print("❌ --csv required"); sys.exit(1)
        bt = BacktestEngine(cfg); df = bt.load_csv(args.csv)
        print(f"📊 Loaded {len(df)} bars"); print(json.dumps(bt.run(df, args.max_bars), indent=2)); return
        
    orch = Orchestrator(cfg); _ORCH = orch
    if args.command == "run": orch.run()
    elif args.command == "state": orch.kv.force_load(); print(json.dumps(orch.kv.data, indent=2, default=str))
    elif args.command == "audit":
        orch.kv.force_load(); j = orch.kv.get("journal", []) or []
        cl = [x for x in j if x.get("outcome") in RiskManagementEngine.CLOSED]
        w = sum(1 for x in cl if str(x.get("outcome", "")).startswith("WIN")); l = sum(1 for x in cl if x.get("outcome") == "LOSS")
        tr = sum(float(x.get("r_achieved", 0) or 0) for x in cl)
        print(f"Trades: {len(j)} | Closed: {len(cl)}\nWins: {w} | Losses: {l} | WR: {w/max(1,len(cl))*100:.1f}%\nTotal R: {tr:+.2f}R | Avg: {tr/max(1,len(cl)):.3f}R")
    elif args.command == "test":
        orch.kv.force_load(); dfs, src, off = orch.data.fetch_all(); logging.info(f"OK: src={src}, M5={len(dfs['5min'])}")

if __name__ == "__main__":
    main()
