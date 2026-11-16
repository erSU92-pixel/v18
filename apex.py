#!/usr/bin/env python3
"""
===============================================================================
APEX PREDATOR v22.0 — SMC LIQUIDITY SWEEP & FVG (INSTITUTIONAL EDITION)
===============================================================================
Strategy: Smart Money Concepts (Order Flow).
Focus   : Asian Range Sweep -> Fair Value Gap -> Entry -> Opposing Liquidity.
Time    : London & New York Killzones ONLY.
===============================================================================
"""
from __future__ import annotations

import subprocess, sys, os

def _ensure_deps():
    req = ["numpy>=1.24,<3.0", "pandas>=2.0,<3.0", "requests>=2.31,<3.0"]
    missing = []
    for mod, pkg in [("numpy", req[0]), ("pandas", req[1]), ("requests", req[2])]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing: 
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_ensure_deps()

import argparse, atexit, hashlib, json, logging, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests

VERSION = "22.0-SMC"
NY_TZ, LONDON_TZ, UTC = ZoneInfo("America/New_York"), ZoneInfo("Europe/London"), timezone.utc

@dataclass
class Config:
    gist_id: str = ""; gist_token: str = ""
    gist_state_file: str = "apex_state_v22.json"; local_state_file: str = "apex_state_v22.json"
    symbol: str = "XAUUSD"
    
    account_balance: float = 1000.0
    risk_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    
    atr_period: int = 14
    sl_atr_mult: float = 1.0     # SL sangat ketat di ujung Sweep
    tp_rr_ratio: float = 3.0     # Minimal 1:3 RR
    
    skip_killzone: bool = False
    dry_run: bool = False
    
    @classmethod
    def from_env(cls):
        def es(k, d): return os.getenv(k, d).strip()
        def ef(k, d): 
            v = os.getenv(k, "").strip()
            try: return float(v) if v else d
            except: return d
        def eb(k, d):
            v = os.getenv(k, "").strip().lower()
            return d if not v else v in ("1", "true", "yes", "on")
        
        c = cls()
        c.gist_id = es("GIST_ID", c.gist_id)
        c.gist_token = es("GIST_TOKEN", c.gist_token)
        c.account_balance = ef("APEX_BALANCE", c.account_balance)
        c.risk_pct = ef("APEX_RISK_PCT", c.risk_pct)
        c.max_daily_loss_pct = ef("APEX_MAX_DAILY_LOSS", c.max_daily_loss_pct)
        c.skip_killzone = eb("APEX_SKIP_KILLZONE_FILTER", c.skip_killzone)
        c.dry_run = eb("APEX_DRY_RUN", c.dry_run)
        return c

def redact(v):
    s = str(v)
    for pat, rep in [(r"(ghp_[A-Za-z0-9_]{30,})", "ghp_****"), (r"(bot[0-9]+:[A-Za-z0-9_-]{30,})", "bot****:****")]:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

class ResilientHttpClient:
    def __init__(self, timeout=12.0, max_retries=3):
        self.timeout, self.max_retries = timeout, max_retries
        self.session = requests.Session(); self.session.headers.update({"User-Agent": f"ApexQuant/{VERSION}"})
    def request(self, method, url, **kw):
        kw.setdefault("timeout", self.timeout); last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.request(method, url, **kw)
                r.raise_for_status(); return r
            except Exception as e:
                last = e
                if attempt == self.max_retries: break
                time.sleep(1.5 ** attempt)
        raise Exception(f"HTTP failed: {redact(last)}")

class KVStore:
    def __init__(self, cfg, http):
        self.cfg, self.http = cfg, http
        self.data = {"journal": [], "state": {}, "circuit_breaker": {}}
        self._executor = ThreadPoolExecutor(max_workers=1); self._pending = []; self._lock = threading.Lock()
        atexit.register(self._drain)
    def _drain(self, timeout=10.0):
        with self._lock: p = list(self._pending)
        for f in p:
            try: f.result(timeout=timeout)
            except: pass
    def force_load(self):
        if self.cfg.gist_id and self.cfg.gist_token:
            try:
                h = {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}
                r = self.http.request("GET", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=h)
                t = r.json().get("files", {}).get(self.cfg.gist_state_file)
                if t and t.get("content"):
                    self.data = json.loads(t["content"]); return
            except: pass
        p = Path(self.cfg.local_state_file)
        if p.exists():
            try: self.data = json.loads(p.read_text(encoding="utf-8")); return
            except: pass
    def get(self, k, d=None): return self.data.get(k, d)
    def set(self, k, v): self.data[k] = v
    def save(self):
        s = json.dumps(self.data, indent=2, default=str)
        if not (self.cfg.gist_id and self.cfg.gist_token):
            try: Path(self.cfg.local_state_file).write_text(s, encoding="utf-8")
            except: pass
        else:
            with self._lock:
                self._pending = [f for f in self._pending if not f.done()]
                self._pending.append(self._executor.submit(self._sync_gist, s))
    def _sync_gist(self, s):
        try:
            h = {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}
            self.http.request("PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=h, json={"files": {self.cfg.gist_state_file: {"content": s}}})
        except: pass

class TelegramNotifier:
    def __init__(self, http, dry=False):
        self.token, self.chat, self.http, self.dry = os.getenv("TELEGRAM_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip(), http, dry
    def send(self, msg):
        if self.dry or not (self.token and self.chat): 
            logging.info(f"[DRY-RUN TG]\n{msg}"); return True
        try: 
            self.http.request("POST", f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat, "text": msg, "parse_mode": "HTML"})
            return True
        except: return False

class MarketData:
    def __init__(self, http): self.http = http
    def fetch(self) -> pd.DataFrame:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key: raise Exception("TWELVE_DATA_API_KEY missing")
        try:
            r = self.http.request("GET", "https://api.twelvedata.com/time_series", 
                params={"symbol": "XAU/USD", "interval": "5min", "outputsize": 800, "apikey": key, "timezone": "UTC"})
            d = r.json()
            if "values" not in d: raise Exception("No data")
            df = pd.DataFrame(d["values"]); df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna().sort_values("datetime").reset_index(drop=True)
        except Exception as e:
            raise Exception(f"Data fetch error: {redact(e)}")

# ═══════════════════ SMART MONEY CONCEPTS (SMC) ═══════════════════
class SMCStrategy:
    def __init__(self, cfg): self.cfg = cfg
    
    def _get_asian_range(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Mendapatkan High/Low dari Sesi Asia (00:00 - 07:00 UTC)"""
        asian_bars = df[(df["datetime"].dt.hour >= 0) & (df["datetime"].dt.hour < 7)]
        if asian_bars.empty: return 0.0, 0.0
        return float(asian_bars["high"].max()), float(asian_bars["low"].min())

    def _is_fvg(self, df: pd.DataFrame, idx: int, is_bullish: bool) -> bool:
        """Fair Value Gap (FVG) 3-candle pattern"""
        if idx < 2: return False
        c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
        if is_bullish:
            return c3["low"] > c1["high"] # Gap up
        else:
            return c3["high"] < c1["low"] # Gap down

    def analyze(self, df: pd.DataFrame) -> dict:
        if len(df) < 50: return {"signal": "NEUTRAL"}
        
        now = df["datetime"].iloc[-1]
        
        # 1. Filter Waktu: Hanya London (07-10) & NY (13-16) UTC
        lh = now.astimezone(LONDON_TZ).hour
        nh = now.astimezone(NY_TZ).hour
        if not ((7 <= lh < 10) or (13 <= nh < 16)):
            return {"signal": "NEUTRAL"}

        asian_high, asian_low = self._get_asian_range(df)
        if asian_high == 0 or asian_low == 0: return {"signal": "NEUTRAL"}

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # 2. Liquidity Sweep & Rejection (SELL)
        # Harga menyapu Asian High tapi ditutup kembali di bawahnya
        if curr["high"] > asian_high and curr["close"] < asian_high and prev["high"] <= asian_high:
            # Cari FVG Bearish
            if self._is_fvg(df, len(df)-1, is_bullish=False) or self._is_fvg(df, len(df)-2, is_bullish=False):
                sl = curr["high"] + 0.5 # SL di atas sumbu sweep
                entry = curr["close"]
                tp = asian_low # Target di likuiditas seberang (Asian Low)
                if (entry - tp) > (sl - entry) * 1.5: # Pastikan minimal 1:1.5 RR
                    return {"signal": "SELL", "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2)}

        # 3. Liquidity Sweep & Rejection (BUY)
        if curr["low"] < asian_low and curr["close"] > asian_low and prev["low"] >= asian_low:
            if self._is_fvg(df, len(df)-1, is_bullish=True) or self._is_fvg(df, len(df)-2, is_bullish=True):
                sl = curr["low"] - 0.5
                entry = curr["close"]
                tp = asian_high
                if (tp - entry) > (entry - sl) * 1.5:
                    return {"signal": "BUY", "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2)}

        return {"signal": "NEUTRAL"}

class RiskManager:
    def __init__(self, cfg, kv): self.cfg, self.kv = cfg, kv
    def calc_lot(self, balance, entry, sl):
        dist = abs(entry - sl)
        if dist < 0.1: return 0.01
        risk_usd = balance * (self.cfg.risk_pct / 100)
        lot = risk_usd / (dist * 100.0)
        return max(0.01, min(5.0, round(lot, 2)))
    
    def check_drawdown(self, balance):
        state = self.kv.get("circuit_breaker", {})
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if state.get("date") != today:
            state = {"date": today, "start_bal": balance, "locked": False}
            self.kv.set("circuit_breaker", state); return True
        dd = (state["start_bal"] - balance) / state["start_bal"] * 100
        return dd < self.cfg.max_daily_loss_pct

class OutcomeTracker:
    def __init__(self, cfg, kv): self.cfg, self.kv = cfg, kv
    def update(self, df):
        journal = self.kv.get("journal", []) or []
        events = []
        for e in journal:
            if e["outcome"] != "PENDING": continue
            bars = df[df["datetime"] > pd.to_datetime(e["time"], utc=True)]
            if bars.empty: continue
            
            for _, b in bars.iterrows():
                if e["signal"] == "BUY":
                    if b["low"] <= e["sl"]: e["outcome"] = "LOSS"; events.append(("CLOSE", e)); break
                    if b["high"] >= e["tp"]: e["outcome"] = "WIN"; events.append(("CLOSE", e)); break
                else:
                    if b["high"] >= e["sl"]: e["outcome"] = "LOSS"; events.append(("CLOSE", e)); break
                    if b["low"] <= e["tp"]: e["outcome"] = "WIN"; events.append(("CLOSE", e)); break
        self.kv.set("journal", journal); self.kv.save()
        return events

class ApexBot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.http = ResilientHttpClient()
        self.kv = KVStore(cfg, self.http)
        self.data = MarketData(self.http)
        self.strategy = SMCStrategy(cfg)
        self.risk = RiskManager(cfg, self.kv)
        self.outcome = OutcomeTracker(cfg, self.kv)
        self.tg = TelegramNotifier(self.http, cfg.dry_run)

    def run(self):
        self.kv.force_load()
        now = datetime.now(UTC)
        
        if now.weekday() >= 5: return
        bal = self.cfg.account_balance
        if not self.risk.check_drawdown(bal): return

        try: df = self.data.fetch()
        except Exception as e: 
            logging.error(e); return
            
        journal = self.kv.get("journal", []) or []
        if any(j["outcome"] == "PENDING" for j in journal): 
            events = self.outcome.update(df)
            for evt, tr in events:
                icon = "✅" if tr["outcome"] == "WIN" else "❌"
                self.tg.send(f"{icon} <b>Closed</b> {tr['signal']} | {tr['outcome']}")
            return

        res = self.strategy.analyze(df)
        if res["signal"] == "NEUTRAL": return
        
        lot = self.risk.calc_lot(bal, res["entry"], res["sl"])
        msg = (f"🏛️ <b>APEX v{VERSION} — {res['signal']}</b>\n\n"
               f"Symbol: <b>{self.cfg.symbol}</b>\n"
               f"Entry: <code>{res['entry']}</code>\n"
               f"SL: <code>{res['sl']}</code>\n"
               f"TP: <code>{res['tp']}</code>\n\n"
               f"📦 Lot: <code>{lot}</code>\n"
               f"🧠 Logic: Liquidity Sweep + FVG")
               
        if self.tg.send(msg):
            journal.append({
                "id": hashlib.md5(f"{now}_{res['signal']}".encode()).hexdigest()[:8],
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "signal": res["signal"], "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
                "lot": lot, "outcome": "PENDING"
            })
            self.kv.set("journal", journal); self.kv.save()

class Backtester:
    def __init__(self, cfg): self.cfg = cfg; self.strategy = SMCStrategy(cfg)
    def run(self, df, max_bars=None):
        bal = self.cfg.account_balance
        trades = []; active = None
        end = len(df) if not max_bars else min(len(df), max_bars)
        
        # Simulasi per bar
        for i in range(50, end):
            hist = df.iloc[:i].reset_index(drop=True)
            if active:
                b = df.iloc[i]
                if active["signal"] == "BUY":
                    if b["low"] <= active["sl"]: active["outcome"] = "LOSS"; bal -= active["risk"]; trades.append(active); active=None
                    elif b["high"] >= active["tp"]: active["outcome"] = "WIN"; bal += active["reward"]; trades.append(active); active=None
                else:
                    if b["high"] >= active["sl"]: active["outcome"] = "LOSS"; bal -= active["risk"]; trades.append(active); active=None
                    elif b["low"] <= active["tp"]: active["outcome"] = "WIN"; bal += active["reward"]; trades.append(active); active=None
                continue
            
            res = self.strategy.analyze(hist)
            if res["signal"] != "NEUTRAL":
                risk_usd = bal * (self.cfg.risk_pct / 100)
                dist_sl = abs(res["entry"] - res["sl"])
                dist_tp = abs(res["entry"] - res["tp"])
                lot = risk_usd / (dist_sl * 100.0)
                active = {
                    "signal": res["signal"], "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
                    "risk": risk_usd, "reward": lot * dist_tp * 100.0
                }
        if not trades: return {"msg": "No trades"}
        w = sum(1 for t in trades if t["outcome"] == "WIN")
        l = sum(1 for t in trades if t["outcome"] == "LOSS")
        return {
            "trades": len(trades), "wins": w, "losses": l,
            "win_rate": round(w/len(trades)*100, 2),
            "final_balance": round(bal, 2),
            "return_pct": round((bal/self.cfg.account_balance - 1)*100, 2)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run", choices=["run", "backtest"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv")
    parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = Config.from_env()
    if args.dry_run: cfg.dry_run = True
    
    if args.command == "backtest":
        if not args.csv: print("❌ --csv required"); sys.exit(1)
        bt = Backtester(cfg)
        df = pd.read_csv(args.csv)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c])
        print(json.dumps(bt.run(df, args.max_bars), indent=2))
    else:
        bot = ApexBot(cfg)
        bot.run()
