"""4-tier institutional XAUUSD market data handler with strict OHLC validation and zero double-offset."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import UTC, Config
from .exceptions import DataFetchError
from .news_filter import parse_iso_utc
from .security import redact


class MarketDataHandler:
    """Acquires, cleans, calibrates, and validates multi-timeframe XAUUSD bars across 4 tiers."""

    def __init__(self, cfg: Config, http: Any) -> None:
        self.cfg = cfg
        self.http = http
        self._last_td_call = 0.0

    def _oanda_base_url(self) -> str:
        if self.cfg.oanda_env == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    def compute_broker_offset(self) -> float:
        """Calculates quote spread difference between OANDA broker feed and TwelveData spot."""
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        oanda_acct = (self.cfg.oanda_account_id or os.getenv("OANDA_ACCOUNT_ID", "")).strip()
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda_key and oanda_acct and td_key):
            return self.cfg.mt5_offset

        try:
            r1 = self.http.request(
                "GET",
                f"{self._oanda_base_url()}/v3/accounts/{oanda_acct}/pricing",
                params={"instruments": "XAU_USD"},
                headers={"Authorization": f"Bearer {oanda_key}"},
            )
            prices = r1.json().get("prices", [])
            if not prices:
                return self.cfg.mt5_offset
            oanda_bid = float(prices[0]["bids"][0]["price"])

            r2 = self.http.request(
                "GET",
                "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key},
            )
            td_price = float(r2.json()["price"])
            if td_price <= 0:
                return self.cfg.mt5_offset

            offset = oanda_bid - td_price
            if abs(offset) > self.cfg.max_broker_offset:
                logging.warning(
                    f"Offset {offset:+.3f} rejected (exceeds ±${self.cfg.max_broker_offset:.2f} bound)"
                )
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
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicates, sorts chronologically, and repairs minor high/low envelope discrepancies."""
        out = df.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "volume" not in out.columns:
            out["volume"] = 100.0
        else:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(100.0)
        out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
        out = out.drop_duplicates(subset=["datetime"], keep="last")
        out = out.sort_values("datetime").reset_index(drop=True)
        # Enforce strict candle envelope consistency
        out["high"] = np.maximum(out["high"], np.maximum(out["open"], out["close"]))
        out["low"] = np.minimum(out["low"], np.minimum(out["open"], out["close"]))
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resamples already-calibrated OHLCV bars without lookahead bias or double-offset."""
        resampled = (
            df.set_index("datetime")
            .resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    # Tier 1: OANDA v20 Institutional Spot Candles (when OANDA_API_KEY is configured)
    def _fetch_oanda(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        key = os.getenv("OANDA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("OANDA_API_KEY not configured")

        base = self._oanda_base_url()

        def fetch_gran(granularity: str, count: int) -> pd.DataFrame:
            r = self.http.request(
                "GET",
                f"{base}/v3/instruments/XAU_USD/candles",
                params={"granularity": granularity, "count": count, "price": "M"},
                headers={"Authorization": f"Bearer {key}"},
            )
            candles = r.json().get("candles", [])
            if not candles:
                raise DataFetchError(f"OANDA {granularity}: empty candles")
            rows: List[dict] = []
            for c in candles:
                mid = c.get("mid")
                if not mid:
                    continue
                rows.append(
                    {
                        "datetime": parse_iso_utc(c["time"]),
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": float(c.get("volume", 100.0)),
                    }
                )
            df = self._clean_ohlcv(pd.DataFrame(rows))
            return self._apply_offset(df, self.cfg.mt5_offset)

        m5 = fetch_gran("M5", min(self.cfg.m5_fetch_size, 1000))
        m15 = fetch_gran("M15", 500)
        h1 = fetch_gran("H1", 800)
        h4 = fetch_gran("H4", 500)

        self._validate(m5, "OANDA-M5", self.cfg.min_bars_m5)
        self._validate(m15, "OANDA-M15", self.cfg.min_bars_m15)
        self._validate(h1, "OANDA-H1", self.cfg.min_bars_h1)
        self._validate(h4, "OANDA-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "OANDA-M5")

        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "OANDA"

    # Tier 2: TwelveData Spot XAU/USD
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
                raise DataFetchError(
                    f"TwelveData {interval}: {data.get('message', 'No values returned')}"
                )
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df["volume"] = 100.0
            df = self._clean_ohlcv(df)
            return self._apply_offset(df, offset)

        m5 = fetch_interval("5min", min(self.cfg.m5_fetch_size, 800))
        h1 = fetch_interval("1h", 800)
        h4 = fetch_interval("4h", 500)
        # m5 already has offset applied; DO NOT apply offset a second time!
        m15 = self._resample(m5, "15min")

        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")

        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def _fetch_yahoo_symbol(
        self, yf_symbol: str, label: str, offset: float
    ) -> Dict[str, pd.DataFrame]:
        def fetch_yf(interval: str, rng: str) -> pd.DataFrame:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
            r = self.http.request("GET", url, params={"interval": interval, "range": rng})
            data = r.json()
            chart_res = data.get("chart", {}).get("result")
            if not chart_res:
                raise DataFetchError(f"{label} {interval}: empty chart result")
            result = chart_res[0]
            timestamps = result.get("timestamp") or []
            quote = (result.get("indicators", {}).get("quote") or [{}])[0]
            if not timestamps or not quote:
                raise DataFetchError(f"{label} {interval}: missing timestamps or OHLC")

            vol_raw = quote.get("volume")
            if not isinstance(vol_raw, list) or len(vol_raw) != len(timestamps):
                vol_raw = [100.0] * len(timestamps)

            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(timestamps, unit="s", utc=True),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "volume": vol_raw,
                }
            )
            df = self._clean_ohlcv(df)
            return self._apply_offset(df, offset)

        m5 = fetch_yf("5m", "14d")
        h1 = fetch_yf("1h", "90d")
        # Resample from calibrated m5 / h1 without re-applying offset!
        m15 = self._resample(m5, "15min")
        h4 = self._resample(h1, "4h")

        self._validate(m5, f"{label}-M5", self.cfg.min_bars_m5)
        self._validate(m15, f"{label}-M15", self.cfg.min_bars_m15)
        self._validate(h1, f"{label}-H1", self.cfg.min_bars_h1)
        self._validate(h4, f"{label}-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, f"{label}-M5")

        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}

    # Tier 3: Yahoo Finance Spot FX Gold (XAUUSD=X)
    def _fetch_yahoo_spot(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        dfs = self._fetch_yahoo_symbol("XAUUSD=X", "YF-Spot", offset)
        return dfs, "YahooSpot"

    # Tier 4: Yahoo Finance COMEX Gold Futures (GC=F) Emergency Fallback
    def _fetch_yahoo_futures(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        logging.warning(
            "Using GC=F Futures emergency fallback — beware of Futures/Spot basis spread!"
        )
        dfs = self._fetch_yahoo_symbol("GC=F", "YF-Futures", offset)
        return dfs, "YahooFutures(GC=F)"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        """Executes cascading failover across 4 verified market data providers."""
        offset = self.compute_broker_offset()
        tiers = [
            ("OANDA", self._fetch_oanda),
            ("TwelveData", self._fetch_twelvedata),
            ("YahooSpot", self._fetch_yahoo_spot),
            ("YahooFutures", self._fetch_yahoo_futures),
        ]
        errors: List[str] = []
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
            vals = df[c].values
            if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
                raise DataFetchError(f"{label} contains zero, negative, or non-finite values in {c}")
        if (df["high"] < df["low"]).any():
            raise DataFetchError(f"{label} contains inverted high < low candles")

    def _check_staleness(self, df: pd.DataFrame, interval_min: int, label: str) -> None:
        now = datetime.now(UTC)
        # Gold market closes Friday ~21:55 UTC and opens Sunday ~22:00 UTC
        if now.weekday() == 5 or (now.weekday() == 4 and now.hour >= 22) or (now.weekday() == 6 and now.hour < 22):
            return
        last_dt = df["datetime"].iloc[-1]
        if hasattr(last_dt, "to_pydatetime"):
            last_dt = last_dt.to_pydatetime()
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
        age_sec = (now - last_dt).total_seconds()
        max_sec = interval_min * 60 * self.cfg.max_stale_mult
        if age_sec > max_sec:
            raise DataFetchError(
                f"{label} data is stale: age {age_sec / 60:.1f}m > limit {max_sec / 60:.1f}m"
            )
