"""Zero-lookahead walk-forward backtester and quantitative journal performance analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import UTC, Config
from .exceptions import StateError
from .risk import RiskManagementEngine
from .signals import SignalGenerator
from .storage import KVStore
from .tracker import OutcomeTracker


@dataclass
class PerformanceMetrics:
    total_signals: int
    filled_trades: int
    wins: int
    losses: int
    stale_or_invalidated: int
    win_rate_pct: float
    profit_factor: float
    expectancy_r: float
    net_r: float
    sharpe_per_trade: float
    sortino_per_trade: float
    max_drawdown_pct: float
    initial_balance: float
    final_balance: float
    total_return_pct: float


def compute_journal_metrics(
    journal: List[Dict[str, Any]],
    initial_balance: float,
    archived_stats: Optional[Dict[str, Any]] = None,
) -> PerformanceMetrics:
    """Computes institutional performance ratios from closed journal records."""
    closed = [
        j
        for j in journal
        if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS") and j.get("r_achieved") is not None
    ]
    closed.sort(key=lambda x: (x.get("close_timestamp") or x.get("timestamp", "")))

    r_list = [float(t["r_achieved"]) for t in closed]
    wins = sum(1 for t in closed if str(t.get("outcome", "")).startswith("WIN"))
    losses = sum(1 for t in closed if t.get("outcome") == "LOSS")
    stale_inv = sum(1 for t in journal if t.get("outcome") in ("STALE", "INVALIDATED"))

    gross_win_r = sum(r for r in r_list if r > 0)
    gross_loss_r = abs(sum(r for r in r_list if r < 0))
    profit_factor = (
        round(gross_win_r / gross_loss_r, 3)
        if gross_loss_r > 1e-9
        else (float("inf") if gross_win_r > 0 else 0.0)
    )
    expectancy_r = round(float(np.mean(r_list)), 4) if r_list else 0.0
    net_r = round(sum(r_list), 4)
    win_rate = round((wins / len(closed)) * 100.0, 2) if closed else 0.0

    # Equity curve & Max Drawdown
    equity = float((archived_stats or {}).get("compounded_equity", initial_balance))
    start_eq = equity
    peak = equity
    max_dd_pct = 0.0
    pct_returns: List[float] = []

    for t in closed:
        r_mult = float(t["r_achieved"])
        risk_pct = float(t.get("risk_pct", 1.0))
        ret_frac = r_mult * (risk_pct / 100.0)
        pct_returns.append(ret_frac)
        equity = max(10.0, equity * (1.0 + ret_frac))
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd)

    sharpe = 0.0
    sortino = 0.0
    if len(pct_returns) >= 2:
        arr = np.array(pct_returns, dtype=float)
        std = float(np.std(arr, ddof=1))
        if std > 1e-12:
            sharpe = round(float(np.mean(arr) / std * math.sqrt(len(arr))), 3)
        downside = arr[arr < 0]
        if len(downside) >= 1:
            d_std = float(np.sqrt(np.mean(downside ** 2)))
            if d_std > 1e-12:
                sortino = round(float(np.mean(arr) / d_std * math.sqrt(len(arr))), 3)

    total_ret_pct = round(((equity - start_eq) / start_eq) * 100.0, 2) if start_eq > 0 else 0.0

    return PerformanceMetrics(
        total_signals=len(journal),
        filled_trades=len(closed),
        wins=wins,
        losses=losses,
        stale_or_invalidated=stale_inv,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        expectancy_r=expectancy_r,
        net_r=net_r,
        sharpe_per_trade=sharpe,
        sortino_per_trade=sortino,
        max_drawdown_pct=round(max_dd_pct, 2),
        initial_balance=round(start_eq, 2),
        final_balance=round(equity, 2),
        total_return_pct=total_ret_pct,
    )


def generate_synthetic_xauusd_m5(bars: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generates realistic calibrated XAUUSD 5-minute OHLCV regime-switching data for stress tests."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    times = [start + timedelta(minutes=5 * i) for i in range(bars)]

    price = 2650.0
    rows = []
    drift = 0.0
    for i, ts in enumerate(times):
        if i % 120 == 0:
            drift = rng.choice([-0.18, 0.0, 0.18])
        vol = 1.35 + 0.6 * abs(math.sin(i / 48.0))
        ret = drift + rng.normal(0, vol)
        open_p = price
        close_p = max(1000.0, open_p + ret)
        wick_up = abs(rng.normal(0, vol * 0.65))
        wick_dn = abs(rng.normal(0, vol * 0.65))
        high_p = max(open_p, close_p) + wick_up
        low_p = min(open_p, close_p) - wick_dn
        rows.append(
            {
                "datetime": ts,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": float(rng.integers(80, 450)),
            }
        )
        price = close_p

    return pd.DataFrame(rows)


class WalkForwardBacktester:
    """Event-driven zero-lookahead backtester reusing live Signal, Risk, and OutcomeTracker."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
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
            .dropna()
            .reset_index()
        )

    def run(self, m5_df: pd.DataFrame, step_bars: int = 3) -> PerformanceMetrics:
        """Walks forward bar-by-bar over M5 data with zero future data visibility."""
        if len(m5_df) < 600:
            raise StateError("Backtest requires at least 600 M5 bars for H4/H1 warmup")

        # Isolated in-memory KVStore (never touches disk or Gist)
        mem_kv = KVStore(self.cfg, http=None)
        risk = RiskManagementEngine(self.cfg, mem_kv)
        sig_gen = SignalGenerator(self.cfg)
        tracker = OutcomeTracker(self.cfg, mem_kv)

        m5_sorted = m5_df.sort_values("datetime").reset_index(drop=True)
        warmup = 450

        for idx in range(warmup, len(m5_sorted) - 1, step_bars):
            m5_slice = m5_sorted.iloc[: idx + 1].copy()
            now_dt = m5_slice["datetime"].iloc[-1].to_pydatetime()
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=UTC)

            # Update pending/open orders strictly up to current closed bar
            tracker.update(m5_slice, now=now_dt)

            balance = risk.dynamic_balance()
            ok, _ = risk.check_circuit_breakers(balance, now=now_dt)
            if not ok:
                continue

            journal = mem_kv.get("journal", []) or []
            if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
                continue
            unfilled = sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled"))
            if unfilled >= self.cfg.max_unfilled_limits:
                continue

            m15_slice = self._resample(m5_slice, "15min")
            h1_slice = self._resample(m5_slice, "1h")
            h4_slice = self._resample(m5_slice, "4h")
            if len(h1_slice) < 30:
                continue

            res = sig_gen.generate(m5_slice, m15_slice, h1_slice, h4_slice)
            if res.direction == "NEUTRAL" or res.verdict == "CANCEL":
                continue

            ob = sig_gen.detect_order_block(m5_slice, res.direction, atr=res.atr)
            try:
                entry, sl, tp1, tp2, tp3 = sig_gen.compute_levels(m5_slice, res.direction, ob)
            except ValueError:
                continue

            verdict_mult = {"EXECUTE_FULL": 1.0, "EXECUTE_HALF": 0.5, "EXECUTE_MICRO": 0.25}.get(
                res.verdict, 1.0
            )
            conf_pct = max(res.buy_ratio, res.sell_ratio) * 100.0
            lot, actual_risk = risk.compute_lot(
                balance, entry, sl, conf_pct, verdict_multiplier=verdict_mult
            )

            anchor_dt = now_dt + timedelta(minutes=5)
            journal.append(
                {
                    "id": f"bt_{idx}",
                    "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "anchor_ts": anchor_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": self.cfg.symbol,
                    "signal": res.direction,
                    "verdict": res.verdict,
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
            )
            mem_kv.set("journal", journal)
            cb = mem_kv.get("circuit_breaker", {}) or {}
            cb["daily_signals_count"] = int(cb.get("daily_signals_count", 0)) + 1
            mem_kv.set("circuit_breaker", cb)

        return compute_journal_metrics(
            mem_kv.get("journal", []) or [],
            self.cfg.account_balance,
            mem_kv.get("archived_stats"),
        )
