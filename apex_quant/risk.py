"""Dynamic position sizing, hard leverage caps, and deadlock-free circuit breakers."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

from .config import UTC, Config
from .news_filter import parse_iso_utc
from .storage import KVStore


class RiskManagementEngine:
    """Calculates exact Gold position sizing and enforces daily/streak drawdown protection."""

    def __init__(self, cfg: Config, kv: KVStore) -> None:
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        """Compounds equity cleanly from archived baseline + chronological closed trades."""
        arch = self.kv.get("archived_stats", {}) or {}
        base = float(arch.get("compounded_equity", self.cfg.account_balance))

        journal = self.kv.get("journal", []) or []
        closed_trades = [
            j
            for j in journal
            if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")
            and j.get("r_achieved") is not None
        ]
        closed_trades.sort(key=lambda x: (x.get("close_timestamp") or x.get("timestamp", "")))

        for t in closed_trades:
            try:
                r_multiple = float(t["r_achieved"])
                risk_pct = float(t.get("risk_pct", 1.0))
                base += r_multiple * (base * (risk_pct / 100.0))
                base = max(10.0, base)
            except (ValueError, TypeError):
                continue

        return max(10.0, round(base, 2))

    def compute_lot(
        self,
        balance: float,
        entry: float,
        sl: float,
        confidence_pct: float,
        verdict_multiplier: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculates exact Gold lot size AFTER verdict scaling so returned risk_pct is 100% accurate.

        Formula:
          Risk Money ($) = Balance * (Risk % / 100) * verdict_multiplier
          SL Distance ($/oz) = |Entry - SL|
          Contract Size = 100 Troy Oz (Standard Gold)
          Lot = Risk Money / (SL Distance * Contract Size)
        """
        if balance <= 0:
            raise ValueError("Balance must be positive")
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-4:
            raise ValueError("Stop loss distance too small for Gold position sizing")
        if entry <= 0:
            raise ValueError("Entry price must be positive")

        conf_factor = max(0.4, min(1.0, confidence_pct / 100.0))
        risk_pct = max(0.1, self.cfg.max_risk_pct_per_trade * conf_factor) * max(0.0, verdict_multiplier)
        risk_dollars = balance * (risk_pct / 100.0)

        raw_lot = risk_dollars / (sl_dist * self.cfg.contract_size)
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(raw_lot, 2)))

        # Hard Leverage Cap Enforcement (e.g. 30:1 max notional / equity)
        notional_value = lot * self.cfg.contract_size * entry
        if (notional_value / balance) > self.cfg.hard_leverage_cap:
            max_permitted_lot = (self.cfg.hard_leverage_cap * balance) / (
                self.cfg.contract_size * entry
            )
            lot = max(self.cfg.min_lot, round(max_permitted_lot, 2))

        actual_risk_pct = (lot * self.cfg.contract_size * sl_dist) / balance * 100.0
        return lot, round(actual_risk_pct, 2)

    def check_circuit_breakers(
        self, current_balance: float, now: datetime | None = None
    ) -> Tuple[bool, str]:
        """Protects capital without creating an infinite lockout after cooldown expires."""
        now = now or datetime.now(UTC)
        today_str = now.strftime("%Y-%m-%d")
        cb = self.kv.get("circuit_breaker", {}) or {}

        # 1. Active Lock Check
        locked_until = cb.get("locked_until")
        if locked_until:
            lock_dt = parse_iso_utc(str(locked_until))
            if now < lock_dt:
                return False, f"CIRCUIT BREAKER ENGAGED: Locked until {locked_until}"
            # Lock has expired cleanly -> clear lock & mark reset anchor so same losses don't re-lock infinitely
            cb["locked_until"] = None
            if cb.get("lock_reason") == "CONSEC_LOSSES":
                cb["consec_loss_reset_ts"] = cb.get("last_loss_ts") or now.strftime("%Y-%m-%d %H:%M:%S")
            cb["lock_reason"] = None
            self.kv.set("circuit_breaker", cb)

        # 2. Daily Drawdown Limit Protection
        day_start_balance = cb.get("day_start_balance")
        last_date = cb.get("last_date", "")
        if last_date != today_str or day_start_balance is None:
            cb["last_date"] = today_str
            cb["day_start_balance"] = current_balance
            cb["daily_signals_count"] = 0
            self.kv.set("circuit_breaker", cb)
            day_start_balance = current_balance

        if day_start_balance > 0:
            daily_drawdown = (day_start_balance - current_balance) / day_start_balance * 100.0
            if daily_drawdown >= self.cfg.max_daily_loss_pct:
                # Lock until next UTC day or dd_pause_hours, whichever is appropriate
                lock_time = (now + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
                cb["locked_until"] = lock_time
                cb["lock_reason"] = "DAILY_DD"
                self.kv.set("circuit_breaker", cb)
                self.kv.save()
                return (
                    False,
                    f"MAX DAILY DRAWDOWN BREACH: {daily_drawdown:.2f}% >= "
                    f"{self.cfg.max_daily_loss_pct}%. Locked for {self.cfg.dd_pause_hours}h.",
                )

        # 3. Consecutive Losses Circuit Breaker (only counts losses since last streak reset)
        journal = self.kv.get("journal", []) or []
        reset_ts = cb.get("consec_loss_reset_ts", "")
        closed = [
            j
            for j in journal
            if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")
            and (not reset_ts or (j.get("close_timestamp") or j.get("timestamp", "")) > reset_ts)
        ]
        closed.sort(key=lambda x: (x.get("close_timestamp") or x.get("timestamp", "")))

        consec_losses = 0
        last_loss_ts = ""
        for trade in reversed(closed):
            if trade.get("outcome") == "LOSS":
                consec_losses += 1
                if not last_loss_ts:
                    last_loss_ts = trade.get("close_timestamp") or trade.get("timestamp", "")
            else:
                break

        if consec_losses >= self.cfg.max_consecutive_losses:
            lock_time = (now + timedelta(hours=self.cfg.dd_pause_hours)).isoformat()
            cb["locked_until"] = lock_time
            cb["lock_reason"] = "CONSEC_LOSSES"
            cb["last_loss_ts"] = last_loss_ts or now.strftime("%Y-%m-%d %H:%M:%S")
            self.kv.set("circuit_breaker", cb)
            self.kv.save()
            return (
                False,
                f"CONSECUTIVE LOSSES BREACH: {consec_losses} losses in a row. "
                f"Trading halted for {self.cfg.dd_pause_hours}h.",
            )

        # 4. Daily Signal Quota
        sig_count = int(cb.get("daily_signals_count", 0))
        if sig_count >= self.cfg.max_daily_signals:
            return (
                False,
                f"DAILY SIGNAL CAP REACHED: {sig_count}/{self.cfg.max_daily_signals} signals today.",
            )

        return True, "All circuit breakers clear."

    def reset_locks(self, now: datetime | None = None) -> None:
        """Manually clears active circuit breaker locks while advancing streak reset anchor."""
        now = now or datetime.now(UTC)
        cb = self.kv.get("circuit_breaker", {}) or {}
        cb["locked_until"] = None
        cb["lock_reason"] = None
        cb["consec_loss_reset_ts"] = now.strftime("%Y-%m-%d %H:%M:%S")
        self.kv.set("circuit_breaker", cb)
        self.kv.save()
