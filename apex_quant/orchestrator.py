"""Master trading bot workflow orchestrator with bar idempotency, process lock, and loop mode."""

from __future__ import annotations

import hashlib
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Optional

from .config import LONDON_TZ, NY_TZ, UTC, VERSION, Config
from .exceptions import ExecutionError, StateError
from .execution import (
    ExecutionAdapter,
    OandaExecutionAdapter,
    OrderRequest,
    SignalOnlyAdapter,
)
from .http_client import ResilientHttpClient
from .market_data import MarketDataHandler
from .news_filter import NewsSafetyFilter
from .notifier import TelegramNotifier
from .risk import RiskManagementEngine
from .security import redact
from .signals import SignalGenerator
from .storage import KVStore
from .tracker import OutcomeTracker


class TradingBotOrchestrator:
    """Master workflow controller coordinating data, signals, risk, execution, and alerts."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.http = ResilientHttpClient(
            timeout=cfg.http_timeout_sec,
            max_retries=cfg.http_max_retries,
            backoff=cfg.http_backoff_factor,
            max_retry_after_sec=cfg.http_max_retry_after_sec,
        )
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.http, kv=self.kv)
        self.tg = TelegramNotifier(self.http, dry_run=cfg.dry_run)
        self.outcome = OutcomeTracker(cfg, self.kv)
        self.executor: ExecutionAdapter = (
            OandaExecutionAdapter(cfg, self.http)
            if cfg.execution_mode == "oanda"
            else SignalOnlyAdapter()
        )
        self._stop_requested = False

    def close(self) -> None:
        self.http.close()

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

    def run(self, now: Optional[datetime] = None) -> None:
        with self.kv.process_lock():
            self._run_locked(now=now)

    def _run_locked(self, now: Optional[datetime] = None) -> None:
        try:
            self.kv.force_load()
        except StateError as e:
            logging.error(f"Persistence state load error: {e}")
            return

        now = now or datetime.now(UTC)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.cfg.kill_switch:
            logging.info("KILL SWITCH ENGAGED — Execution halted.")
            return

        # Gold market closes Friday ~21:55 UTC and reopens Sunday ~22:00 UTC
        if now.weekday() == 5 or (now.weekday() == 4 and now.hour >= 22) or (now.weekday() == 6 and now.hour < 22):
            logging.info("Weekend market closed — Skipping iteration.")
            return

        in_killzone = self._in_killzone(now)
        journal_pre = self.kv.get("journal", []) or []
        has_pending_or_open = any(j.get("outcome") == "PENDING" for j in journal_pre)

        if not in_killzone and not has_pending_or_open:
            logging.info(
                "Outside institutional killzone (07:00-10:00 London, 08:00-11:00 & 13:00-16:00 NY) "
                "— Skipping iteration (set APEX_SKIP_KILLZONE_FILTER=true to override)."
            )
            return

        balance = self.risk.dynamic_balance()

        # Step 1: Evaluate Circuit Breakers (skip only if inside killzone; still track pending trades)
        if in_killzone:
            ok, reason = self.risk.check_circuit_breakers(balance, now=now)
            if not ok:
                logging.info(f"Circuit Breaker Triggered: {reason}")
                if not has_pending_or_open:
                    return
                allow_new_signals = False
            else:
                allow_new_signals = True
        else:
            logging.info(
                "Outside institutional killzone — Monitoring active/pending trades only (no new entries)."
            )
            allow_new_signals = False

        # Step 2: High-Impact News & Rollover Blackout Filter (blocks new entries, still tracks open trades)
        if allow_new_signals:
            in_blackout, news_title = self.news.is_blackout(now=now)
            if in_blackout:
                logging.info(f"Safety Blackout Active: [{news_title}] — Halting new signals.")
                self.kv.save()
                if not has_pending_or_open:
                    return
                allow_new_signals = False

        # Step 3: Fetch Market Data with 4-Tier Failover
        try:
            dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            err = redact(e)
            logging.error(f"Fatal Market Data Failure: {err}")
            self.tg.send(
                f"⚠️ <b>Market Data Warning</b>\n<code>{TelegramNotifier.escape(err[:220])}</code>"
            )
            return

        # Step 4: Update Trade Outcomes & Realized R
        try:
            events = self.outcome.update(dfs["5min"], now=now)
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

        if not allow_new_signals:
            return

        balance = self.risk.dynamic_balance()

        # Slice latest closed bar (strictly eliminate unclosed candle repainting)
        m5_closed = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15_closed = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1_closed = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4_closed = dfs["4h"].iloc[:-1].reset_index(drop=True)

        if len(m5_closed) < self.cfg.min_bars_m5 or len(h1_closed) < self.cfg.min_bars_h1:
            logging.warning("Insufficient bars post-closed trim")
            return

        last_price = float(m5_closed["close"].iloc[-1])
        signal_bar_dt = m5_closed["datetime"].iloc[-1].to_pydatetime()
        if signal_bar_dt.tzinfo is None:
            signal_bar_dt = signal_bar_dt.replace(tzinfo=UTC)
        signal_bar_iso = signal_bar_dt.strftime("%Y-%m-%d %H:%M:%S")
        anchor_ts_iso = (signal_bar_dt + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

        # Bar-level Idempotency Guard
        state = self.kv.get("state", {}) or {}
        if state.get("last_processed_bar_iso") == signal_bar_iso:
            logging.info(f"Bar {signal_bar_iso} UTC already processed — Idempotent skip.")
            return

        # Step 5: Check Active Position & Pending Limits
        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("Existing active position already open — Skipping new signal.")
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        unfilled_count = sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled"))
        if unfilled_count >= self.cfg.max_unfilled_limits:
            logging.info(f"Max unfilled pending limits ({unfilled_count}) reached — Skipping.")
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        # Step 6: Generate Quantitative Parliament Signal
        result = self.signal.generate(m5_closed, m15_closed, h1_closed, h4_closed)
        logging.info(
            f"Parliament Consensus: {result.direction} | "
            f"BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%} | "
            f"Verdict={result.verdict} | Active={result.active_models}/{result.total_models} | "
            f"Skip={result.skip_reason}"
        )

        if result.direction == "NEUTRAL" or result.verdict == "CANCEL":
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        # Step 7: Directional Cooldown Check
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
                    logging.info(f"Signal cooldown ({cool_window} min) active — Skipping.")
                    self._update_state(last_price, now_str, signal_bar_iso)
                    return
            except Exception:
                pass

        # Step 8: Detect Unmitigated Order Block & Compute Levels
        ob_zone = self.signal.detect_order_block(m5_closed, result.direction, atr=result.atr)
        try:
            entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5_closed, result.direction, ob_zone)
        except ValueError as e:
            logging.warning(f"Level computation rejected: {e}")
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        # Step 9: Dynamic Position Sizing (compute lot & risk AFTER verdict scaling!)
        verdict_mult = {"EXECUTE_FULL": 1.0, "EXECUTE_HALF": 0.5, "EXECUTE_MICRO": 0.25}.get(
            result.verdict, 1.0
        )
        conf_pct = max(result.buy_ratio, result.sell_ratio) * 100.0
        try:
            lot, actual_risk = self.risk.compute_lot(
                balance, entry, sl, conf_pct, verdict_multiplier=verdict_mult
            )
        except ValueError as e:
            logging.warning(f"Position sizing rejected: {e}")
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        trade_id = hashlib.md5(f"{signal_bar_iso}_{result.direction}".encode()).hexdigest()[:10]

        # Step 10: Optional Broker Execution Adapter
        try:
            exec_resp = self.executor.submit_limit_order(
                OrderRequest(
                    symbol=self.cfg.symbol,
                    side=result.direction,
                    entry=entry,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    lot=lot,
                    client_order_id=trade_id,
                )
            )
        except ExecutionError as e:
            logging.error(f"Broker execution rejected: {redact(e)}")
            self.tg.send(
                f"❌ <b>Order Execution Rejected</b>\n<code>{TelegramNotifier.escape(redact(e))}</code>"
            )
            self._update_state(last_price, now_str, signal_bar_iso)
            return

        # Step 11: Format MT5-Ready Signal Message
        action_name = f"{result.direction} LIMIT"
        ob_display = (
            f"✅ {ob_zone['high']:.2f}–{ob_zone['low']:.2f}"
            if ob_zone
            else "⚠️ ATR Dynamic Buffer"
        )

        tg_message = (
            f"🏛️ <b>APEX QUANT v{VERSION} — {action_name}</b>\n\n"
            f"Asset: <b>{TelegramNotifier.escape(self.cfg.symbol)}</b> ({TelegramNotifier.escape(src)})\n"
            f"Order Type: <b>{action_name}</b> [{TelegramNotifier.escape(exec_resp.mode)}]\n"
            f"Entry Price: <code>{entry:.2f}</code>\n"
            f"Stop Loss: <code>{sl:.2f}</code>\n\n"
            f"🎯 TP1: <code>{tp1:.2f}</code> (1/3 Close + Move SL to BE)\n"
            f"🎯 TP2: <code>{tp2:.2f}</code> (1/3 Close)\n"
            f"🎯 TP3: <code>{tp3:.2f}</code> (Runner Target)\n\n"
            f"📦 <b>Suggested Lot: <code>{lot}</code></b>\n"
            f"💰 Balance: <code>${balance:.2f}</code> | Risk: <code>{actual_risk:.2f}%</code>\n"
            f"📊 Consensus: BUY={result.buy_ratio:.0%} | SELL={result.sell_ratio:.0%}\n"
            f"🧠 Verdict: <b>{result.verdict}</b> | Active Models: {result.active_models}/{result.total_models}\n"
            f"🌐 Regime: <b>{result.regime}</b> (Hurst={result.hurst:.2f}, Entropy={result.entropy:.2f}, H4={result.htf_bias})\n"
            f"🧱 SMC Zone: {ob_display}\n\n"
            f"⏰ Bar Close Anchor: <code>{anchor_ts_iso} UTC</code>"
        )

        sent = self.tg.send(tg_message)
        if sent:
            trade_record = {
                "id": trade_id,
                "broker_order_id": exec_resp.broker_order_id,
                "timestamp": now_str,
                "signal_bar_ts": signal_bar_iso,
                "anchor_ts": anchor_ts_iso,
                "symbol": self.cfg.symbol,
                "data_source": src,
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
                "mfe_r": 0.0,
                "mae_r": 0.0,
            }
            journal.append(trade_record)
            self.kv.set("journal", journal)

            cb = self.kv.get("circuit_breaker", {}) or {}
            cb["daily_signals_count"] = int(cb.get("daily_signals_count", 0)) + 1
            self.kv.set("circuit_breaker", cb)

            state["last_signal_type"] = result.direction
            state["last_signal_ts"] = now_str
            self.kv.set("state", state)

        self._update_state(last_price, now_str, signal_bar_iso)

    def run_loop(self, interval_sec: int = 300) -> None:
        """Continuous daemon loop with graceful SIGINT/SIGTERM shutdown for VPS/Docker."""
        def _handle_stop(signum: int, _frame: object) -> None:
            logging.info(f"Received shutdown signal ({signum}). Finishing current cycle cleanly...")
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        logging.info(f"Starting APEX QUANT v{VERSION} continuous loop (interval={interval_sec}s)...")
        while not self._stop_requested:
            try:
                self.run()
            except Exception as e:
                logging.exception(f"Unhandled loop iteration error: {redact(e)}")
            for _ in range(max(1, interval_sec)):
                if self._stop_requested:
                    break
                time.sleep(1)
        logging.info("APEX QUANT daemon stopped gracefully.")

    def _update_state(self, last_price: float, now_str: str, bar_iso: str) -> None:
        state = self.kv.get("state", {}) or {}
        state["last_run_utc"] = now_str
        state["last_gold_price"] = last_price
        state["last_processed_bar_iso"] = bar_iso
        state["engine_version"] = VERSION
        self.kv.set("state", state)
        self.kv.save()
