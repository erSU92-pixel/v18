"""Trade outcome tracking with exact weighted partial TP/SL R-multiple accounting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import UTC, Config
from .storage import KVStore


class OutcomeTracker:
    """Monitors active and pending limit trades against closed candlestick data."""

    def __init__(self, cfg: Config, kv: KVStore) -> None:
        self.cfg = cfg
        self.kv = kv

    def _compute_realized_r(
        self,
        ep: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        hit_tp1: bool,
        hit_tp2: bool,
        hit_tp3: bool,
        hit_be: bool,
        hit_sl: bool,
    ) -> float:
        """
        Computes exact weighted R-multiple based on actual SL distance and partial TP weights.
        """
        sl_dist = abs(ep - sl)
        if sl_dist <= 1e-9:
            return 0.0

        r_tp1 = abs(tp1 - ep) / sl_dist
        r_tp2 = abs(tp2 - ep) / sl_dist
        r_tp3 = abs(tp3 - ep) / sl_dist
        w1, w2, w3 = self.cfg.tp1_weight, self.cfg.tp2_weight, self.cfg.tp3_weight

        if hit_tp3:
            gross_r = (w1 * r_tp1) + (w2 * r_tp2) + (w3 * r_tp3)
        elif hit_be:
            gross_r = (w1 * r_tp1 if hit_tp1 else 0.0) + (w2 * r_tp2 if hit_tp2 else 0.0)
        elif hit_sl:
            gross_r = -1.0
        else:
            gross_r = 0.0

        return round(gross_r - self.cfg.slippage_r_cost, 4)

    def update(self, m5: pd.DataFrame, now: Optional[datetime] = None) -> List[Tuple[str, dict]]:
        journal = self.kv.get("journal", []) or []
        events: List[Tuple[str, dict]] = []
        now = now or datetime.now(UTC)
        modified = False

        if m5.empty:
            return events

        # Ensure m5 datetimes are sorted UTC timestamps
        m5_sorted = m5.sort_values("datetime")

        for entry in journal:
            if entry.get("outcome") != "PENDING":
                continue

            # Prefer exact bar_close_anchor_ts (signal bar open + 5m) if recorded, else timestamp
            anchor_str = entry.get("anchor_ts") or entry.get("timestamp", "")
            try:
                anchor_ts = datetime.strptime(anchor_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except Exception:
                anchor_ts = now - timedelta(hours=24)

            age_hours = (now - anchor_ts).total_seconds() / 3600.0
            if not entry.get("filled") and age_hours > self.cfg.pending_stale_hours:
                entry["outcome"] = "STALE"
                entry["filled"] = 0
                entry["r_achieved"] = 0.0
                entry["close_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
                events.append(("STALE", dict(entry)))
                modified = True
                continue

            side = str(entry.get("signal", ""))
            ep = float(entry.get("entry", 0.0))
            sl = float(entry.get("sl", 0.0))
            tp1 = float(entry.get("tp1", 0.0))
            tp2 = float(entry.get("tp2", 0.0))
            tp3 = float(entry.get("tp3", 0.0))

            if ep <= 0 or sl <= 0 or side not in ("BUY", "SELL"):
                continue

            bars = m5_sorted[m5_sorted["datetime"] >= anchor_ts]
            if bars.empty:
                continue

            filled = bool(entry.get("filled"))
            fill_index: Optional[int] = None

            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and float(b.low) <= ep:
                        filled, fill_index = True, idx
                        entry["fill_timestamp"] = b.datetime.strftime("%Y-%m-%d %H:%M:%S")
                        break
                    if side == "SELL" and float(b.high) >= ep:
                        filled, fill_index = True, idx
                        entry["fill_timestamp"] = b.datetime.strftime("%Y-%m-%d %H:%M:%S")
                        break

                if not filled:
                    chase_limit = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    max_h = float(bars["high"].max())
                    min_l = float(bars["low"].min())
                    if side == "BUY" and max_h > ep + chase_limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        entry["close_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                        continue
                    if side == "SELL" and min_l < ep - chase_limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        entry["close_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
                        events.append(("INVALIDATED", dict(entry)))
                        modified = True
                        continue
                    continue

                entry["filled"] = 1
                modified = True

            # Determine where to resume walking bars after fill or prior update
            last_eval_str = entry.get("last_eval_ts")
            if fill_index is not None:
                walk_bars = bars.iloc[fill_index:]
            elif last_eval_str:
                try:
                    last_eval_dt = datetime.strptime(last_eval_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                    walk_bars = bars[bars["datetime"] > last_eval_dt]
                except Exception:
                    walk_bars = bars
            else:
                walk_bars = bars

            if walk_bars.empty:
                continue

            hit_tp1 = bool(entry.get("hit_tp1"))
            hit_tp2 = bool(entry.get("hit_tp2"))
            hit_tp3 = bool(entry.get("hit_tp3"))
            hit_sl = bool(entry.get("hit_sl"))
            hit_be = bool(entry.get("hit_be"))
            sl_dist = abs(ep - sl)
            mfe_r = float(entry.get("mfe_r", 0.0))
            mae_r = float(entry.get("mae_r", 0.0))
            close_bar_ts: Optional[str] = None

            for b in walk_bars.itertuples(index=False):
                if hit_tp3 or hit_sl or hit_be:
                    break

                b_high = float(b.high)
                b_low = float(b.low)
                if sl_dist > 0:
                    if side == "BUY":
                        mfe_r = max(mfe_r, (b_high - ep) / sl_dist)
                        mae_r = max(mae_r, max(0.0, (ep - b_low) / sl_dist))
                    else:
                        mfe_r = max(mfe_r, (ep - b_low) / sl_dist)
                        mae_r = max(mae_r, max(0.0, (b_high - ep) / sl_dist))

                current_sl = ep if hit_tp1 else sl
                bar_ts_str = b.datetime.strftime("%Y-%m-%d %H:%M:%S")

                if side == "BUY":
                    # Conservative worst-case intra-bar resolution: SL/BE checked first
                    if b_low <= current_sl:
                        if hit_tp1:
                            hit_be = True
                        else:
                            hit_sl = True
                        close_bar_ts = bar_ts_str
                        break
                    if not hit_tp1 and b_high >= tp1:
                        hit_tp1 = True
                    if not hit_tp2 and b_high >= tp2:
                        hit_tp2 = True
                    if not hit_tp3 and b_high >= tp3:
                        hit_tp3 = True
                        close_bar_ts = bar_ts_str
                        break
                else:
                    if b_high >= current_sl:
                        if hit_tp1:
                            hit_be = True
                        else:
                            hit_sl = True
                        close_bar_ts = bar_ts_str
                        break
                    if not hit_tp1 and b_low <= tp1:
                        hit_tp1 = True
                    if not hit_tp2 and b_low <= tp2:
                        hit_tp2 = True
                    if not hit_tp3 and b_low <= tp3:
                        hit_tp3 = True
                        close_bar_ts = bar_ts_str
                        break

                entry["last_eval_ts"] = bar_ts_str
                modified = True

            entry["hit_tp1"] = int(hit_tp1)
            entry["hit_tp2"] = int(hit_tp2)
            entry["hit_tp3"] = int(hit_tp3)
            entry["hit_sl"] = int(hit_sl)
            entry["hit_be"] = int(hit_be)
            entry["mfe_r"] = round(mfe_r, 2)
            entry["mae_r"] = round(mae_r, 2)

            closed_now = hit_tp3 or hit_be or hit_sl
            if closed_now:
                if hit_tp3:
                    entry["outcome"] = "WIN_TP3"
                elif hit_be:
                    entry["outcome"] = "WIN_BE"
                else:
                    entry["outcome"] = "LOSS"

                entry["r_achieved"] = self._compute_realized_r(
                    ep, sl, tp1, tp2, tp3, hit_tp1, hit_tp2, hit_tp3, hit_be, hit_sl
                )
                entry["close_timestamp"] = close_bar_ts or now.strftime("%Y-%m-%d %H:%M:%S")
                events.append(("CLOSE", dict(entry)))
                modified = True

        if modified:
            self.kv.set("journal", journal)
            self.kv.save()

        return events
