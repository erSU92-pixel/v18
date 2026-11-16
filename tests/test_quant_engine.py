"""Quantitative signal, market data calibration, tracker, and backtest tests (requires numpy/pandas)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

try:
    import numpy as np
    import pandas as pd
    from apex_quant.backtester import WalkForwardBacktester, generate_synthetic_xauusd_m5
    from apex_quant.config import UTC, Config
    from apex_quant.indicators import Indicators
    from apex_quant.market_data import MarketDataHandler
    from apex_quant.signals import SignalGenerator
    from apex_quant.storage import KVStore
    from apex_quant.tracker import OutcomeTracker

    HAS_QUANT_DEPS = True
except ImportError:
    HAS_QUANT_DEPS = False


@unittest.skipUnless(HAS_QUANT_DEPS, "numpy/pandas not installed in current Python environment")
class TestQuantEngine(unittest.TestCase):
    def test_causal_lowpass_has_zero_future_leakage(self) -> None:
        df1 = generate_synthetic_xauusd_m5(bars=200, seed=7)
        df2 = df1.copy()
        # Mutate future bar (last 5 bars) dramatically
        df2.loc[df2.index[-5:], "close"] += 250.0

        f1 = Indicators.causal_lowpass(df1["close"])
        f2 = Indicators.causal_lowpass(df2["close"])
        # Filter output at index -10 MUST be bit-for-bit identical!
        self.assertAlmostEqual(float(f1.iloc[-10]), float(f2.iloc[-10]), places=9)

    def test_hurst_zero_variance_does_not_crash(self) -> None:
        flat = pd.Series([2600.0] * 100)
        h = Indicators.hurst(flat)
        self.assertTrue(0.0 <= h <= 1.0)

    def test_resample_does_not_double_apply_offset(self) -> None:
        start = datetime(2026, 11, 10, 0, 0, tzinfo=UTC)
        df = pd.DataFrame(
            {
                "datetime": [start + timedelta(minutes=5 * i) for i in range(6)],
                "open": [2600.0] * 6,
                "high": [2601.0] * 6,
                "low": [2599.0] * 6,
                "close": [2600.5] * 6,
                "volume": [100.0] * 6,
            }
        )
        calibrated_m5 = MarketDataHandler._apply_offset(df, 1.50)
        m15 = MarketDataHandler._resample(calibrated_m5, "15min")
        self.assertAlmostEqual(float(m15["open"].iloc[0]), 2601.50, places=2)

    def test_outcome_tracker_exact_weighted_partial_r(self) -> None:
        cfg = Config(tp1_weight=1 / 3, tp2_weight=1 / 3, tp3_weight=1 / 3, slippage_r_cost=0.0)
        kv = KVStore(cfg)
        start = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        # BUY @ 2600, SL @ 2590 (SL dist = 10)
        # TP1 @ 2610 (1R), TP2 @ 2624 (2.4R), TP3 @ 2640 (4.0R)
        kv.set(
            "journal",
            [
                {
                    "id": "tr1",
                    "timestamp": "2026-11-10 09:55:00",
                    "anchor_ts": "2026-11-10 10:00:00",
                    "symbol": "XAUUSD",
                    "signal": "BUY",
                    "entry": 2600.0,
                    "sl": 2590.0,
                    "tp1": 2610.0,
                    "tp2": 2624.0,
                    "tp3": 2640.0,
                    "filled": 0,
                    "outcome": "PENDING",
                }
            ],
        )
        m5 = pd.DataFrame(
            [
                {"datetime": start, "open": 2602, "high": 2603, "low": 2599.5, "close": 2601, "volume": 100},  # fills
                {"datetime": start + timedelta(minutes=5), "open": 2601, "high": 2611, "low": 2600.5, "close": 2610, "volume": 100},  # hits TP1
                {"datetime": start + timedelta(minutes=10), "open": 2610, "high": 2625, "low": 2609, "close": 2620, "volume": 100},  # hits TP2
                {"datetime": start + timedelta(minutes=15), "open": 2620, "high": 2621, "low": 2599.0, "close": 2600, "volume": 100},  # hits BE
            ]
        )
        tracker = OutcomeTracker(cfg, kv)
        events = tracker.update(m5, now=start + timedelta(minutes=20))
        self.assertEqual(len(events), 1)
        trade = events[0][1]
        self.assertEqual(trade["outcome"], "WIN_BE")
        # 1/3 * 1.0R + 1/3 * 2.4R + 1/3 * 0R = 1.1333R!
        self.assertAlmostEqual(float(trade["r_achieved"]), (1.0 + 2.4) / 3.0, places=3)

    def test_backtester_runs_end_to_end(self) -> None:
        cfg = Config(skip_killzone_filter=True, enable_ml_models=False)
        df = generate_synthetic_xauusd_m5(bars=700, seed=99)
        bt = WalkForwardBacktester(cfg)
        metrics = bt.run(df, step_bars=10)
        self.assertGreaterEqual(metrics.final_balance, 10.0)


if __name__ == "__main__":
    unittest.main()
