"""Unit tests for core stdlib-safe subsystems (Config, Security, KVStore, NewsFilter, Risk)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from apex_quant.config import UTC, Config
from apex_quant.exceptions import ConfigError
from apex_quant.news_filter import NewsSafetyFilter, parse_iso_utc
from apex_quant.risk import RiskManagementEngine
from apex_quant.security import redact
from apex_quant.storage import KVStore


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self._payload


class _FakeHttp:
    def __init__(self, payload: object = None, should_fail: bool = False) -> None:
        self.payload = payload if payload is not None else []
        self.should_fail = should_fail
        self.calls = 0

    def request(self, method: str, url: str, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("Simulated network outage")
        return _FakeResponse(self.payload)


class TestConfigAndSecurity(unittest.TestCase):
    def test_config_validate_rejects_bad_tp_weights(self) -> None:
        cfg = Config(tp1_weight=0.5, tp2_weight=0.5, tp3_weight=0.5)
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_config_validate_rejects_bad_contract_size(self) -> None:
        cfg = Config(contract_size=10.0)
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_redact_masks_secrets(self) -> None:
        raw = (
            "https://api.twelvedata.com/time_series?apikey=secret12345 "
            "Authorization: Bearer tok_abc_123 ghp_123456789012345678901234567890 "
            "bot1234567:ABCDEFGHIJKLMNOPQRSTUVWX"
        )
        cleaned = redact(raw)
        self.assertNotIn("secret12345", cleaned)
        self.assertNotIn("tok_abc_123", cleaned)
        self.assertNotIn("ghp_1234567890", cleaned)
        self.assertNotIn("ABCDEFGHIJKLMNOP", cleaned)


class TestKVStoreAndArchival(unittest.TestCase):
    def test_atomic_write_and_archival_preserves_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            lock_path = Path(tmp) / "engine.lock"
            cfg = Config(
                local_state_file=str(state_path),
                lock_file=str(lock_path),
                account_balance=1000.0,
                journal_max_entries=3,
            )
            kv = KVStore(cfg)
            kv.force_load()

            # 5 closed trades (+1R at 1% risk each) + 1 pending trade
            journal = [
                {"id": f"t{i}", "timestamp": f"2026-01-01 00:0{i}:00", "outcome": "WIN_TP3", "r_achieved": 1.0, "risk_pct": 1.0}
                for i in range(5)
            ]
            journal.append({"id": "p1", "timestamp": "2026-01-01 01:00:00", "outcome": "PENDING"})
            kv.set("journal", journal)
            kv.save()

            risk = RiskManagementEngine(cfg, kv)
            # Equity after 5 compounded +1% wins = 1000 * (1.01^5) = 1051.01
            self.assertAlmostEqual(risk.dynamic_balance(), 1051.01, places=2)
            # Pending trade must never be pruned
            kept_ids = [j["id"] for j in kv.get("journal")]
            self.assertIn("p1", kept_ids)
            self.assertEqual(len(kept_ids), 3)

    def test_gist_migrates_legacy_v18_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(
                gist_id="abc123",
                gist_token="tok_xyz",
                gist_state_file="apex_state_v19.json",
                local_state_file=os.path.join(tmp, "s.json"),
                lock_file=os.path.join(tmp, "l.lock"),
            )
            legacy_payload = {
                "files": {
                    "apex_state_v18.json": {
                        "content": json.dumps({"journal": [{"id": "legacy1", "outcome": "WIN_TP3", "r_achieved": 1.2}]})
                    }
                }
            }
            kv = KVStore(cfg, http=_FakeHttp(payload=legacy_payload))
            kv.force_load()
            self.assertEqual(len(kv.get("journal")), 1)
            self.assertEqual(kv.get("journal")[0]["id"], "legacy1")


class TestNewsFilter(unittest.TestCase):
    def test_iso_z_parsing_python310(self) -> None:
        dt = parse_iso_utc("2026-11-15T14:30:00Z")
        self.assertEqual(dt.tzinfo, UTC)
        self.assertEqual(dt.hour, 14)

    def test_rollover_blackout(self) -> None:
        cfg = Config()
        nf = NewsSafetyFilter(cfg, _FakeHttp())
        blackout, msg = nf.is_blackout(now=datetime(2026, 11, 16, 22, 0, tzinfo=UTC))
        self.assertTrue(blackout)
        self.assertIn("Rollover", msg)

    def test_news_blackout_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(
                local_state_file=os.path.join(tmp, "s.json"),
                lock_file=os.path.join(tmp, "l.lock"),
            )
            kv = KVStore(cfg)
            now = datetime(2026, 11, 16, 13, 25, tzinfo=UTC)
            events = [
                {
                    "country": "USD",
                    "impact": "High",
                    "title": "CPI m/m",
                    "date": "2026-11-16T13:30:00Z",
                }
            ]
            http = _FakeHttp(payload=events)
            nf = NewsSafetyFilter(cfg, http, kv=kv)
            blocked, title = nf.is_blackout(now=now)
            self.assertTrue(blocked)
            self.assertIn("CPI", title)
            # Second call within TTL must hit cache without calling HTTP again
            blocked2, _ = nf.is_blackout(now=now + timedelta(minutes=1))
            self.assertTrue(blocked2)
            self.assertEqual(http.calls, 1)


class TestRiskManagement(unittest.TestCase):
    def test_lot_and_post_verdict_risk_pct_accuracy(self) -> None:
        cfg = Config(account_balance=10000.0, max_risk_pct_per_trade=1.0, hard_leverage_cap=30.0)
        kv = KVStore(cfg)
        risk = RiskManagementEngine(cfg, kv)

        lot_full, risk_full = risk.compute_lot(10000.0, entry=2600.0, sl=2590.0, confidence_pct=100.0, verdict_multiplier=1.0)
        # Risk = $100; SL dist = $10/oz; Contract = 100 oz -> 100 / (10 * 100) = 0.10 lot
        self.assertEqual(lot_full, 0.10)
        self.assertEqual(risk_full, 1.0)

        lot_half, risk_half = risk.compute_lot(10000.0, entry=2600.0, sl=2590.0, confidence_pct=100.0, verdict_multiplier=0.5)
        self.assertEqual(lot_half, 0.05)
        self.assertEqual(risk_half, 0.5)

    def test_hard_leverage_cap(self) -> None:
        cfg = Config(account_balance=1000.0, max_risk_pct_per_trade=2.0, hard_leverage_cap=30.0)
        kv = KVStore(cfg)
        risk = RiskManagementEngine(cfg, kv)
        # Entry = 3000, SL = 2999 ($1/oz). 2% = $20 -> 0.20 lot -> notional $60,000 (60x > 30x!)
        lot, _ = risk.compute_lot(1000.0, entry=3000.0, sl=2999.0, confidence_pct=100.0)
        # Max permitted notional = $30,000 -> 0.10 lot
        self.assertLessEqual(lot * 100.0 * 3000.0 / 1000.0, 30.0 + 1e-6)

    def test_consecutive_losses_no_infinite_lockout_after_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(
                local_state_file=os.path.join(tmp, "s.json"),
                lock_file=os.path.join(tmp, "l.lock"),
                max_consecutive_losses=3,
                dd_pause_hours=24,
            )
            kv = KVStore(cfg)
            kv.set(
                "journal",
                [
                    {"id": "1", "timestamp": "2026-11-10 10:00:00", "outcome": "LOSS", "r_achieved": -1.0, "risk_pct": 1.0},
                    {"id": "2", "timestamp": "2026-11-10 11:00:00", "outcome": "LOSS", "r_achieved": -1.0, "risk_pct": 1.0},
                    {"id": "3", "timestamp": "2026-11-10 12:00:00", "outcome": "LOSS", "r_achieved": -1.0, "risk_pct": 1.0},
                ],
            )
            risk = RiskManagementEngine(cfg, kv)
            t0 = datetime(2026, 11, 10, 13, 0, tzinfo=UTC)
            ok1, reason1 = risk.check_circuit_breakers(970.0, now=t0)
            self.assertFalse(ok1)
            self.assertIn("CONSECUTIVE LOSSES", reason1)

            # Still locked 12 hours later
            ok2, _ = risk.check_circuit_breakers(970.0, now=t0 + timedelta(hours=12))
            self.assertFalse(ok2)

            # 25 hours later (after pause expires) MUST UNLOCK cleanly instead of re-locking forever!
            ok3, _ = risk.check_circuit_breakers(970.0, now=t0 + timedelta(hours=25))
            self.assertTrue(ok3)


if __name__ == "__main__":
    unittest.main()
