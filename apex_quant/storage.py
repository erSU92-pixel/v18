"""Thread-safe, crash-safe atomic persistence with GitHub Gist cloud synchronization."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from .config import VERSION, Config
from .exceptions import StateError
from .security import redact


def _default_store() -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "journal": [],
        "state": {},
        "metrics": {},
        "circuit_breaker": {},
        "news_cache": {},
        "archived_stats": {},
    }


class KVStore:
    """Thread-safe and process-safe state store backed by atomic JSON + optional GitHub Gist."""

    def __init__(self, cfg: Config, http: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.http = http
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = _default_store()
        self.loaded = False

    @contextmanager
    def process_lock(self) -> Iterator[None]:
        """Advisory file lock preventing concurrent cron/action overlapping writes."""
        lock_path = Path(self.cfg.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = lock_path.open("a+", encoding="utf-8")
        try:
            if HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if HAS_FCNTL:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()

    def _hydrate_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        base = _default_store()
        if not isinstance(raw, dict):
            raise StateError("Corrupted state payload: root JSON value must be an object")
        for key, default_val in base.items():
            val = raw.get(key)
            if val is None:
                raw[key] = default_val
        return raw

    def force_load(self) -> None:
        """Loads state from GitHub Gist or falls back to local atomic JSON storage."""
        with self._lock:
            if self.cfg.gist_id and self.cfg.gist_token and self.http is not None:
                try:
                    url = f"https://api.github.com/gists/{self.cfg.gist_id}"
                    headers = {
                        "Authorization": f"Bearer {self.cfg.gist_token}",
                        "Accept": "application/vnd.github+json",
                    }
                    resp = self.http.request("GET", url, headers=headers)
                    files = resp.json().get("files", {})
                    target_file = files.get(self.cfg.gist_state_file)
                    if not (target_file and target_file.get("content")) and self.cfg.gist_state_file != "apex_state_v18.json":
                        legacy_file = files.get("apex_state_v18.json")
                        if legacy_file and legacy_file.get("content"):
                            target_file = legacy_file
                            logging.info("Migrating existing state from legacy Gist file: apex_state_v18.json")
                    if target_file and target_file.get("content"):
                        parsed = json.loads(target_file["content"])
                        self.data = self._hydrate_schema(parsed)
                        self.loaded = True
                        logging.info("State successfully loaded from GitHub Gist cloud storage.")
                        return
                except Exception as e:
                    logging.warning(
                        f"Gist state read failed: {redact(e)}. Falling back to local storage."
                    )

            p = Path(self.cfg.local_state_file)
            if p.exists():
                try:
                    parsed = json.loads(p.read_text(encoding="utf-8"))
                    self.data = self._hydrate_schema(parsed)
                    self.loaded = True
                    logging.info(f"State loaded from local file: {self.cfg.local_state_file}")
                    return
                except Exception as e:
                    raise StateError(f"Local state file unreadable/corrupted ({p}): {e}") from e

            self.data = _default_store()
            self.loaded = True
            logging.info("No existing state found. Initialized fresh state store.")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value

    def archive_old_journal_if_needed(self) -> int:
        """Prunes old terminal trades beyond journal_max_entries while preserving compounding equity."""
        with self._lock:
            journal = self.data.get("journal", []) or []
            limit = self.cfg.journal_max_entries
            if len(journal) <= limit:
                return 0

            pending = [j for j in journal if j.get("outcome") == "PENDING"]
            terminal = [j for j in journal if j.get("outcome") != "PENDING"]
            terminal.sort(key=lambda x: x.get("timestamp", ""))

            keep_terminal_count = max(0, limit - len(pending))
            if len(terminal) <= keep_terminal_count:
                return 0

            prune_count = len(terminal) - keep_terminal_count
            to_archive = terminal[:prune_count]
            kept_terminal = terminal[prune_count:]

            arch = self.data.get("archived_stats", {}) or {}
            equity = float(arch.get("compounded_equity", self.cfg.account_balance))
            closed_trades = int(arch.get("closed_trades", 0))
            wins = int(arch.get("wins", 0))
            losses = int(arch.get("losses", 0))
            net_r = float(arch.get("net_r", 0.0))

            for t in to_archive:
                outcome = str(t.get("outcome", ""))
                if outcome in ("WIN_TP3", "WIN_BE", "LOSS") and t.get("r_achieved") is not None:
                    try:
                        r_mult = float(t["r_achieved"])
                        risk_pct = float(t.get("risk_pct", 1.0))
                        equity += r_mult * (equity * (risk_pct / 100.0))
                        equity = max(10.0, equity)
                        closed_trades += 1
                        net_r += r_mult
                        if outcome.startswith("WIN"):
                            wins += 1
                        else:
                            losses += 1
                    except (ValueError, TypeError):
                        continue

            self.data["archived_stats"] = {
                "compounded_equity": round(equity, 4),
                "closed_trades": closed_trades,
                "wins": wins,
                "losses": losses,
                "net_r": round(net_r, 4),
            }
            self.data["journal"] = kept_terminal + pending
            logging.info(f"Archived {prune_count} old journal entries; preserved equity=${equity:.2f}")
            return prune_count

    def _atomic_write_local(self, serialized: str) -> None:
        target = Path(self.cfg.local_state_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(serialized)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def save(self) -> None:
        """Persists memory atomically to local disk and syncs to GitHub Gist if configured."""
        with self._lock:
            self.archive_old_journal_if_needed()
            serialized = json.dumps(self.data, indent=2, default=str)
            try:
                self._atomic_write_local(serialized)
            except Exception as e:
                logging.error(f"Local state atomic write error: {e}")

            if self.cfg.gist_id and self.cfg.gist_token and self.http is not None:
                try:
                    url = f"https://api.github.com/gists/{self.cfg.gist_id}"
                    headers = {
                        "Authorization": f"Bearer {self.cfg.gist_token}",
                        "Accept": "application/vnd.github+json",
                    }
                    payload = {
                        "description": f"Apex Quant State v{VERSION} - Auto Saved",
                        "files": {self.cfg.gist_state_file: {"content": serialized}},
                    }
                    self.http.request("PATCH", url, headers=headers, json=payload)
                    logging.info("State synchronized to GitHub Gist.")
                except Exception as e:
                    logging.error(f"Gist sync error: {redact(e)}")
