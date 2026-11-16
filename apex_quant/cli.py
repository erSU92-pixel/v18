"""Command-line interface entrypoint for APEX QUANT v19.0."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from .config import VERSION, Config
from .exceptions import ConfigError


def setup_logging(verbose: bool = False) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="APEX QUANT Institutional Quantitative Trading Engine for Gold (XAUUSD)"
    )
    parser.add_argument(
        "command",
        choices=["run", "test", "audit", "state", "backtest", "reset-lock", "version"],
        help="Command to execute",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Run without live Telegram/broker orders")
    parser.add_argument("--loop", action="store_true", help="Run continuously in daemon mode (for Docker/VPS)")
    parser.add_argument("--interval", type=int, default=300, help="Loop sleep interval in seconds (default: 300)")
    parser.add_argument("--csv", type=str, default="", help="Optional M5 OHLCV CSV path for backtest")
    parser.add_argument("--bars", type=int, default=2500, help="Synthetic M5 bars for backtest (default: 2500)")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = Config.from_env()
    if args.dry_run:
        cfg.dry_run = True

    try:
        cfg.validate()
    except ConfigError as e:
        logging.critical(f"Configuration validation failed: {e}")
        sys.exit(1)

    if args.command == "version":
        print(f"APEX QUANT Engine v{VERSION}")
        sys.exit(0)

    if args.command == "backtest":
        import pandas as pd

        from .backtester import WalkForwardBacktester, generate_synthetic_xauusd_m5

        if args.csv:
            df = pd.read_csv(args.csv)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        else:
            logging.info(f"Running walk-forward backtest on {args.bars} synthetic calibrated XAUUSD M5 bars...")
            df = generate_synthetic_xauusd_m5(bars=args.bars)
        bt = WalkForwardBacktester(cfg)
        metrics = bt.run(df)
        print(json.dumps(asdict(metrics), indent=2))
        sys.exit(0)

    if args.command == "audit":
        from .backtester import compute_journal_metrics
        from .storage import KVStore

        kv = KVStore(cfg)
        kv.force_load()
        journal = kv.get("journal", []) or []
        metrics = compute_journal_metrics(
            journal, cfg.account_balance, kv.get("archived_stats")
        )
        print(json.dumps(asdict(metrics), indent=2))
        sys.exit(0)

    from .orchestrator import TradingBotOrchestrator

    orchestrator = TradingBotOrchestrator(cfg)
    try:
        if args.command == "run":
            if args.loop:
                orchestrator.run_loop(interval_sec=args.interval)
            else:
                orchestrator.run()
        elif args.command == "state":
            orchestrator.kv.force_load()
            print(json.dumps(orchestrator.kv.data, indent=2, default=str))
        elif args.command == "reset-lock":
            orchestrator.kv.force_load()
            orchestrator.risk.reset_locks()
            logging.info("Circuit breaker lock cleared and streak reset anchor updated.")
        elif args.command == "test":
            logging.info("Running system connectivity self-test...")
            orchestrator.kv.force_load()
            dfs, src, offset = orchestrator.data.fetch_all()
            logging.info(
                f"Self-test SUCCESS: Source={src}, M5 Bars={len(dfs['5min'])}, "
                f"H4 Bars={len(dfs['4h'])}, Offset={offset:+.3f}"
            )
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
