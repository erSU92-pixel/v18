"""Broker execution adapters (Signal-Only default + optional OANDA v20 LIMIT order routing)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .config import Config
from .exceptions import ExecutionError
from .security import redact


@dataclass
class OrderRequest:
    symbol: str
    side: str       # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    lot: float
    client_order_id: str


@dataclass
class OrderResponse:
    accepted: bool
    mode: str
    broker_order_id: Optional[str] = None
    message: str = ""


class ExecutionAdapter(ABC):
    @abstractmethod
    def submit_limit_order(self, order: OrderRequest) -> OrderResponse:
        """Submits a limit order or logs signal-only execution intent."""


class SignalOnlyAdapter(ExecutionAdapter):
    def submit_limit_order(self, order: OrderRequest) -> OrderResponse:
        return OrderResponse(
            accepted=True,
            mode="signal_only",
            broker_order_id=order.client_order_id,
            message="Signal-only mode: alert dispatched for manual/MT5 execution.",
        )


class OandaExecutionAdapter(ExecutionAdapter):
    """Routes live/practice LIMIT orders to OANDA v20 REST API with SL and TP3 attached."""

    def __init__(self, cfg: Config, http: Any) -> None:
        self.cfg = cfg
        self.http = http
        self.api_key = os.getenv("OANDA_API_KEY", "").strip()
        self.account_id = (cfg.oanda_account_id or os.getenv("OANDA_ACCOUNT_ID", "")).strip()

    def _base_url(self) -> str:
        if self.cfg.oanda_env == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    def submit_limit_order(self, order: OrderRequest) -> OrderResponse:
        if self.cfg.dry_run:
            return OrderResponse(
                accepted=True,
                mode="oanda_dry_run",
                broker_order_id=f"DRY-{order.client_order_id}",
                message="Dry-run: skipped live OANDA order submission.",
            )
        if not (self.api_key and self.account_id):
            raise ExecutionError("OANDA_API_KEY and OANDA_ACCOUNT_ID required for oanda mode")

        # 1 standard lot = 100 units of XAU_USD in OANDA (1 unit = 1 troy oz)
        units = int(round(order.lot * self.cfg.contract_size))
        if units < 1:
            raise ExecutionError(f"Computed OANDA units ({units}) < 1")
        signed_units = str(units if order.side == "BUY" else -units)

        payload = {
            "order": {
                "type": "LIMIT",
                "instrument": "XAU_USD",
                "units": signed_units,
                "price": f"{order.entry:.{self.cfg.price_decimals}f}",
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "clientExtensions": {"id": order.client_order_id[:32]},
                "stopLossOnFill": {
                    "price": f"{order.sl:.{self.cfg.price_decimals}f}",
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": f"{order.tp2:.{self.cfg.price_decimals}f}",
                    "timeInForce": "GTC",
                },
            }
        }
        url = f"{self._base_url()}/v3/accounts/{self.account_id}/orders"
        try:
            resp = self.http.request(
                "POST",
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            body = resp.json()
            tx = body.get("orderCreateTransaction", {})
            oid = tx.get("id")
            logging.info(f"OANDA LIMIT order created: id={oid}, units={signed_units} @ {order.entry}")
            return OrderResponse(
                accepted=True,
                mode=f"oanda_{self.cfg.oanda_env}",
                broker_order_id=str(oid) if oid else order.client_order_id,
                message="OANDA order accepted",
            )
        except Exception as e:
            raise ExecutionError(f"OANDA order placement failed: {redact(e)}") from e
