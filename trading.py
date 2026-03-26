import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    opened_at: str


class PaperWallet:
    def __init__(
        self,
        state_file: str = "wallet_state.json",
        initial_balance: float = 1000.0,
    ) -> None:
        self.state_path = Path(state_file)
        self.initial_balance = initial_balance
        self.state = self._load_or_initialize_state()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "balance": self.initial_balance,
            "position": None,
            "realized_pnl": 0.0,
            "transactions": [],
            "last_updated": self._utc_now(),
        }

    def _load_or_initialize_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            state = self._default_state()
            self._save_state(state)
            return state

        with self.state_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        current = state if state is not None else self.state
        current["last_updated"] = self._utc_now()
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)

    def get_balance(self) -> float:
        return float(self.state["balance"])

    def get_position(self) -> Optional[Position]:
        position_data = self.state.get("position")
        if not position_data:
            return None
        return Position(**position_data)

    def has_open_position(self) -> bool:
        return self.state.get("position") is not None

    def current_unrealized_pnl_percent(self, current_price: float) -> float:
        position = self.get_position()
        if not position:
            return 0.0
        if position.avg_price <= 0:
            return 0.0
        return ((current_price - position.avg_price) / position.avg_price) * 100.0

    def buy(self, symbol: str, price: float, usd_amount: Optional[float] = None) -> Dict[str, Any]:
        if self.has_open_position():
            return {"success": False, "reason": "Open position already exists."}

        amount = usd_amount if usd_amount is not None else self.get_balance()
        balance = self.get_balance()

        if amount <= 0:
            return {"success": False, "reason": "Buy amount must be positive."}
        if amount > balance:
            return {"success": False, "reason": "Insufficient balance."}
        if price <= 0:
            return {"success": False, "reason": "Invalid price."}

        quantity = round(amount / price, 8)
        spent = quantity * price

        self.state["balance"] = round(balance - spent, 6)
        self.state["position"] = asdict(
            Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                opened_at=self._utc_now(),
            )
        )

        tx = {
            "type": "BUY",
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "usd_value": spent,
            "timestamp": self._utc_now(),
        }
        self.state["transactions"].append(tx)
        self._save_state()
        return {"success": True, "transaction": tx, "balance": self.get_balance()}

    def sell(self, price: float) -> Dict[str, Any]:
        position = self.get_position()
        if not position:
            return {"success": False, "reason": "No open position to sell."}
        if price <= 0:
            return {"success": False, "reason": "Invalid price."}

        proceeds = position.quantity * price
        cost = position.quantity * position.avg_price
        pnl = proceeds - cost
        pnl_percent = ((price - position.avg_price) / position.avg_price) * 100.0

        self.state["balance"] = round(self.get_balance() + proceeds, 6)
        self.state["realized_pnl"] = round(float(self.state.get("realized_pnl", 0.0)) + pnl, 6)
        self.state["position"] = None

        tx = {
            "type": "SELL",
            "symbol": position.symbol,
            "price": price,
            "quantity": position.quantity,
            "usd_value": proceeds,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "timestamp": self._utc_now(),
        }
        self.state["transactions"].append(tx)
        self._save_state()
        return {
            "success": True,
            "transaction": tx,
            "balance": self.get_balance(),
            "realized_pnl": self.state["realized_pnl"],
        }

    def recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.state.get("transactions", [])[-limit:]

