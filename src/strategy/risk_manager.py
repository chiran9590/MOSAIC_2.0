from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RiskConfig:
    capital: float = 100000.0
    risk_per_trade_pct: float = 0.5
    max_position_value_pct: float = 20.0
    stop_loss_pct: float = 1.0
    estimated_round_trip_cost_pct: float = 0.20


@dataclass
class RiskAssessment:
    accepted: bool
    direction: str
    entry_price: float
    stop_loss: float
    quantity: int
    position_value: float
    maximum_price_risk: float
    estimated_transaction_cost: float
    estimated_total_risk: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskManager:

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._validate_config()

    def _validate_config(self):

        c = self.config

        if c.capital <= 0:
            raise ValueError("Capital must be positive.")

        if not 0 < c.risk_per_trade_pct <= 100:
            raise ValueError("Invalid risk percentage.")

        if not 0 < c.max_position_value_pct <= 100:
            raise ValueError("Invalid position limit.")

        if not 0 < c.stop_loss_pct < 100:
            raise ValueError("Invalid stop-loss percentage.")

        if not 0 <= c.estimated_round_trip_cost_pct < 100:
            raise ValueError("Invalid transaction cost percentage.")

    def assess_trade(
        self,
        direction: str,
        entry_price: float,
    ) -> RiskAssessment:

        direction = direction.upper()

        if direction not in {"LONG", "SHORT"}:
            raise ValueError("Direction must be LONG or SHORT.")

        entry_price = float(entry_price)

        if not 0 < entry_price < float("inf"):
            raise ValueError("Entry price must be finite and positive.")

        c = self.config

        risk_budget = (
            c.capital * c.risk_per_trade_pct / 100
        )

        stop_distance = (
            entry_price * c.stop_loss_pct / 100
        )

        stop_loss = (
            entry_price - stop_distance
            if direction == "LONG"
            else entry_price + stop_distance
        )

        risk_per_share = stop_distance

        cost_per_share = (
            entry_price *
            c.estimated_round_trip_cost_pct / 100
        )

        total_risk_per_share = (
            risk_per_share + cost_per_share
        )

        max_quantity_by_risk = int(
            risk_budget / total_risk_per_share
        )

        max_position_value = (
            c.capital * c.max_position_value_pct / 100
        )

        max_quantity_by_value = int(
            max_position_value / entry_price
        )

        quantity = min(
            max_quantity_by_risk,
            max_quantity_by_value,
        )

        position_value = quantity * entry_price

        maximum_price_risk = quantity * risk_per_share

        estimated_transaction_cost = (
            quantity * cost_per_share
        )

        estimated_total_risk = (
            maximum_price_risk +
            estimated_transaction_cost
        )

        accepted = quantity > 0

        reason = (
            "Trade fits configured risk limits."
            if accepted
            else "Risk budget or position limit is too small."
        )

        return RiskAssessment(
            accepted=accepted,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            quantity=quantity,
            position_value=position_value,
            maximum_price_risk=maximum_price_risk,
            estimated_transaction_cost=estimated_transaction_cost,
            estimated_total_risk=estimated_total_risk,
            reason=reason,
        )


if __name__ == "__main__":

    manager = RiskManager()

    result = manager.assess_trade(
        direction="LONG",
        entry_price=1500.0,
    )

    print("\nMOSAIC 2.0 - RISK MANAGER TEST")
    print("=" * 55)

    for key, value in result.to_dict().items():
        print(f"{key}: {value}")

    print("=" * 55)
