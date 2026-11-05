from dataclasses import dataclass, asdict
from typing import Any

from src.strategy.decision_engine import DecisionEngine
from src.strategy.risk_manager import RiskManager, RiskConfig


@dataclass
class StrategyResult:
    symbol: str
    decision: str
    direction: str | None
    prediction: str
    probabilities: dict[str, float]
    candle_timestamp: str
    prediction_horizon_minutes: int
    model_version: str
    model_status: str

    risk_accepted: bool
    entry_price: float | None
    stop_loss: float | None
    quantity: int
    position_value: float
    estimated_total_risk: float
    risk_reason: str

    paper_trading_only: bool
    real_order_placed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyPipeline:
    """
    Connects model predictions to decision and risk evaluation.

    This pipeline only creates paper-trading candidates.
    It never places real broker orders.
    """

    def __init__(
        self,
        risk_config: RiskConfig | None = None,
    ):
        self.decision_engine = DecisionEngine()
        self.risk_manager = RiskManager(risk_config)

    def evaluate(
        self,
        prediction: dict[str, Any],
        entry_price: float | None = None,
    ) -> StrategyResult:

        # 1. Convert prediction into a directional decision.
        decision = self.decision_engine.evaluate(prediction)

        # 2. HOLD or NO_TRADE does not need position sizing.
        if decision.direction is None:

            return StrategyResult(
                symbol=decision.symbol,
                decision=decision.decision,
                direction=None,
                prediction=decision.prediction,
                probabilities=decision.probabilities,
                candle_timestamp=decision.candle_timestamp,
                prediction_horizon_minutes=decision.prediction_horizon_minutes,
                model_version=decision.model_version,
                model_status=decision.status,
                risk_accepted=False,
                entry_price=None,
                stop_loss=None,
                quantity=0,
                position_value=0.0,
                estimated_total_risk=0.0,
                risk_reason="No directional candidate to assess.",
                paper_trading_only=True,
                real_order_placed=False,
                reason=decision.reason,
            )

        # 3. Entry price is required for risk assessment.
        if entry_price is None:
            return StrategyResult(
                symbol=decision.symbol,
                decision="PENDING_RISK_CHECK",
                direction=decision.direction,
                prediction=decision.prediction,
                probabilities=decision.probabilities,
                candle_timestamp=decision.candle_timestamp,
                prediction_horizon_minutes=decision.prediction_horizon_minutes,
                model_version=decision.model_version,
                model_status=decision.status,
                risk_accepted=False,
                entry_price=None,
                stop_loss=None,
                quantity=0,
                position_value=0.0,
                estimated_total_risk=0.0,
                risk_reason="Entry price is required for risk assessment.",
                paper_trading_only=True,
                real_order_placed=False,
                reason="Directional candidate generated; risk check pending.",
            )

        # 4. Apply risk and position-sizing rules.
        risk = self.risk_manager.assess_trade(
            direction=decision.direction,
            entry_price=entry_price,
        )

        # 5. Final paper-trading result.
        final_decision = (
            "PAPER_TRADE_CANDIDATE"
            if risk.accepted
            else "REJECTED_BY_RISK"
        )

        return StrategyResult(
            symbol=decision.symbol,
            decision=final_decision,
            direction=decision.direction,
            prediction=decision.prediction,
            probabilities=decision.probabilities,
            candle_timestamp=decision.candle_timestamp,
            prediction_horizon_minutes=decision.prediction_horizon_minutes,
            model_version=decision.model_version,
            model_status=decision.status,
            risk_accepted=risk.accepted,
            entry_price=risk.entry_price,
            stop_loss=risk.stop_loss,
            quantity=risk.quantity,
            position_value=risk.position_value,
            estimated_total_risk=risk.estimated_total_risk,
            risk_reason=risk.reason,
            paper_trading_only=True,
            real_order_placed=False,
            reason=decision.reason,
        )


if __name__ == "__main__":

    pipeline = StrategyPipeline()

    sample_prediction = {
        "symbol": "RELIANCE",
        "prediction": "UP",
        "probabilities": {
            "DOWN": 0.20,
            "SIDEWAYS": 0.15,
            "UP": 0.65,
        },
        "candle_timestamp": "2026-11-04 10:00:00",
        "prediction_horizon_minutes": 30,
        "model_version": "mosaic_candidate_no_macd_v1",
        "status": "EXPERIMENTAL",
    }

    result = pipeline.evaluate(
        prediction=sample_prediction,
        entry_price=1500.0,
    )

    print("\nMOSAIC 2.0 - STRATEGY PIPELINE")
    print("=" * 65)

    for key, value in result.to_dict().items():
        print(f"{key}: {value}")

    print("=" * 65)
    print("PAPER TRADING ONLY")
    print("No real broker orders were placed.")
