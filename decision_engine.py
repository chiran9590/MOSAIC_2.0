cat > src/strategy/decision_engine.py <<'PY'
from dataclasses import dataclass, asdict
from typing import Any
import math


VALID_LABELS = {"UP", "DOWN", "SIDEWAYS"}


@dataclass
class TradingDecision:
    symbol: str
    decision: str
    direction: str | None
    prediction: str
    probabilities: dict[str, float]
    candle_timestamp: str
    prediction_horizon_minutes: int
    model_version: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    """
    Converts a MOSAIC model prediction into a structured decision.

    This engine does not execute orders.

    UP       -> LONG_CANDIDATE
    DOWN     -> SHORT_CANDIDATE
    SIDEWAYS -> HOLD

    All candidates remain experimental and require risk checks
    before they can be considered for paper trading.
    """

    def evaluate(self, prediction: dict[str, Any]) -> TradingDecision:

        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be a dictionary.")

        required_fields = {
            "symbol",
            "prediction",
            "probabilities",
            "candle_timestamp",
            "prediction_horizon_minutes",
            "model_version",
            "status",
        }

        missing = required_fields - prediction.keys()

        if missing:
            raise ValueError(
                f"Missing prediction fields: {sorted(missing)}"
            )

        # --------------------------------------------------
        # 1. Validate basic fields
        # --------------------------------------------------

        symbol = str(prediction["symbol"]).strip()

        if not symbol:
            raise ValueError("Symbol cannot be empty.")

        label = str(prediction["prediction"]).upper().strip()

        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid prediction label: {label}"
            )

        timestamp = str(prediction["candle_timestamp"]).strip()

        if not timestamp:
            raise ValueError("Candle timestamp cannot be empty.")

        model_version = str(prediction["model_version"]).strip()

        if not model_version:
            raise ValueError("Model version cannot be empty.")

        model_status = str(prediction["status"]).upper().strip()

        # --------------------------------------------------
        # 2. Validate probabilities
        # --------------------------------------------------

        probabilities = prediction["probabilities"]

        if not isinstance(probabilities, dict):
            raise ValueError(
                "Probabilities must be a dictionary."
            )

        if set(probabilities.keys()) != VALID_LABELS:
            raise ValueError(
                "Probabilities must contain exactly "
                "UP, DOWN and SIDEWAYS."
            )

        clean_probabilities = {}

        for key, value in probabilities.items():

            try:
                probability = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid probability for {key}."
                )

            if (
                not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    f"Probability for {key} must be finite "
                    "and between 0 and 1."
                )

            clean_probabilities[key] = probability

        total_probability = sum(
            clean_probabilities.values()
        )

        if abs(total_probability - 1.0) > 0.001:
            raise ValueError(
                "Probabilities must sum to 1. "
                f"Received {total_probability:.6f}."
            )

        # The predicted label must have the highest probability.
        # Ties are allowed.
        highest_probability = max(
            clean_probabilities.values()
        )

        if clean_probabilities[label] < highest_probability:
            raise ValueError(
                "Prediction label does not match "
                "the highest probability."
            )

        # --------------------------------------------------
        # 3. Validate prediction horizon
        # --------------------------------------------------

        try:
            horizon_raw = prediction[
                "prediction_horizon_minutes"
            ]

            horizon = int(horizon_raw)

            if isinstance(horizon_raw, bool):
                raise ValueError

            if float(horizon_raw) != horizon:
                raise ValueError

        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Prediction horizon must be a positive integer."
            )

        if horizon <= 0:
            raise ValueError(
                "Prediction horizon must be positive."
            )

        # --------------------------------------------------
        # 4. Check model status
        # --------------------------------------------------

        if model_status != "EXPERIMENTAL":

            return TradingDecision(
                symbol=symbol,
                decision="NO_TRADE",
                direction=None,
                prediction=label,
                probabilities=clean_probabilities,
                candle_timestamp=timestamp,
                prediction_horizon_minutes=horizon,
                model_version=model_version,
                status=model_status,
                reason=(
                    "Model status is not approved for "
                    "experimental decision evaluation."
                ),
            )

        # --------------------------------------------------
        # 5. Convert prediction into a decision
        # --------------------------------------------------

        if label == "UP":

            decision = "LONG_CANDIDATE"
            direction = "LONG"

            reason = (
                "Model predicts UP. This is a directional "
                "candidate only. Risk checks are still required."
            )

        elif label == "DOWN":

            decision = "SHORT_CANDIDATE"
            direction = "SHORT"

            reason = (
                "Model predicts DOWN. This is a directional "
                "candidate only. Risk checks are still required."
            )

        else:

            decision = "HOLD"
            direction = None

            reason = (
                "Model predicts a sideways market. "
                "No directional candidate generated."
            )

        return TradingDecision(
            symbol=symbol,
            decision=decision,
            direction=direction,
            prediction=label,
            probabilities=clean_probabilities,
            candle_timestamp=timestamp,
            prediction_horizon_minutes=horizon,
            model_version=model_version,
            status="EXPERIMENTAL",
            reason=reason,
        )


if __name__ == "__main__":

    engine = DecisionEngine()

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

    result = engine.evaluate(sample_prediction)

    print("\nMOSAIC 2.0 - DECISION ENGINE TEST")
    print("=" * 60)

    for key, value in result.to_dict().items():
        print(f"{key}: {value}")

    print("=" * 60)
    print("No real orders were placed.")
PY
