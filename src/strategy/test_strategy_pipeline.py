import unittest

from src.strategy.strategy_pipeline import StrategyPipeline
from src.strategy.risk_manager import RiskConfig


class TestStrategyPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = StrategyPipeline()

        self.up_prediction = {
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

    def test_long_paper_candidate(self):
        result = self.pipeline.evaluate(
            self.up_prediction,
            entry_price=1500,
        )

        self.assertEqual(
            result.decision,
            "PAPER_TRADE_CANDIDATE",
        )
        self.assertEqual(result.direction, "LONG")
        self.assertTrue(result.risk_accepted)
        self.assertEqual(result.quantity, 13)
        self.assertEqual(result.stop_loss, 1485)

    def test_short_paper_candidate(self):
        prediction = self.up_prediction.copy()
        prediction["prediction"] = "DOWN"
        prediction["probabilities"] = {
            "DOWN": 0.65,
            "SIDEWAYS": 0.15,
            "UP": 0.20,
        }

        result = self.pipeline.evaluate(
            prediction,
            entry_price=1500,
        )

        self.assertEqual(
            result.decision,
            "PAPER_TRADE_CANDIDATE",
        )
        self.assertEqual(result.direction, "SHORT")
        self.assertEqual(result.stop_loss, 1515)

    def test_sideways_prediction_returns_hold(self):
        prediction = self.up_prediction.copy()
        prediction["prediction"] = "SIDEWAYS"
        prediction["probabilities"] = {
            "DOWN": 0.15,
            "SIDEWAYS": 0.65,
            "UP": 0.20,
        }

        result = self.pipeline.evaluate(
            prediction,
            entry_price=1500,
        )

        self.assertEqual(result.decision, "HOLD")
        self.assertIsNone(result.direction)
        self.assertEqual(result.quantity, 0)

    def test_missing_entry_price(self):
        result = self.pipeline.evaluate(
            self.up_prediction
        )

        self.assertEqual(
            result.decision,
            "PENDING_RISK_CHECK",
        )
        self.assertFalse(result.risk_accepted)
        self.assertEqual(result.quantity, 0)

    def test_invalid_entry_price_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline.evaluate(
                self.up_prediction,
                entry_price=0,
            )

    def test_risk_rejects_oversized_trade(self):
        small_capital_pipeline = StrategyPipeline(
            RiskConfig(
                capital=100,
                risk_per_trade_pct=0.5,
                max_position_value_pct=20,
                stop_loss_pct=1,
                estimated_round_trip_cost_pct=0.20,
            )
        )

        result = small_capital_pipeline.evaluate(
            self.up_prediction,
            entry_price=1500,
        )

        self.assertEqual(
            result.decision,
            "REJECTED_BY_RISK",
        )
        self.assertFalse(result.risk_accepted)
        self.assertEqual(result.quantity, 0)

    def test_no_real_orders_are_placed(self):
        result = self.pipeline.evaluate(
            self.up_prediction,
            entry_price=1500,
        )

        self.assertTrue(result.paper_trading_only)
        self.assertFalse(result.real_order_placed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
