import unittest

from src.strategy.decision_engine import DecisionEngine


class TestDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine()

    def make_prediction(self, label="UP"):
        probabilities = {
            "UP": 0.60,
            "DOWN": 0.25,
            "SIDEWAYS": 0.15,
        }

        return {
            "symbol": "RELIANCE",
            "prediction": label,
            "probabilities": probabilities,
            "candle_timestamp": "2026-11-04 10:00:00",
            "prediction_horizon_minutes": 30,
            "model_version": "mosaic_candidate_no_macd_v1",
            "status": "EXPERIMENTAL",
        }

    def test_up_prediction(self):
        result = self.engine.evaluate(
            self.make_prediction("UP")
        )
        self.assertEqual(result.decision, "LONG_CANDIDATE")

    def test_down_prediction(self):
        result = self.engine.evaluate(
            self.make_prediction("DOWN")
        )
        self.assertEqual(result.decision, "SHORT_CANDIDATE")

    def test_sideways_prediction(self):
        result = self.engine.evaluate(
            self.make_prediction("SIDEWAYS")
        )
        self.assertEqual(result.decision, "HOLD")

    def test_invalid_probability_sum(self):
        prediction = self.make_prediction()
        prediction["probabilities"]["UP"] = 0.90

        with self.assertRaises(ValueError):
            self.engine.evaluate(prediction)

    def test_missing_field(self):
        prediction = self.make_prediction()
        del prediction["model_version"]

        with self.assertRaises(ValueError):
            self.engine.evaluate(prediction)

    def test_invalid_label(self):
        prediction = self.make_prediction()
        prediction["prediction"] = "UNKNOWN"

        with self.assertRaises(ValueError):
            self.engine.evaluate(prediction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
