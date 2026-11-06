import inspect
import unittest

from fastapi.testclient import TestClient

from src.api.main import app, api_create_paper_trade


class TestPaperTradeApiSafety(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_fabricated_approved_payload_is_rejected(self):
        fabricated_payload = {
            "symbol": "RELIANCE",
            "decision": "PAPER_TRADE_CANDIDATE",
            "direction": "LONG",
            "prediction": "UP",
            "candle_timestamp": "2026-11-06 10:00:00",
            "model_version": "fabricated_model",
            "model_status": "PRODUCTION_APPROVED",
            "risk_accepted": True,
            "entry_price": 1500.0,
            "stop_loss": 1485.0,
            "quantity": 10,
            "position_value": 15000.0,
            "estimated_total_risk": 150.0,
            "probabilities": {
                "UP": 0.65,
                "DOWN": 0.20,
                "SIDEWAYS": 0.15,
            },
            "paper_trading_only": True,
            "real_order_placed": False,
        }

        response = self.client.post(
            "/api/v1/paper-trades",
            json={"strategy_result": fabricated_payload},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("experimental", response.json()["detail"].lower())

    def test_creation_without_payload_is_rejected(self):
        response = self.client.post("/api/v1/paper-trades")

        self.assertEqual(response.status_code, 409)

    def test_endpoint_does_not_accept_client_supplied_arguments(self):
        parameters = inspect.signature(api_create_paper_trade).parameters

        self.assertEqual(len(parameters), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
