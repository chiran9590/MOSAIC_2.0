import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.paper_trading import database
from src.paper_trading.trade_store import create_paper_trade


def sample_trade():
    return {
        "symbol": "RELIANCE",
        "decision": "PAPER_TRADE_CANDIDATE",
        "direction": "LONG",
        "prediction": "UP",
        "candle_timestamp": "2026-11-06 10:00:00",
        "model_version": "api_integration_test_v1",
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


class TestMosaicApiIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "api_test.sqlite3"

        self.env_patch = patch.dict(os.environ, {"DATABASE_URL": ""})
        self.env_patch.start()

        self.db_patch = patch.object(
            database,
            "DATABASE_PATH",
            self.db_path,
        )
        self.db_patch.start()

        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db_patch.stop()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_root_endpoint_reports_experimental_status(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["name"], "MOSAIC 2.0")
        self.assertEqual(data["status"], "EXPERIMENTAL")
        self.assertFalse(data["paper_trade_creation_enabled"])
        self.assertFalse(data["live_order_execution"])

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertEqual(
            response.json()["service"],
            "MOSAIC 2.0 API",
        )

    def test_mock_market_returns_valid_synthetic_candles(self):
        response = self.client.get("/api/v1/market/mock")

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["symbol"], "RELIANCE")
        self.assertEqual(data["data_source"], "MOCK")
        self.assertTrue(data["synthetic_data"])
        self.assertEqual(data["total_candles"], 75)
        self.assertEqual(len(data["candles"]), 75)

        for candle in data["candles"]:
            self.assertGreater(candle["high"], 0)
            self.assertGreater(candle["low"], 0)
            self.assertGreater(candle["volume"], -1)

            self.assertLessEqual(
                candle["low"],
                min(candle["open"], candle["close"]),
            )
            self.assertGreaterEqual(
                candle["high"],
                max(candle["open"], candle["close"]),
            )

    def test_all_trades_endpoint_returns_empty_state(self):
        response = self.client.get("/api/v1/paper-trades")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["total"], 0)
        self.assertEqual(response.json()["trades"], [])

    def test_open_trades_endpoint_returns_empty_state(self):
        response = self.client.get("/api/v1/paper-trades/open")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["total"], 0)
        self.assertEqual(response.json()["trades"], [])

    def test_empty_performance_response(self):
        response = self.client.get(
            "/api/v1/paper-trades/performance"
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["total_trades"], 0)
        self.assertEqual(data["closed_trades"], 0)
        self.assertEqual(data["net_pnl"], 0.0)
        self.assertEqual(data["win_rate_pct"], 0.0)
        self.assertEqual(data["equity_curve"], [])
        self.assertEqual(data["execution_mode"], "PAPER_TRADING_ONLY")
        self.assertFalse(data["real_order_placed"])

    def test_trade_retrieval_and_performance_after_closure(self):
        created = create_paper_trade(
            sample_trade(),
            data_source="TEST",
            synthetic_data=True,
        )

        trade_id = created["trade_id"]

        all_response = self.client.get("/api/v1/paper-trades")
        self.assertEqual(all_response.status_code, 200)
        self.assertEqual(all_response.json()["total"], 1)

        open_response = self.client.get(
            "/api/v1/paper-trades/open"
        )
        self.assertEqual(open_response.status_code, 200)
        self.assertEqual(open_response.json()["total"], 1)

        close_response = self.client.post(
            f"/api/v1/paper-trades/{trade_id}/close",
            json={
                "exit_price": 1520.0,
                "estimated_round_trip_cost_pct": 0,
            },
        )

        self.assertEqual(close_response.status_code, 200)
        self.assertEqual(close_response.json()["net_pnl"], 200.0)

        open_after_close = self.client.get(
            "/api/v1/paper-trades/open"
        )
        self.assertEqual(open_after_close.json()["total"], 0)

        performance_response = self.client.get(
            "/api/v1/paper-trades/performance"
        )

        self.assertEqual(performance_response.status_code, 200)

        performance = performance_response.json()

        self.assertEqual(performance["total_trades"], 1)
        self.assertEqual(performance["closed_trades"], 1)
        self.assertEqual(performance["winning_trades"], 1)
        self.assertEqual(performance["net_pnl"], 200.0)
        self.assertEqual(performance["current_equity"], 100200.0)
        self.assertEqual(len(performance["equity_curve"]), 1)

    def test_trade_creation_remains_disabled(self):
        response = self.client.post("/api/v1/paper-trades")

        self.assertEqual(response.status_code, 409)
        self.assertIn(
            "experimental",
            response.json()["detail"].lower(),
        )

    def test_closing_nonexistent_trade_returns_400(self):
        response = self.client.post(
            "/api/v1/paper-trades/99999/close",
            json={"exit_price": 1500.0},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "not found",
            response.json()["detail"].lower(),
        )

    def test_invalid_close_request_returns_422(self):
        response = self.client.post(
            "/api/v1/paper-trades/1/close",
            json={"exit_price": 0},
        )

        self.assertEqual(response.status_code, 422)

    def test_invalid_live_symbol_returns_422(self):
        response = self.client.post(
            "/api/v1/predict/live",
            json={"symbol": "INVALID"},
        )

        self.assertEqual(response.status_code, 422)

    def test_live_prediction_provider_failure_returns_502(self):
        with patch(
            "src.api.main.SamcoMarketDataProvider",
            side_effect=RuntimeError("Simulated provider failure"),
        ):
            response = self.client.post(
                "/api/v1/predict/live",
                json={"symbol": "RELIANCE"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("Live Samco prediction failed", response.json()["detail"])

    def test_live_strategy_provider_failure_returns_502(self):
        with patch(
            "src.api.main.SamcoMarketDataProvider",
            side_effect=RuntimeError("Simulated provider failure"),
        ):
            response = self.client.post(
                "/api/v1/strategy/live",
                json={"symbol": "RELIANCE"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("Live Samco strategy failed", response.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
