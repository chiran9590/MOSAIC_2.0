import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paper_trading.trade_store import (
    create_paper_trade,
    get_all_trades,
    get_open_trades,
    close_paper_trade,
)


def sample_trade(direction="LONG"):
    entry = 1500.0
    stop = 1485.0 if direction == "LONG" else 1515.0

    return {
        "symbol": "RELIANCE",
        "decision": "PAPER_TRADE_CANDIDATE",
        "direction": direction,
        "prediction": "UP" if direction == "LONG" else "DOWN",
        "candle_timestamp": "2026-11-04 10:00:00",
        "model_version": "test_model_v1",
        "risk_accepted": True,
        "entry_price": entry,
        "stop_loss": stop,
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


class TestPaperTradeStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite3"

        self.db_patch = patch(
            "src.paper_trading.database.DATABASE_PATH",
            self.db_path,
        )
        self.db_patch.start()

        from src.paper_trading.database import initialize_database
        initialize_database()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_create_trade(self):
        result = create_paper_trade(
            sample_trade(),
            data_source="MOCK",
            synthetic_data=True,
        )

        self.assertTrue(result["created"])
        self.assertEqual(len(get_all_trades()), 1)

    def test_duplicate_signal_is_not_saved_twice(self):
        first = create_paper_trade(
            sample_trade(), "MOCK", True
        )
        second = create_paper_trade(
            sample_trade(), "MOCK", True
        )

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["trade_id"], second["trade_id"])
        self.assertEqual(len(get_all_trades()), 1)

    def test_reject_unaccepted_candidate(self):
        trade = sample_trade()
        trade["risk_accepted"] = False

        with self.assertRaises(ValueError):
            create_paper_trade(trade, "MOCK", True)

    def test_get_open_trades(self):
        create_paper_trade(sample_trade(), "MOCK", True)

        self.assertEqual(len(get_open_trades()), 1)

    def test_close_long_trade(self):
        created = create_paper_trade(
            sample_trade("LONG"), "MOCK", True
        )

        result = close_paper_trade(
            created["trade_id"],
            exit_price=1520.0,
            estimated_round_trip_cost_pct=0,
        )

        self.assertEqual(result["gross_pnl"], 200.0)
        self.assertEqual(result["net_pnl"], 200.0)
        self.assertEqual(len(get_open_trades()), 0)

    def test_close_short_trade(self):
        created = create_paper_trade(
            sample_trade("SHORT"), "MOCK", True
        )

        result = close_paper_trade(
            created["trade_id"],
            exit_price=1480.0,
            estimated_round_trip_cost_pct=0,
        )

        self.assertEqual(result["gross_pnl"], 200.0)

    def test_cannot_close_trade_twice(self):
        created = create_paper_trade(
            sample_trade(), "MOCK", True
        )

        close_paper_trade(
            created["trade_id"],
            exit_price=1520.0,
            estimated_round_trip_cost_pct=0,
        )

        with self.assertRaises(ValueError):
            close_paper_trade(
                created["trade_id"],
                exit_price=1530.0,
            )

    def test_invalid_exit_price(self):
        created = create_paper_trade(
            sample_trade(), "MOCK", True
        )

        with self.assertRaises(ValueError):
            close_paper_trade(created["trade_id"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
