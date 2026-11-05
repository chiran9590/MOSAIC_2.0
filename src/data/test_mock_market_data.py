import unittest
import pandas as pd

from src.data.mock_market_data import MockMarketDataProvider
from src.inference.live_candle_aggregator import (
    aggregate_completed_5min
)


class TestMockMarketData(unittest.TestCase):

    def setUp(self):
        self.provider = MockMarketDataProvider(seed=42)

        self.candles = self.provider.fetch_minute_candles(
            symbol="RELIANCE",
            trading_date="2026-11-03"
        )

    def test_provider_metadata(self):
        self.assertEqual(self.provider.provider_name, "MOCK")
        self.assertTrue(self.provider.is_synthetic)

    def test_candle_count(self):
        self.assertEqual(len(self.candles), 375)

    def test_required_columns(self):
        expected = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        self.assertEqual(list(self.candles.columns), expected)

    def test_session_boundaries(self):
        self.assertEqual(
            self.candles.iloc[0]["timestamp"],
            pd.Timestamp("2026-11-03 09:15:00")
        )

        self.assertEqual(
            self.candles.iloc[-1]["timestamp"],
            pd.Timestamp("2026-11-03 15:29:00")
        )

    def test_ohlc_integrity(self):
        self.assertTrue(
            (self.candles["high"] >= self.candles["open"]).all()
        )

        self.assertTrue(
            (self.candles["high"] >= self.candles["close"]).all()
        )

        self.assertTrue(
            (self.candles["low"] <= self.candles["open"]).all()
        )

        self.assertTrue(
            (self.candles["low"] <= self.candles["close"]).all()
        )

        self.assertTrue(
            (self.candles["volume"] >= 0).all()
        )

    def test_reproducibility(self):
        second_provider = MockMarketDataProvider(seed=42)

        second = second_provider.fetch_minute_candles(
            "RELIANCE",
            "2026-11-03"
        )

        pd.testing.assert_frame_equal(
            self.candles,
            second
        )

    def test_five_minute_aggregation(self):
        five_minute = aggregate_completed_5min(
            self.candles,
            now=pd.Timestamp("2026-11-03 15:30:00")
        )

        self.assertEqual(len(five_minute), 75)

        self.assertEqual(
            five_minute.iloc[0]["timestamp"],
            pd.Timestamp("2026-11-03 09:15:00")
        )

        self.assertEqual(
            five_minute.iloc[-1]["timestamp"],
            pd.Timestamp("2026-11-03 15:25:00")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
