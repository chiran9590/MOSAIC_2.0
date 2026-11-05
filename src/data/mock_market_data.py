import numpy as np
import pandas as pd

from src.data.market_data_base import MarketDataProvider


class MockMarketDataProvider(MarketDataProvider):

    def __init__(self, seed=42, initial_price=1400.0):
        self.seed = seed
        self.initial_price = initial_price

    @property
    def provider_name(self):
        return "MOCK"

    @property
    def is_synthetic(self):
        return True

    def fetch_minute_candles(
        self,
        symbol="RELIANCE",
        trading_date="2026-11-03"
    ):

        if not symbol:
            raise ValueError("Symbol cannot be empty.")

        date = pd.Timestamp(trading_date).normalize()

        # NSE regular session: 09:15 to 15:30.
        timestamps = pd.date_range(
            start=date + pd.Timedelta(hours=9, minutes=15),
            periods=375,
            freq="1min"
        )

        rng = np.random.default_rng(self.seed)

        # Generate a deterministic synthetic price path.
        returns = rng.normal(
            loc=0.00002,
            scale=0.0007,
            size=375
        )

        closes = self.initial_price * np.exp(
            np.cumsum(returns)
        )

        opens = np.empty(375)
        opens[0] = self.initial_price
        opens[1:] = closes[:-1]

        # Generate small positive upper/lower wicks.
        upper_wicks = rng.uniform(0.0001, 0.0010, 375)
        lower_wicks = rng.uniform(0.0001, 0.0010, 375)

        highs = np.maximum(opens, closes) * (
            1 + upper_wicks
        )

        lows = np.minimum(opens, closes) * (
            1 - lower_wicks
        )

        volumes = rng.integers(
            low=100,
            high=15000,
            size=375
        )

        candles = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })

        return candles
