from abc import ABC, abstractmethod
import pandas as pd


class MarketDataProvider(ABC):
    """
    Common interface for MOSAIC market-data providers.

    All providers must return one-minute OHLCV candles.
    Timestamps represent candle start times in IST.
    """

    REQUIRED_COLUMNS = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    @property
    @abstractmethod
    def provider_name(self):
        pass

    @property
    @abstractmethod
    def is_synthetic(self):
        pass

    @abstractmethod
    def fetch_minute_candles(
        self,
        symbol: str,
        trading_date: str
    ) -> pd.DataFrame:
        pass
