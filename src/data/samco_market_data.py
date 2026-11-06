import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.market_data_base import MarketDataProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

BASE_URL = "https://tradeapi.samco.in"
INTRADAY_URL = f"{BASE_URL}/intraday/candleData"


class SamcoMarketDataProvider(MarketDataProvider):

    def __init__(self):
        self.session_token = os.getenv("SAMCO_SESSION_TOKEN")

        if not self.session_token:
            raise ValueError(
                "Missing SAMCO_SESSION_TOKEN. "
                "Generate a Samco session token first."
            )

    @property
    def provider_name(self):
        return "SAMCO"

    @property
    def is_synthetic(self):
        return False

    def fetch_minute_candles(
        self,
        symbol="RELIANCE",
        trading_date=None,
    ):

        if not symbol:
            raise ValueError("Symbol cannot be empty.")

        if not trading_date:
            raise ValueError("Trading date is required.")

        date = pd.Timestamp(trading_date).strftime("%Y-%m-%d")

        params = {
            "exchange": "NSE",
            "symbolName": symbol.upper(),
            "fromDate": f"{date} 09:15:00",
            "toDate": f"{date} 15:30:00",
            "interval": "1",
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-session-token": self.session_token,
        }

        try:
            response = requests.get(
                INTRADAY_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Samco market-data request failed: "
                f"{type(exc).__name__}"
            ) from None

        except ValueError:
            raise RuntimeError(
                "Samco returned an invalid JSON response."
            ) from None

        if result.get("status") != "Success":
            raise RuntimeError(
                "Samco market-data API error: "
                + str(result.get("statusMessage", "Unknown error"))
            )

        candle_data = result.get("intradayCandleData", [])

        if not candle_data:
            raise RuntimeError(
                f"No intraday candles returned for "
                f"{symbol} on {date}."
            )

        candles = pd.DataFrame(candle_data)

        required_mapping = {
            "dateTime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        missing = [
            column for column in required_mapping
            if column not in candles.columns
        ]

        if missing:
            raise RuntimeError(
                f"Samco response is missing columns: {missing}"
            )

        candles = candles[list(required_mapping)].rename(
            columns=required_mapping
        )

        candles["timestamp"] = pd.to_datetime(
            candles["timestamp"],
            errors="raise",
        )

        numeric_columns = [
            "open", "high", "low", "close", "volume"
        ]

        for column in numeric_columns:
            candles[column] = pd.to_numeric(
                candles[column],
                errors="raise",
            )

        candles["volume"] = candles["volume"].astype("int64")

        candles = (
            candles
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )

        if candles.empty:
            raise RuntimeError("No valid candles after normalization.")

        if (candles[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError("Samco returned invalid OHLC prices.")

        if (candles["volume"] < 0).any():
            raise RuntimeError("Samco returned invalid volume values.")

        return candles[
            self.REQUIRED_COLUMNS
        ]
