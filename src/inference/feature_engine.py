
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable when running scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import calculate_features


FEATURES_PATH = (
    PROJECT_ROOT
    / "models/saved/mosaic_candidate_no_macd_v1_features.json"
)

import json

with open(FEATURES_PATH, "r") as file:
    REQUIRED_FEATURES = json.load(file)


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]


class FeatureEngineError(Exception):
    """Raised when incoming candle data is invalid."""


def validate_candles(candles: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(candles, pd.DataFrame):
        raise FeatureEngineError(
            "Input must be a pandas DataFrame."
        )

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in candles.columns
    ]

    if missing:
        raise FeatureEngineError(
            f"Missing candle columns: {missing}"
        )

    df = candles[REQUIRED_COLUMNS].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise FeatureEngineError(
            "Invalid candle timestamps detected."
        )

    # The historical training data uses naive IST timestamps.
    # Do not silently mix timezone-aware and naive timestamps.
    if df["timestamp"].dt.tz is not None:
        raise FeatureEngineError(
            "Expected timezone-naive IST timestamps, "
            "matching the historical training dataset."
        )

    for column in [
        "open", "high", "low", "close", "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[REQUIRED_COLUMNS[1:]].isna().any().any():
        raise FeatureEngineError(
            "Missing or invalid OHLCV values."
        )

    if not np.isfinite(
        df[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    ).all():
        raise FeatureEngineError(
            "Infinite OHLCV values detected."
        )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise FeatureEngineError(
            "Prices must be positive."
        )

    if (df["volume"] < 0).any():
        raise FeatureEngineError(
            "Volume cannot be negative."
        )

    if (df["high"] < df["low"]).any():
        raise FeatureEngineError(
            "High cannot be lower than low."
        )

    if (
        df["high"] <
        df[["open", "close"]].max(axis=1)
    ).any():
        raise FeatureEngineError(
            "High is inconsistent with open/close."
        )

    if (
        df["low"] >
        df[["open", "close"]].min(axis=1)
    ).any():
        raise FeatureEngineError(
            "Low is inconsistent with open/close."
        )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise FeatureEngineError(
            "Duplicate candle timestamps detected."
        )

    # Validate regular-session 5-minute candle start times.
    minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
    )

    minutes_since_open = minutes - (9 * 60 + 15)

    valid_session = (
        (minutes_since_open >= 0)
        & (minutes_since_open < 375)
        & (minutes_since_open % 5 == 0)
        & (df["timestamp"].dt.second == 0)
    )

    if not valid_session.all():
        raise FeatureEngineError(
            "Invalid NSE regular-session 5-minute timestamp."
        )

    return df


def calculate_live_features(
    candles: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the exact historical training features
    from completed 5-minute OHLCV candles.

    Important:
    - Input candles must already be completed.
    - Features reset at each trading date.
    - No future data or labels are used.
    """

    df = validate_candles(candles)

    if df.empty:
        raise FeatureEngineError(
            "No candle data received."
        )

    # Reuse the exact historical feature implementation.
    result = calculate_features(df)

    result = result.replace(
        [np.inf, -np.inf], np.nan
    )

    missing_features = [
        feature for feature in REQUIRED_FEATURES
        if feature not in result.columns
    ]

    if missing_features:
        raise FeatureEngineError(
            f"Feature calculation failed: {missing_features}"
        )

    return result


def get_latest_features(
    candles: pd.DataFrame
) -> pd.DataFrame:
    """
    Return the most recent row with all 19 valid features.
    """

    result = calculate_live_features(candles)

    valid = result.dropna(
        subset=REQUIRED_FEATURES
    )

    if valid.empty:
        raise FeatureEngineError(
            "Insufficient historical candles to calculate "
            "all 19 features."
        )

    latest = valid.iloc[[-1]].copy()

    values = latest[REQUIRED_FEATURES].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise FeatureEngineError(
            "Latest features contain invalid values."
        )

    return latest[
        ["timestamp"] + REQUIRED_FEATURES
    ].reset_index(drop=True)
