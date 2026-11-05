
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]


def normalize_timestamp(value):
    """
    Convert timestamps to timezone-naive Indian Standard Time.
    Historical MOSAIC data uses naive IST timestamps.
    """
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(IST).tz_localize(None)

    return timestamp


def aggregate_completed_5min(minute_data, now=None):
    """
    Convert 1-minute OHLCV candles into completed 5-minute candles.

    Assumptions:
    - Each 1-minute candle timestamp represents its starting time.
    - All timestamps are interpreted in IST.
    - A 1-minute candle is complete only after its ending time.
    - A 5-minute candle is emitted only after its ending time.
    - All five expected minute candles must exist.
    """

    if not isinstance(minute_data, pd.DataFrame):
        raise TypeError("minute_data must be a pandas DataFrame")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in minute_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if minute_data.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = minute_data[REQUIRED_COLUMNS].copy()

    # Normalize timestamps.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise"
    )

    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_convert(IST)
            .dt.tz_localize(None)
        )

    # Validate numeric OHLCV data.
    numeric_columns = [
        "open", "high", "low", "close", "volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[numeric_columns].isna().any().any():
        raise ValueError("OHLCV data contains missing values")

    if not df[numeric_columns].apply(
        lambda col: col.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
    ).all().all():
        raise ValueError("OHLCV data contains non-finite values")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")

    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("Invalid high price")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("Invalid low price")

    if (df["high"] < df["low"]).any():
        raise ValueError("High cannot be below low")

    # Minute candles must start on exact minute boundaries.
    if (
        (df["timestamp"].dt.second != 0).any()
        or (df["timestamp"].dt.microsecond != 0).any()
    ):
        raise ValueError("Minute timestamps must have zero seconds")

    # Reject duplicate minute timestamps.
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate minute timestamps detected")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Determine current IST time.
    if now is None:
        current_time = pd.Timestamp(datetime.now(IST))
    else:
        current_time = pd.Timestamp(now)

    current_time = normalize_timestamp(current_time)

    # A 1-minute candle is complete only when timestamp + 1 minute <= now.
    df = df[
        df["timestamp"] + pd.Timedelta(minutes=1) <= current_time
    ].copy()

    # NSE regular session: 09:15 to 15:30.
    minutes_since_open = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 15)
    )

    df = df[
        (minutes_since_open >= 0)
        & (minutes_since_open < 375)
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Calculate 5-minute bucket start.
    minutes_since_open = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 15)
    )

    bucket_number = minutes_since_open // 5

    df["bucket_timestamp"] = (
        df["timestamp"].dt.normalize()
        + pd.Timedelta(hours=9, minutes=15)
        + pd.to_timedelta(bucket_number * 5, unit="m")
    )

    results = []

    for bucket_time, group in df.groupby("bucket_timestamp"):

        group = group.sort_values("timestamp")

        expected_timestamps = pd.date_range(
            start=bucket_time,
            periods=5,
            freq="1min"
        )

        actual_timestamps = pd.DatetimeIndex(
            group["timestamp"]
        )

        # Reject incomplete buckets and missing minutes.
        if not actual_timestamps.equals(expected_timestamps):
            continue

        candle_end = bucket_time + pd.Timedelta(minutes=5)

        # Never emit a candle before its close time.
        if candle_end > current_time:
            continue

        results.append({
            "timestamp": bucket_time,
            "open": group.iloc[0]["open"],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group.iloc[-1]["close"],
            "volume": group["volume"].sum()
        })

    return pd.DataFrame(
        results,
        columns=REQUIRED_COLUMNS
    ).reset_index(drop=True)
