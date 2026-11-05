
import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = "data/raw/reliance_5min.csv"
OUTPUT_FILE = "data/processed/reliance_features.csv"


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def create_features():

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    numeric_columns = [
        "open", "high", "low", "close", "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Calculate features separately for each trading day.
    grouped = df.groupby(
        df["timestamp"].dt.date,
        sort=False,
    )

    # 1. Returns
    df["return_1"] = grouped["close"].pct_change(1)
    df["return_3"] = grouped["close"].pct_change(3)
    df["return_6"] = grouped["close"].pct_change(6)

    # 2. Moving averages
    df["sma_5"] = grouped["close"].transform(
        lambda x: x.rolling(5).mean()
    )

    df["sma_10"] = grouped["close"].transform(
        lambda x: x.rolling(10).mean()
    )

    df["sma_20"] = grouped["close"].transform(
        lambda x: x.rolling(20).mean()
    )

    # Relative distance from moving averages
    df["close_sma5_ratio"] = df["close"] / df["sma_5"] - 1
    df["close_sma10_ratio"] = df["close"] / df["sma_10"] - 1
    df["close_sma20_ratio"] = df["close"] / df["sma_20"] - 1

    # 3. Exponential moving averages
    df["ema_5"] = grouped["close"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    df["ema_10"] = grouped["close"].transform(
        lambda x: x.ewm(span=10, adjust=False).mean()
    )

    df["ema_ratio"] = df["ema_5"] / df["ema_10"] - 1

    # 4. RSI
    df["rsi_14"] = grouped["close"].transform(
        lambda x: calculate_rsi(x, 14)
    )

    # 5. Volatility
    df["volatility_5"] = grouped["return_1"].transform(
        lambda x: x.rolling(5).std()
    )

    df["volatility_10"] = grouped["return_1"].transform(
        lambda x: x.rolling(10).std()
    )

    # 6. Volume features
    df["volume_change"] = grouped["volume"].transform(
        lambda x: x.pct_change()
    )

    df["volume_sma_10"] = grouped["volume"].transform(
        lambda x: x.rolling(10).mean()
    )

    df["relative_volume"] = (
        df["volume"] / df["volume_sma_10"]
    )

    # 7. Candle characteristics
    df["high_low_range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    df["open_close_change"] = (
        df["close"] - df["open"]
    ) / df["open"]

    # 8. Time information
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # Remove rows where rolling indicators are unavailable.
    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna().reset_index(drop=True)

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nMOSAIC 2.0 — FEATURE ENGINEERING")
    print("--------------------------------")
    print("Rows:", len(df))
    print("Features:", len(df.columns))
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    create_features()
