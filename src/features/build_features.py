
import pandas as pd
import numpy as np
from pathlib import Path

# Input and output files
INPUT_FILE = Path("data/processed/reliance_5min.csv")
LABEL_FILE = Path("data/processed/reliance_30min_labels.csv")
OUTPUT_FILE = Path("data/processed/reliance_features.csv")


def calculate_features(df):

    df = df.copy()

    # Trading date for intraday calculations
    df["trading_date"] = df["timestamp"].dt.date

    grouped = df.groupby("trading_date", sort=False)

    # -----------------------------------------
    # 1. PRICE RETURNS
    # -----------------------------------------

    df["return_1"] = grouped["close"].transform(
        lambda x: x.pct_change(1)
    )

    df["return_3"] = grouped["close"].transform(
        lambda x: x.pct_change(3)
    )

    df["return_6"] = grouped["close"].transform(
        lambda x: x.pct_change(6)
    )

    df["return_12"] = grouped["close"].transform(
        lambda x: x.pct_change(12)
    )

    # -----------------------------------------
    # 2. MOVING AVERAGES
    # -----------------------------------------

    df["SMA_5"] = grouped["close"].transform(
        lambda x: x.rolling(5).mean()
    )

    df["SMA_10"] = grouped["close"].transform(
        lambda x: x.rolling(10).mean()
    )

    df["EMA_5"] = grouped["close"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    df["EMA_10"] = grouped["close"].transform(
        lambda x: x.ewm(span=10, adjust=False).mean()
    )

    # -----------------------------------------
    # 3. RSI (14)
    # -----------------------------------------

    def calculate_rsi(series):

        delta = series.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        # Handle periods with no losses or no movement
        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain > 0), 100
        )

        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain == 0), 50
        )

        return rsi

    df["RSI_14"] = grouped["close"].transform(
        calculate_rsi
    )

    # -----------------------------------------
    # 4. MACD
    # -----------------------------------------

    def calculate_macd(series):

        ema_12 = series.ewm(
            span=12, adjust=False
        ).mean()

        ema_26 = series.ewm(
            span=26, adjust=False
        ).mean()

        macd = ema_12 - ema_26

        signal = macd.ewm(
            span=9, adjust=False
        ).mean()

        return pd.DataFrame({
            "MACD": macd,
            "MACD_signal": signal,
            "MACD_histogram": macd - signal
        }, index=series.index)

    macd_features = grouped["close"].apply(
        calculate_macd
    )

    macd_features = macd_features.reset_index(
        level=0, drop=True
    )

    df[["MACD", "MACD_signal", "MACD_histogram"]] = (
        macd_features
    )

    # -----------------------------------------
    # 5. ATR (14)
    # -----------------------------------------

    previous_close = grouped["close"].shift(1)

    tr_1 = df["high"] - df["low"]

    tr_2 = (df["high"] - previous_close).abs()

    tr_3 = (df["low"] - previous_close).abs()

    df["true_range"] = pd.concat(
        [tr_1, tr_2, tr_3], axis=1
    ).max(axis=1)

    df["ATR_14"] = df.groupby(
        "trading_date"
    )["true_range"].transform(
        lambda x: x.rolling(14).mean()
    )

    # -----------------------------------------
    # 6. VOLUME FEATURES
    # -----------------------------------------

    df["volume_change"] = grouped["volume"].transform(
        lambda x: x.pct_change()
    )

    df["volume_avg_20"] = grouped["volume"].transform(
        lambda x: x.rolling(20).mean()
    )

    df["volume_ratio"] = (
        df["volume"] / df["volume_avg_20"]
    )

    # -----------------------------------------
    # 7. CANDLE FEATURES
    # -----------------------------------------

    df["high_low_range_pct"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    df["candle_body_pct"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    df["upper_wick_pct"] = (
        (df["high"] - df[["open", "close"]].max(axis=1))
        / df["close"]
    )

    df["lower_wick_pct"] = (
        (df[["open", "close"]].min(axis=1) - df["low"])
        / df["close"]
    )

    # -----------------------------------------
    # 8. TIME FEATURES
    # -----------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["minute"] = df["timestamp"].dt.minute

    df["time_of_day"] = (
        df["hour"] * 60 + df["minute"]
    )

    return df


def main():

    print("\nMOSAIC 2.0 - FEATURE ENGINEERING")
    print("=" * 60)

    # Check files
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not LABEL_FILE.exists():
        raise FileNotFoundError(LABEL_FILE)

    # Load complete 5-minute dataset
    candles = pd.read_csv(INPUT_FILE)

    candles["timestamp"] = pd.to_datetime(
        candles["timestamp"]
    )

    candles = candles.sort_values(
        "timestamp"
    ).drop_duplicates(
        subset=["timestamp"]
    ).reset_index(drop=True)

    print("Original 5-minute candles:", len(candles))

    # Calculate historical features before joining labels
    features = calculate_features(candles)

    # Load target labels
    labels = pd.read_csv(LABEL_FILE)

    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"]
    )

    print("Labeled records:", len(labels))

    # Keep only the target and timestamp.
    # future_return is deliberately excluded.
    labels = labels[["timestamp", "label"]]

    # Join features with target labels
    result = features.merge(
        labels,
        on="timestamp",
        how="inner",
        validate="one_to_one"
    )

    # Remove rows without sufficient historical information
    result = result.replace(
        [np.inf, -np.inf], np.nan
    )

    feature_columns = [
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "SMA_5",
        "SMA_10",
        "EMA_5",
        "EMA_10",
        "RSI_14",
        "MACD",
        "MACD_signal",
        "MACD_histogram",
        "ATR_14",
        "volume_change",
        "volume_ratio",
        "high_low_range_pct",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "hour",
        "minute",
        "time_of_day"
    ]

    result = result.dropna(
        subset=feature_columns + ["label"]
    ).copy()

    # Keep useful columns only.
    # Raw OHLCV values are retained for analysis.
    output_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ] + feature_columns + ["label"]

    result = result[output_columns]

    # Save final feature dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\nFEATURE ENGINEERING SUMMARY")
    print("-" * 60)

    print("Original candles:", len(candles))
    print("Labeled records:", len(labels))
    print("Final feature records:", len(result))
    print(
        "Records removed due to insufficient features:",
        len(labels) - len(result)
    )

    print("\nNumber of features:", len(feature_columns))

    print("\nFeature columns:")
    for column in feature_columns:
        print("-", column)

    print("\nMissing values:")
    print(result.isnull().sum().sum())

    print("\nDuplicate timestamps:")
    print(result["timestamp"].duplicated().sum())

    print("\nClass distribution:")
    print(result["label"].value_counts())

    print("\nClass percentages:")
    print(
        (result["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nFirst 5 records:")
    print(result.head().to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)

    print("\nFEATURE ENGINEERING COMPLETED")


if __name__ == "__main__":
    main()
