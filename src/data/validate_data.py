
import pandas as pd
from pathlib import Path


def validate_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]

    for col in price_columns + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n========== MOSAIC 2.0 DATA VALIDATION ==========")
    print("Total rows:", len(df))

    print("\n1. Missing values:")
    print(df[required_columns].isnull().sum())

    print("\n2. Duplicate timestamps:")
    print(df["timestamp"].duplicated().sum())

    print("\n3. Invalid OHLC rows:")
    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[price_columns] <= 0).any(axis=1)
    )
    print(invalid_ohlc.sum())

    print("\n4. Invalid volume rows:")
    print((df["volume"] < 0).sum())

    print("\n5. Timestamp ordering:")
    print("Chronologically sorted:", df["timestamp"].is_monotonic_increasing)

    print("\n6. Date range:")
    print("Start:", df["timestamp"].min())
    print("End:", df["timestamp"].max())

    print("\n7. Trading-session gaps:")
    # Count gaps larger than 5 minutes within the same date.
    sorted_df = df.sort_values("timestamp")
    time_diff = sorted_df.groupby(
        sorted_df["timestamp"].dt.date
    )["timestamp"].diff()

    gaps = time_diff > pd.Timedelta(minutes=5)
    print("Gaps larger than 5 minutes:", gaps.sum())

    print("\n================================================")


if __name__ == "__main__":
    validate_data("data/raw/reliance_5min.csv")
