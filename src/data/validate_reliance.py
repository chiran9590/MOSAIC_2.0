
import pandas as pd
from pathlib import Path

FILE = Path("data/raw/reliance_1min.csv")


def main():

    print("\nMOSAIC 2.0 - DATA VALIDATION")
    print("=" * 55)

    if not FILE.exists():
        print("Dataset not found:", FILE)
        return

    df = pd.read_csv(FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    print("\n1. BASIC INFORMATION")
    print("-" * 40)

    print("Total records:", len(df))
    print("Columns:", list(df.columns))
    print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

    print("\n2. MISSING VALUES")
    print("-" * 40)

    print(df[required_columns].isnull().sum())

    print("\n3. DUPLICATE TIMESTAMPS")
    print("-" * 40)

    print(
        "Duplicates:",
        df["timestamp"].duplicated().sum()
    )

    print("\n4. INVALID OHLC RECORDS")
    print("-" * 40)

    invalid_ohlc = df[
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    ]

    print("Invalid OHLC rows:", len(invalid_ohlc))

    print("\n5. INVALID VOLUME")
    print("-" * 40)

    print(
        "Negative volume rows:",
        (df["volume"] < 0).sum()
    )

    print("\n6. TRADING SESSION CHECK")
    print("-" * 40)

    outside_session = df[
        (df["timestamp"].dt.time < pd.Timestamp("09:15").time()) |
        (df["timestamp"].dt.time > pd.Timestamp("15:29").time())
    ]

    print("Outside 09:15–15:29:", len(outside_session))

    print("\n7. RECORDS PER TRADING DAY")
    print("-" * 40)

    daily_counts = df.groupby(
        df["timestamp"].dt.date
    ).size()

    print(daily_counts.describe())

    print("\n8. MINUTE GAPS")
    print("-" * 40)

    df = df.sort_values("timestamp")

    gaps = df.groupby(
        df["timestamp"].dt.date
    )["timestamp"].diff()

    unusual_gaps = gaps[
        (gaps > pd.Timedelta(minutes=1)) &
        gaps.notna()
    ]

    print("Gaps greater than one minute:", len(unusual_gaps))

    print("\n9. PRICE SUMMARY")
    print("-" * 40)

    print(df[["open", "high", "low", "close", "volume"]].describe())

    print("\nVALIDATION COMPLETED")


if __name__ == "__main__":
    main()
