
import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/raw/reliance_1min.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "reliance_1min_clean.csv"


SPECIAL_SESSIONS = [
    "2022-10-24",
    "2023-11-12",
    "2024-11-01",
    "2025-10-21",
    "2024-03-02",
    "2024-05-18",
]


def main():

    print("\nMOSAIC 2.0 - DATA CLEANING")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    original_count = len(df)

    print("Original records:", original_count)

    # 1. Remove duplicate timestamps.
    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # 2. Remove special trading sessions.
    special_dates = pd.to_datetime(SPECIAL_SESSIONS).date

    df = df[
        ~df["timestamp"].dt.date.isin(special_dates)
    ].copy()

    # 3. Keep regular NSE trading hours.
    start_time = pd.Timestamp("09:15").time()
    end_time = pd.Timestamp("15:29").time()

    df = df[
        (df["timestamp"].dt.time >= start_time) &
        (df["timestamp"].dt.time <= end_time)
    ].copy()

    # 4. Remove zero or negative volume.
    df = df[df["volume"] > 0].copy()

    # 5. Remove invalid OHLC records.
    valid_ohlc = (
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    )

    df = df[valid_ohlc].copy()

    # 6. Sort by timestamp.
    df = df.sort_values("timestamp")

    df = df.reset_index(drop=True)

    # 7. Save cleaned data.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCLEANING SUMMARY")
    print("-" * 45)

    print("Original records:", original_count)
    print("Cleaned records:", len(df))
    print("Removed records:", original_count - len(df))

    print("\nDate range:")
    print(df["timestamp"].min())
    print(df["timestamp"].max())

    print("\nRemaining zero-volume records:")
    print((df["volume"] <= 0).sum())

    print("\nRemaining duplicate timestamps:")
    print(df["timestamp"].duplicated().sum())

    print("\nSaved to:", OUTPUT_FILE)

    print("\nDATA CLEANING COMPLETED")


if __name__ == "__main__":
    main()
