
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/reliance_1min_clean.csv")
OUTPUT_FILE = Path("data/processed/reliance_5min.csv")


def main():
    print("\nMOSAIC 2.0 - 5-MINUTE AGGREGATION")
    print("=" * 55)

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Input 1-minute records:", len(df))

    # Calculate minutes elapsed since 09:15.
    minutes_since_open = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 15)
    )

    df["trading_date"] = df["timestamp"].dt.date
    df["slot"] = minutes_since_open // 5
    df["minute_offset"] = minutes_since_open % 5

    # Only regular-session timestamps are eligible.
    df = df[
        (minutes_since_open >= 0)
        & (minutes_since_open < 375)
    ].copy()

    candles = []

    # Process each trading date and five-minute slot.
    for (trading_date, slot), group in df.groupby(
        ["trading_date", "slot"]
    ):

        group = group.sort_values("timestamp")

        # Every candle must contain exactly five consecutive minutes.
        if len(group) != 5:
            continue

        expected_offsets = [0, 1, 2, 3, 4]

        if group["minute_offset"].tolist() != expected_offsets:
            continue

        candle = {
            "timestamp": group["timestamp"].iloc[0],
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": group["volume"].sum()
        }

        candles.append(candle)

    result = pd.DataFrame(candles)

    result = result.sort_values("timestamp").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nAGGREGATION SUMMARY")
    print("-" * 55)

    print("Input 1-minute records:", len(df))
    print("Output 5-minute candles:", len(result))
    print("Discarded incomplete candles:", 
          (len(df) - len(result) * 5))

    print("\nDate range:")
    print(result["timestamp"].min())
    print(result["timestamp"].max())

    print("\nMissing values:")
    print(result.isnull().sum())

    print("\nDuplicate timestamps:")
    print(result["timestamp"].duplicated().sum())

    print("\nFirst 5 candles:")
    print(result.head().to_string(index=False))

    print("\nCandles per day:")
    print(result.groupby(result["timestamp"].dt.date).size().describe())

    print("\nSaved to:", OUTPUT_FILE)
    print("\n5-MINUTE AGGREGATION COMPLETED")


if __name__ == "__main__":
    main()
