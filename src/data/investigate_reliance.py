
import pandas as pd
from pathlib import Path

FILE = Path("data/raw/reliance_1min.csv")


def main():

    print("\nMOSAIC 2.0 - DATA INVESTIGATION")
    print("=" * 60)

    df = pd.read_csv(FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Records outside normal trading session
    print("\n1. OUTSIDE TRADING SESSION")
    print("-" * 50)

    start = pd.Timestamp("09:15").time()
    end = pd.Timestamp("15:29").time()

    outside = df[
        (df["timestamp"].dt.time < start) |
        (df["timestamp"].dt.time > end)
    ]

    print("Total:", len(outside))
    print(outside.head(20).to_string(index=False))

    # 2. Missing minute gaps
    print("\n2. MISSING MINUTE GAPS")
    print("-" * 50)

    df["date"] = df["timestamp"].dt.date

    df["previous_timestamp"] = (
        df.groupby("date")["timestamp"].shift(1)
    )

    df["gap_minutes"] = (
        df["timestamp"] - df["previous_timestamp"]
    ).dt.total_seconds() / 60

    gaps = df[df["gap_minutes"] > 1]

    print("Total gaps:", len(gaps))

    print(
        gaps[
            ["previous_timestamp", "timestamp", "gap_minutes"]
        ].to_string(index=False)
    )

    # 3. Days with fewer than 375 records
    print("\n3. DAYS WITH FEWER THAN 375 RECORDS")
    print("-" * 50)

    daily = df.groupby("date").size()

    unusual_days = daily[daily < 375]

    print("Number of unusual days:", len(unusual_days))
    print(unusual_days.to_string())

    # 4. Zero volume records
    print("\n4. ZERO VOLUME RECORDS")
    print("-" * 50)

    zero_volume = df[df["volume"] == 0]

    print("Total:", len(zero_volume))
    print(zero_volume.head(20).to_string(index=False))

    # 5. Date distribution
    print("\n5. RECORDS BY YEAR")
    print("-" * 50)

    print(
        df.groupby(df["timestamp"].dt.year).size()
    )

    print("\nINVESTIGATION COMPLETED")


if __name__ == "__main__":
    main()
