
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("data/processed/reliance_5min.csv")
OUTPUT_FILE = Path("data/processed/reliance_30min_labels.csv")

# Label configuration
THRESHOLD = 0.0015  # 0.15%
HORIZON_MINUTES = 30
HORIZON_CANDLES = 6


def create_labels():

    print("\nMOSAIC 2.0 - LABEL GENERATION")
    print("=" * 55)

    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Load data
    df = pd.read_csv(INPUT_FILE)

    print("Original 5-minute candles:", len(df))

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    # Remove invalid records
    df = df.dropna(
        subset=["timestamp", "close"]
    )

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(
        subset=["timestamp"]
    ).reset_index(drop=True)

    # Identify trading day
    df["trading_date"] = df["timestamp"].dt.date

    # Calculate future closing price
    df["future_close"] = (
        df.groupby("trading_date")["close"]
        .shift(-HORIZON_CANDLES)
    )

    # Calculate future timestamp
    df["future_timestamp"] = (
        df.groupby("trading_date")["timestamp"]
        .shift(-HORIZON_CANDLES)
    )

    # Validate exact 30-minute prediction horizon
    valid_horizon = (
        df["future_timestamp"] - df["timestamp"]
        == pd.Timedelta(minutes=HORIZON_MINUTES)
    )

    df = df[valid_horizon].copy()

    # Calculate future percentage return
    df["future_return"] = (
        df["future_close"] / df["close"] - 1
    )

    # Assign labels
    df["label"] = "SIDEWAYS"

    df.loc[
        df["future_return"] > THRESHOLD,
        "label"
    ] = "UP"

    df.loc[
        df["future_return"] < -THRESHOLD,
        "label"
    ] = "DOWN"

    # Remove temporary columns
    df = df.drop(
        columns=[
            "trading_date",
            "future_close",
            "future_timestamp"
        ]
    )

    # Save labeled dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    # Display results
    print("\nLABELING SUMMARY")
    print("-" * 55)

    print("Original candles:", len(pd.read_csv(INPUT_FILE)))
    print("Valid labeled records:", len(df))

    print(
        "Removed records:",
        len(pd.read_csv(INPUT_FILE)) - len(df)
    )

    print("\nClass counts:")
    print(
        df["label"].value_counts()
        .reindex(["UP", "DOWN", "SIDEWAYS"], fill_value=0)
    )

    print("\nClass percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .reindex(["UP", "DOWN", "SIDEWAYS"], fill_value=0)
        .round(2)
    )

    print("\nFuture return statistics:")
    print(df["future_return"].describe())

    print("\nFirst 10 labeled records:")
    print(df.head(10).to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)

    print("\nLABEL GENERATION COMPLETED")


if __name__ == "__main__":
    create_labels()
