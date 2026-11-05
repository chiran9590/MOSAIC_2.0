
import pandas as pd

from src.inference.live_candle_aggregator import (
    aggregate_completed_5min
)

MINUTE_FILE = "data/processed/reliance_1min_clean.csv"
HISTORICAL_FILE = "data/processed/reliance_5min.csv"


def main():

    print("\nMOSAIC 2.0 - LIVE AGGREGATION PARITY TEST")
    print("=" * 75)

    minute_data = pd.read_csv(MINUTE_FILE)
    historical = pd.read_csv(HISTORICAL_FILE)

    minute_data["timestamp"] = pd.to_datetime(
        minute_data["timestamp"]
    )

    historical["timestamp"] = pd.to_datetime(
        historical["timestamp"]
    )

    print(f"1-minute records: {len(minute_data)}")
    print(f"Historical 5-minute records: {len(historical)}")

    # Set evaluation time after the final historical candle.
    evaluation_time = (
        minute_data["timestamp"].max()
        + pd.Timedelta(minutes=2)
    )

    generated = aggregate_completed_5min(
        minute_data,
        now=evaluation_time
    )

    print(f"Generated 5-minute records: {len(generated)}")

    # Compare timestamps and OHLCV.
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    comparison = historical[columns].merge(
        generated[columns],
        on="timestamp",
        how="outer",
        suffixes=("_historical", "_generated"),
        indicator=True
    )

    missing = comparison[
        comparison["_merge"] != "both"
    ]

    print("\nRECORD COMPARISON")
    print("-" * 75)

    print(f"Comparable records: {(comparison['_merge'] == 'both').sum()}")
    print(f"Missing/extra records: {len(missing)}")

    if not missing.empty:
        print("\nFirst mismatched timestamps:")
        print(missing[["timestamp", "_merge"]].head(10))

        raise AssertionError(
            "Historical and generated candle timestamps differ."
        )

    print("\nOHLCV COMPARISON")
    print("-" * 75)

    for col in ["open", "high", "low", "close", "volume"]:

        difference = (
            comparison[f"{col}_historical"]
            - comparison[f"{col}_generated"]
        ).abs()

        max_difference = difference.max()

        passed = difference.le(1e-9).all()

        print(
            f"{col:12} Maximum difference: "
            f"{max_difference:.12f} "
            f"[{'PASS' if passed else 'FAIL'}]"
        )

        if not passed:
            raise AssertionError(
                f"Aggregation mismatch in {col}"
            )

    print("\n" + "=" * 75)
    print("LIVE AGGREGATION PARITY TEST PASSED")
    print("Generated candles match historical OHLCV data.")
    print("=" * 75)


if __name__ == "__main__":
    main()
