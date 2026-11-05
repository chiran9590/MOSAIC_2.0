
import pandas as pd

from src.inference.live_candle_aggregator import (
    aggregate_completed_5min
)


def make_minutes(start="2026-11-04 09:15:00"):
    timestamps = pd.date_range(
        start=start,
        periods=5,
        freq="1min"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100, 101, 102, 103, 104],
        "high": [102, 103, 104, 105, 106],
        "low": [99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume": [100, 200, 300, 400, 500]
    })


def run_test(name, function):
    try:
        function()
        print(f"{name:45} [PASS]")
    except Exception as error:
        print(f"{name:45} [FAIL]")
        raise error


def test_complete_candle():

    data = make_minutes()

    result = aggregate_completed_5min(
        data,
        now="2026-11-04 09:20:00"
    )

    assert len(result) == 1
    assert result.iloc[0]["open"] == 100
    assert result.iloc[0]["high"] == 106
    assert result.iloc[0]["low"] == 99
    assert result.iloc[0]["close"] == 105
    assert result.iloc[0]["volume"] == 1500


def test_incomplete_candle():

    data = make_minutes()

    result = aggregate_completed_5min(
        data,
        now="2026-11-04 09:19:59"
    )

    assert len(result) == 0


def test_missing_minute():

    data = make_minutes()
    data = data.drop(index=2).reset_index(drop=True)

    result = aggregate_completed_5min(
        data,
        now="2026-11-04 09:20:00"
    )

    assert len(result) == 0


def test_duplicate_minute():

    data = make_minutes()

    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True
    )

    try:
        aggregate_completed_5min(
            data,
            now="2026-11-04 09:20:00"
        )
    except ValueError:
        return

    raise AssertionError("Duplicate timestamp was not rejected")


def test_future_minute():

    data = make_minutes()

    future_row = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-11-04 09:20:00"),
        "open": 105,
        "high": 107,
        "low": 104,
        "close": 106,
        "volume": 600
    }])

    data = pd.concat(
        [data, future_row],
        ignore_index=True
    )

    result = aggregate_completed_5min(
        data,
        now="2026-11-04 09:20:00"
    )

    assert len(result) == 1
    assert result.iloc[0]["close"] == 105
    assert result.iloc[0]["volume"] == 1500


def main():

    print("\nMOSAIC 2.0 - LIVE AGGREGATION SAFETY TEST")
    print("=" * 75)

    run_test("Complete five-minute candle", test_complete_candle)
    run_test("Incomplete five-minute candle", test_incomplete_candle)
    run_test("Missing one-minute record", test_missing_minute)
    run_test("Duplicate minute timestamp", test_duplicate_minute)
    run_test("Future/incomplete minute record", test_future_minute)

    print("=" * 75)
    print("ALL LIVE AGGREGATION SAFETY TESTS PASSED")


if __name__ == "__main__":
    main()
