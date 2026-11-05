
import os
import pandas as pd
import numpy as np

# ============================================================
# MOSAIC 2.0 - BACKTEST TIMING AUDIT
# ============================================================

print("\nMOSAIC 2.0 - BACKTEST TIMING AUDIT")
print("=" * 70)

PREDICTIONS_PATH = "reports/final_test_predictions.csv"
CANDLES_PATH = "data/processed/reliance_5min.csv"

OUTPUT_PATH = "reports/backtest_timing_audit.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["timestamp"]
)

candles = pd.read_csv(
    CANDLES_PATH,
    parse_dates=["timestamp"]
)

candles = candles.sort_values(
    "timestamp"
).reset_index(drop=True)

predictions = predictions.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Predictions:", len(predictions))
print("Candles:", len(candles))

# ============================================================
# 2. VALIDATE DATA
# ============================================================

required_candle_columns = [
    "timestamp",
    "open",
    "close"
]

for col in required_candle_columns:
    if col not in candles.columns:
        raise ValueError(f"Missing candle column: {col}")

if candles["timestamp"].duplicated().any():
    raise ValueError("Duplicate candle timestamps found.")

if predictions["timestamp"].duplicated().any():
    raise ValueError("Duplicate prediction timestamps found.")

# Create lookup dictionaries.

open_prices = candles.set_index(
    "timestamp"
)["open"].to_dict()

close_prices = candles.set_index(
    "timestamp"
)["close"].to_dict()

candle_timestamps = set(
    candles["timestamp"]
)

# ============================================================
# 3. CHECK FUTURE RETURN ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("FUTURE RETURN ALIGNMENT")
print("=" * 70)

audit = predictions[
    ["timestamp", "predicted_label", "future_return"]
].copy()

audit["future_timestamp"] = (
    audit["timestamp"] + pd.Timedelta(minutes=30)
)

audit["entry_timestamp"] = (
    audit["timestamp"] + pd.Timedelta(minutes=5)
)

audit["exit_timestamp"] = (
    audit["entry_timestamp"] + pd.Timedelta(minutes=30)
)

audit["current_close"] = audit["timestamp"].map(
    close_prices
)

audit["future_close"] = audit["future_timestamp"].map(
    close_prices
)

audit["reconstructed_future_return"] = (
    audit["future_close"] / audit["current_close"]
) - 1

audit["return_difference"] = (
    audit["future_return"]
    - audit["reconstructed_future_return"]
)

valid_alignment = audit[
    audit["reconstructed_future_return"].notna()
    & audit["future_return"].notna()
]

if valid_alignment.empty:
    raise ValueError(
        "No records available for future-return alignment check."
    )

max_difference = (
    valid_alignment["return_difference"].abs().max()
)

mean_difference = (
    valid_alignment["return_difference"].abs().mean()
)

mismatches = (
    valid_alignment["return_difference"].abs() > 1e-8
).sum()

print("Comparable records:", len(valid_alignment))
print(f"Maximum absolute difference: {max_difference:.12f}")
print(f"Mean absolute difference:    {mean_difference:.12f}")
print("Mismatched records:", mismatches)

# ============================================================
# 4. CHECK ENTRY AND EXIT PRICE AVAILABILITY
# ============================================================

print("\n" + "=" * 70)
print("ENTRY AND EXIT PRICE AVAILABILITY")
print("=" * 70)

audit["signal_exists"] = (
    audit["timestamp"].isin(candle_timestamps)
)

audit["entry_exists"] = (
    audit["entry_timestamp"].isin(candle_timestamps)
)

audit["exit_exists"] = (
    audit["exit_timestamp"].isin(candle_timestamps)
)

audit["entry_price"] = (
    audit["entry_timestamp"].map(open_prices)
)

audit["exit_price"] = (
    audit["exit_timestamp"].map(open_prices)
)

audit["same_trading_day"] = (
    audit["timestamp"].dt.date
    == audit["exit_timestamp"].dt.date
)

def classify_price_status(row):

    if not row["signal_exists"]:
        return "Missing signal candle"

    if not row["entry_exists"] and not row["exit_exists"]:
        return "Missing entry and exit"

    if not row["entry_exists"]:
        return "Missing entry"

    if not row["exit_exists"]:
        return "Missing exit"

    if not row["same_trading_day"]:
        return "Crosses trading day"

    return "Valid"

audit["price_status"] = audit.apply(
    classify_price_status,
    axis=1
)

print("\nPrice availability:")

print(
    audit["price_status"].value_counts()
)

# ============================================================
# 5. ANALYSE MISSING PRICE TIMESTAMPS
# ============================================================

print("\n" + "=" * 70)
print("MISSING PRICE ANALYSIS")
print("=" * 70)

missing = audit[
    audit["price_status"] != "Valid"
].copy()

if len(missing) > 0:

    print("\nFirst 20 missing records:")

    print(
        missing[
            [
                "timestamp",
                "predicted_label",
                "entry_timestamp",
                "exit_timestamp",
                "price_status"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("No missing prices found.")

# ============================================================
# 6. ANALYSE TRADING-DAY BOUNDARIES
# ============================================================

print("\n" + "=" * 70)
print("TRADING DAY BOUNDARY ANALYSIS")
print("=" * 70)

candles["trading_date"] = (
    candles["timestamp"].dt.date
)

daily_candles = (
    candles.groupby("trading_date")
    .agg(
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        candle_count=("timestamp", "count")
    )
)

print("\nCandle statistics:")

print(
    daily_candles["candle_count"].describe()
)

print("\nDays with fewer than 75 candles:")

print(
    daily_candles[
        daily_candles["candle_count"] < 75
    ].head(20).to_string()
)

# ============================================================
# 7. CHECK TRADE ELIGIBILITY
# ============================================================

print("\n" + "=" * 70)
print("DIRECTIONAL TRADE ELIGIBILITY")
print("=" * 70)

directional = audit[
    audit["predicted_label"].isin(["UP", "DOWN"])
].copy()

eligible = directional[
    directional["price_status"] == "Valid"
].copy()

print("Directional signals:", len(directional))
print("Signals with valid prices:", len(eligible))
print(
    "Signals without valid prices:",
    len(directional) - len(eligible)
)

# ============================================================
# 8. SAVE AUDIT REPORT
# ============================================================

audit.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("TIMING AUDIT COMPLETED")
print("=" * 70)

print("Saved:", OUTPUT_PATH)

print("\nReview the future-return alignment and missing-price results.")
