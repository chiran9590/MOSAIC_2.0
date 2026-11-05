
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.feature_engine import (
    calculate_live_features,
    REQUIRED_FEATURES
)


CANDLE_PATH = (
    PROJECT_ROOT / "data/processed/reliance_5min.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT / "data/processed/splits/train.csv"
)

VAL_PATH = (
    PROJECT_ROOT / "data/processed/splits/validation.csv"
)


print("\nMOSAIC 2.0 - FEATURE PARITY TEST")
print("=" * 75)

# --------------------------------------------------
# 1. LOAD HISTORICAL CANDLES
# --------------------------------------------------

candles = pd.read_csv(CANDLE_PATH)

candles["timestamp"] = pd.to_datetime(
    candles["timestamp"]
)

print("Historical candles:", len(candles))

# --------------------------------------------------
# 2. CALCULATE FEATURES USING INFERENCE ENGINE
# --------------------------------------------------

calculated = calculate_live_features(candles)

print("Calculated feature records:", len(calculated))

# --------------------------------------------------
# 3. LOAD ORIGINAL TRAINING AND VALIDATION DATA
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

train["timestamp"] = pd.to_datetime(
    train["timestamp"]
)

val["timestamp"] = pd.to_datetime(
    val["timestamp"]
)

reference = pd.concat(
    [train, val],
    ignore_index=True
)

# --------------------------------------------------
# 4. MERGE BY TIMESTAMP
# --------------------------------------------------

comparison = calculated[
    ["timestamp"] + REQUIRED_FEATURES
].merge(
    reference[
        ["timestamp"] + REQUIRED_FEATURES
    ],
    on="timestamp",
    how="inner",
    suffixes=("_live", "_training"),
    validate="one_to_one"
)

print("Comparable records:", len(comparison))

if len(comparison) != len(reference):
    raise RuntimeError(
        "Not all training/validation records matched "
        "historical candle data."
    )

# --------------------------------------------------
# 5. COMPARE EVERY FEATURE
# --------------------------------------------------

print("\nFEATURE COMPARISON")
print("-" * 75)

all_passed = True

for feature in REQUIRED_FEATURES:

    live_values = comparison[
        f"{feature}_live"
    ].to_numpy(dtype=float)

    training_values = comparison[
        f"{feature}_training"
    ].to_numpy(dtype=float)

    differences = np.abs(
        live_values - training_values
    )

    max_difference = differences.max()

    passed = np.allclose(
        live_values,
        training_values,
        rtol=1e-8,
        atol=1e-8
    )

    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"

    print(
        f"{feature:25s} "
        f"Max difference: {max_difference:.12f} "
        f"[{status}]"
    )

# --------------------------------------------------
# 6. TEST LATEST FEATURE ROW
# --------------------------------------------------

print("\nLATEST FEATURE TEST")
print("-" * 75)

latest = calculated.dropna(
    subset=REQUIRED_FEATURES
).iloc[-1]

print("Latest candle:", latest["timestamp"])

for feature in REQUIRED_FEATURES:
    print(
        f"{feature:25s}: "
        f"{latest[feature]}"
    )

# --------------------------------------------------
# 7. FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 75)

if all_passed:
    print("FEATURE PARITY TEST PASSED")
    print("Inference calculations match training features.")
else:
    print("FEATURE PARITY TEST FAILED")
    print("Do not connect live data until discrepancies are fixed.")

print("=" * 75)
