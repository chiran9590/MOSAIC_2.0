
import os
import numpy as np
import pandas as pd

# ==========================================================
# MOSAIC 2.0 - ADVANCED FEATURE ENGINEERING
# ==========================================================

print("\nMOSAIC 2.0 - ADVANCED FEATURE ENGINEERING")
print("=" * 65)

CANDLE_PATH = "data/processed/reliance_5min.csv"
BASE_FEATURE_PATH = "data/processed/reliance_features.csv"

SPLIT_DIR = "data/processed/splits"
OUTPUT_PATH = "data/processed/reliance_advanced_features.csv"
ADVANCED_SPLIT_DIR = "data/processed/advanced_splits"

os.makedirs(ADVANCED_SPLIT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. Load datasets
# ----------------------------------------------------------

print("\nLoading datasets...")

candles = pd.read_csv(CANDLE_PATH)
base = pd.read_csv(BASE_FEATURE_PATH)

candles["timestamp"] = pd.to_datetime(candles["timestamp"])
base["timestamp"] = pd.to_datetime(base["timestamp"])

candles = candles.sort_values("timestamp").reset_index(drop=True)
base = base.sort_values("timestamp").reset_index(drop=True)

required_columns = [
    "timestamp", "open", "high", "low", "close", "volume"
]

missing = set(required_columns) - set(candles.columns)

if missing:
    raise ValueError(f"Missing candle columns: {missing}")

candles["trading_date"] = candles["timestamp"].dt.date

print("5-minute candles:", len(candles))
print("Original feature records:", len(base))

# ----------------------------------------------------------
# 2. Advanced indicator calculations
# ----------------------------------------------------------

def calculate_advanced_features(group):

    group = group.copy()

    high = group["high"].astype(float)
    low = group["low"].astype(float)
    close = group["close"].astype(float)
    volume = group["volume"].astype(float)

    previous_close = close.shift(1)

    # ======================================================
    # A. BOLLINGER BANDS
    # ======================================================

    rolling_mean = close.rolling(
        window=20, min_periods=20
    ).mean()

    rolling_std = close.rolling(
        window=20, min_periods=20
    ).std()

    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    band_range = upper_band - lower_band

    group["bb_position"] = (
        (close - lower_band) /
        band_range.replace(0, np.nan)
    )

    group["bb_width_pct"] = (
        band_range /
        rolling_mean.replace(0, np.nan)
    ) * 100

    # ======================================================
    # B. VWAP - RESET EACH TRADING DAY
    # ======================================================

    typical_price = (high + low + close) / 3

    cumulative_pv = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()

    vwap = (
        cumulative_pv /
        cumulative_volume.replace(0, np.nan)
    )

    group["close_vwap_pct"] = (
        (close - vwap) /
        vwap.replace(0, np.nan)
    ) * 100

    # ======================================================
    # C. ADX, POSITIVE DI AND NEGATIVE DI
    # ======================================================

    upward_move = high.diff()
    downward_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (upward_move > downward_move) &
            (upward_move > 0),
            upward_move,
            0.0
        ),
        index=group.index
    )

    minus_dm = pd.Series(
        np.where(
            (downward_move > upward_move) &
            (downward_move > 0),
            downward_move,
            0.0
        ),
        index=group.index
    )

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    # Wilder-style exponential smoothing
    atr_smoothed = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    plus_di = (
        100 * plus_smoothed /
        atr_smoothed.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_smoothed /
        atr_smoothed.replace(0, np.nan)
    )

    di_sum = plus_di + minus_di

    dx = (
        100 * (plus_di - minus_di).abs() /
        di_sum.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    group["adx_14"] = adx
    group["plus_di_14"] = plus_di
    group["minus_di_14"] = minus_di
    group["di_spread"] = plus_di - minus_di

    # ======================================================
    # D. STOCHASTIC OSCILLATOR
    # ======================================================

    lowest_low = low.rolling(
        window=14, min_periods=14
    ).min()

    highest_high = high.rolling(
        window=14, min_periods=14
    ).max()

    stoch_range = highest_high - lowest_low

    stoch_k = (
        100 * (close - lowest_low) /
        stoch_range.replace(0, np.nan)
    )

    stoch_d = stoch_k.rolling(
        window=3, min_periods=3
    ).mean()

    group["stochastic_k"] = stoch_k
    group["stochastic_d"] = stoch_d

    # ======================================================
    # E. ON-BALANCE VOLUME
    # ======================================================

    price_direction = np.sign(close.diff()).fillna(0)

    obv = (price_direction * volume).cumsum()

    group["obv_change"] = (
        obv.diff() /
        volume.rolling(20, min_periods=20).mean()
    )

    volume_sum = volume.rolling(
        window=20, min_periods=20
    ).sum()

    obv_change_20 = obv - obv.shift(20)

    group["obv_ratio"] = (
        obv_change_20 /
        volume_sum.replace(0, np.nan)
    )

    return group


# ----------------------------------------------------------
# 3. Calculate indicators independently per trading day
# ----------------------------------------------------------

print("\nCalculating advanced indicators...")

feature_groups = []

for date, group in candles.groupby("trading_date", sort=True):

    result = calculate_advanced_features(group)

    feature_groups.append(result)

advanced = pd.concat(
    feature_groups,
    ignore_index=True
)

ADVANCED_FEATURES = [
    "bb_position",
    "bb_width_pct",
    "close_vwap_pct",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "di_spread",
    "stochastic_k",
    "stochastic_d",
    "obv_change",
    "obv_ratio"
]

# ----------------------------------------------------------
# 4. Merge with existing 22-feature dataset
# ----------------------------------------------------------

print("\nMerging with original features...")

advanced_columns = advanced[
    ["timestamp"] + ADVANCED_FEATURES
]

combined = base.merge(
    advanced_columns,
    on="timestamp",
    how="inner",
    validate="one_to_one"
)

all_features = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
] + ADVANCED_FEATURES

combined[all_features] = combined[all_features].replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(combined)

combined = combined.dropna(
    subset=all_features + ["label"]
).copy()

combined = combined.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Records before cleaning:", before)
print("Records after cleaning:", len(combined))
print("Records removed:", before - len(combined))
print("Total features:", len(all_features))

# ----------------------------------------------------------
# 5. Data quality checks
# ----------------------------------------------------------

print("\nDATA QUALITY CHECK")
print("-" * 50)

print("Missing feature values:", int(
    combined[all_features].isna().sum().sum()
))

print("Duplicate timestamps:", int(
    combined["timestamp"].duplicated().sum()
))

print("\nNew feature statistics:")

print(
    combined[ADVANCED_FEATURES]
    .describe()
    .round(3)
    .to_string()
)

print("\nClass distribution:")

print(
    combined["label"]
    .value_counts()
    .to_string()
)

# ----------------------------------------------------------
# 6. Save advanced dataset
# ----------------------------------------------------------

combined.to_csv(OUTPUT_PATH, index=False)

print("\nSaved advanced dataset:")
print(OUTPUT_PATH)

# ----------------------------------------------------------
# 7. Preserve original chronological split dates
# ----------------------------------------------------------

print("\nCreating advanced chronological splits...")

for split_name in ["train", "validation", "test"]:

    original_split_path = os.path.join(
        SPLIT_DIR,
        f"{split_name}.csv"
    )

    original_split = pd.read_csv(original_split_path)

    original_split["timestamp"] = pd.to_datetime(
        original_split["timestamp"]
    )

    allowed_dates = set(
        original_split["timestamp"].dt.date
    )

    split_data = combined[
        combined["timestamp"].dt.date.isin(allowed_dates)
    ].copy()

    output_split = os.path.join(
        ADVANCED_SPLIT_DIR,
        f"{split_name}.csv"
    )

    split_data.to_csv(output_split, index=False)

    print(
        f"{split_name.capitalize()}: "
        f"{len(split_data)} records saved"
    )

print("\nADVANCED FEATURE ENGINEERING COMPLETED")
print("=" * 65)
