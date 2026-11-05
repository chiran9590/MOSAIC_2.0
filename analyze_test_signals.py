
import os
import pandas as pd
import numpy as np

# ============================================================
# MOSAIC 2.0 - TEST SIGNAL ANALYSIS
# ============================================================

print("\nMOSAIC 2.0 - TEST SIGNAL ANALYSIS")
print("=" * 70)

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PATH = "reports/final_test_predictions.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.0015

# ============================================================
# 2. LOAD TEST PREDICTIONS
# ============================================================

print("\nLoading final test predictions...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["timestamp"]
)

required_columns = [
    "timestamp",
    "actual_label",
    "predicted_label",
    "future_return",
    "probability_down",
    "probability_sideways",
    "probability_up"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df.empty:
    raise ValueError("Prediction file is empty.")

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found.")

if df["future_return"].isna().any():
    raise ValueError("Missing future returns found.")

print(f"Test records: {len(df)}")
print("Data loaded successfully.")

# ============================================================
# 3. BASIC DATA VALIDATION
# ============================================================

print("\nValidating predictions...")

probability_columns = [
    "probability_down",
    "probability_sideways",
    "probability_up"
]

if not np.isfinite(df[probability_columns].to_numpy()).all():
    raise ValueError("Invalid probability values found.")

if not np.allclose(
    df[probability_columns].sum(axis=1),
    1.0,
    atol=1e-5
):
    raise ValueError("Prediction probabilities do not sum to 1.")

valid_classes = {"DOWN", "SIDEWAYS", "UP"}

if not set(df["predicted_label"]).issubset(valid_classes):
    raise ValueError("Unexpected predicted class found.")

if not set(df["actual_label"]).issubset(valid_classes):
    raise ValueError("Unexpected actual class found.")

print("Predictions and probabilities are valid.")

# ============================================================
# 4. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_counts = (
    df["predicted_label"]
    .value_counts()
    .reindex(["DOWN", "SIDEWAYS", "UP"], fill_value=0)
)

prediction_percentages = (
    prediction_counts / len(df) * 100
)

distribution = pd.DataFrame({
    "predicted_signals": prediction_counts,
    "percentage": prediction_percentages
})

print(distribution.round(2))

# ============================================================
# 5. ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED DISTRIBUTION")
print("=" * 70)

comparison = pd.crosstab(
    df["actual_label"],
    df["predicted_label"],
    margins=True
)

print(comparison)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_signal_confusion_analysis.csv"
    )
)

# ============================================================
# 6. FUTURE RETURN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FUTURE RETURN ANALYSIS BY PREDICTED SIGNAL")
print("=" * 70)

# Future return is expressed as a decimal.
# Convert to percentage for reporting.

df["future_return_pct"] = (
    df["future_return"] * 100
)

# A positive future return supports an UP prediction.
# A negative future return supports a DOWN prediction.
# SIDEWAYS means the absolute return is within the label threshold.

df["directional_hit"] = np.where(
    df["predicted_label"] == "UP",
    df["future_return"] > 0,
    np.where(
        df["predicted_label"] == "DOWN",
        df["future_return"] < 0,
        np.nan
    )
)

df["sideways_hit"] = np.where(
    df["predicted_label"] == "SIDEWAYS",
    df["future_return"].abs() <= THRESHOLD,
    np.nan
)

# ============================================================
# 7. SIGNAL PERFORMANCE SUMMARY
# ============================================================

summary = (
    df.groupby("predicted_label")
    .agg(
        total_signals=("future_return", "count"),
        mean_return=("future_return_pct", "mean"),
        median_return=("future_return_pct", "median"),
        std_return=("future_return_pct", "std"),
        minimum_return=("future_return_pct", "min"),
        maximum_return=("future_return_pct", "max")
    )
    .reindex(["DOWN", "SIDEWAYS", "UP"])
)

summary["directional_hit_rate"] = (
    df.groupby("predicted_label")["directional_hit"]
    .mean()
    .reindex(["DOWN", "SIDEWAYS", "UP"])
    * 100
)

summary["sideways_hit_rate"] = (
    df.groupby("predicted_label")["sideways_hit"]
    .mean()
    .reindex(["DOWN", "SIDEWAYS", "UP"])
    * 100
)

print(summary.round(3))

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_signal_return_analysis.csv"
    )
)

# ============================================================
# 8. PREDICTION CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION CONFIDENCE ANALYSIS")
print("=" * 70)

df["max_probability"] = (
    df[probability_columns].max(axis=1)
)

confidence_summary = (
    df.groupby("predicted_label")["max_probability"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max"
    )
    .reindex(["DOWN", "SIDEWAYS", "UP"])
)

print(confidence_summary.round(4))

confidence_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_signal_confidence_analysis.csv"
    )
)

# ============================================================
# 9. CONFIDENCE BUCKET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE BUCKET ANALYSIS")
print("=" * 70)

df["confidence_bucket"] = pd.cut(
    df["max_probability"],
    bins=[0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01],
    labels=[
        "Below 40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "Above 80%"
    ],
    include_lowest=True
)

confidence_buckets = (
    df.groupby(
        "confidence_bucket",
        observed=False
    )
    .agg(
        signals=("future_return", "count"),
        average_return=("future_return_pct", "mean"),
        median_return=("future_return_pct", "median")
    )
)

print(confidence_buckets.round(3))

confidence_buckets.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_confidence_bucket_analysis.csv"
    )
)

# ============================================================
# 10. DAILY SIGNAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DAILY SIGNAL DISTRIBUTION")
print("=" * 70)

df["trading_date"] = df["timestamp"].dt.date

daily_signals = (
    df.groupby(["trading_date", "predicted_label"])
    .size()
    .unstack(fill_value=0)
)

for label in ["DOWN", "SIDEWAYS", "UP"]:
    if label not in daily_signals.columns:
        daily_signals[label] = 0

daily_signals = daily_signals[
    ["DOWN", "SIDEWAYS", "UP"]
]

daily_signals["TOTAL"] = daily_signals.sum(axis=1)

print("\nDaily signal statistics:")

print(
    daily_signals[
        ["DOWN", "SIDEWAYS", "UP", "TOTAL"]
    ].describe().round(2)
)

daily_signals.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_daily_signal_distribution.csv"
    )
)

# ============================================================
# 11. OVERALL DIRECTIONAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL DIRECTIONAL ANALYSIS")
print("=" * 70)

up_signals = df[df["predicted_label"] == "UP"]

down_signals = df[df["predicted_label"] == "DOWN"]

sideways_signals = df[
    df["predicted_label"] == "SIDEWAYS"
]

up_hit_rate = (
    (up_signals["future_return"] > 0).mean() * 100
    if len(up_signals) else np.nan
)

down_hit_rate = (
    (down_signals["future_return"] < 0).mean() * 100
    if len(down_signals) else np.nan
)

sideways_hit_rate = (
    (
        sideways_signals["future_return"].abs()
        <= THRESHOLD
    ).mean() * 100
    if len(sideways_signals) else np.nan
)

print(f"UP signals: {len(up_signals)}")
print(f"UP positive-return rate: {up_hit_rate:.2f}%")

print(f"\nDOWN signals: {len(down_signals)}")
print(f"DOWN negative-return rate: {down_hit_rate:.2f}%")

print(f"\nSIDEWAYS signals: {len(sideways_signals)}")
print(f"SIDEWAYS within-threshold rate: {sideways_hit_rate:.2f}%")

# ============================================================
# 12. SAVE COMPLETE ANALYSIS DATA
# ============================================================

analysis_path = os.path.join(
    OUTPUT_DIR,
    "test_signal_analysis_complete.csv"
)

df.to_csv(
    analysis_path,
    index=False
)

# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TEST SIGNAL ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated reports:")

print("1. test_signal_confusion_analysis.csv")
print("2. test_signal_return_analysis.csv")
print("3. test_signal_confidence_analysis.csv")
print("4. test_confidence_bucket_analysis.csv")
print("5. test_daily_signal_distribution.csv")
print("6. test_signal_analysis_complete.csv")

print("\nNo model retraining or parameter tuning was performed.")

print("\nMOSAIC 2.0 Stage 8 - Step 1 completed successfully.")
