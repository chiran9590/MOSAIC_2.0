
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    log_loss,
)

# ==========================================================
# MOSAIC 2.0 - NORMALIZED FEATURE EXPERIMENT
# ==========================================================

TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "normalized_feature_experiment.csv"

TARGET = "label"

ORIGINAL_FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "SMA_5",
    "SMA_10",
    "EMA_5",
    "EMA_10",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "ATR_14",
    "volume_change",
    "volume_ratio",
    "high_low_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "hour",
    "minute",
    "time_of_day",
]

PRICE_FEATURES = [
    "SMA_5",
    "SMA_10",
    "EMA_5",
    "EMA_10",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "ATR_14",
]

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

print("\nMOSAIC 2.0 - NORMALIZED FEATURE EXPERIMENT")
print("=" * 75)

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)

for df in (train, val):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if df[ORIGINAL_FEATURES + ["close", TARGET]].isna().any().any():
        raise ValueError("Missing values detected in required columns.")

print("\nDATA")
print("-" * 75)
print("Training records:", len(train))
print("Validation records:", len(val))
print("Training dates:", train["timestamp"].min(), "to", train["timestamp"].max())
print("Validation dates:", val["timestamp"].min(), "to", val["timestamp"].max())

if train["timestamp"].max() >= val["timestamp"].min():
    raise ValueError("Training and validation periods overlap.")

# ----------------------------------------------------------
# CREATE NORMALIZED FEATURES
# ----------------------------------------------------------

def add_normalized_features(df):

    df = df.copy()

    close = df["close"].replace(0, np.nan)

    # Moving-average relative distances
    df["close_sma5_distance"] = (
        df["close"] - df["SMA_5"]
    ) / close

    df["close_sma10_distance"] = (
        df["close"] - df["SMA_10"]
    ) / close

    df["close_ema5_distance"] = (
        df["close"] - df["EMA_5"]
    ) / close

    df["close_ema10_distance"] = (
        df["close"] - df["EMA_10"]
    ) / close

    # MACD normalized by price
    df["MACD_pct"] = df["MACD"] / close * 100

    df["MACD_signal_pct"] = (
        df["MACD_signal"] / close * 100
    )

    df["MACD_histogram_pct"] = (
        df["MACD_histogram"] / close * 100
    )

    # ATR normalized by price
    df["ATR_pct"] = df["ATR_14"] / close * 100

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


train = add_normalized_features(train)
val = add_normalized_features(val)

NORMALIZED_FEATURES = [
    "close_sma5_distance",
    "close_sma10_distance",
    "close_ema5_distance",
    "close_ema10_distance",
    "RSI_14",
    "MACD_pct",
    "MACD_signal_pct",
    "MACD_histogram_pct",
    "ATR_pct",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "volume_change",
    "volume_ratio",
    "high_low_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "hour",
    "minute",
    "time_of_day",
]

COMBINED_FEATURES = ORIGINAL_FEATURES + [
    feature for feature in NORMALIZED_FEATURES
    if feature not in ORIGINAL_FEATURES
]

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])


# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

def calculate_metrics(y_true, predictions, probabilities, classes):

    directional_classes = ["DOWN", "UP"]

    recalls = recall_score(
        y_true,
        predictions,
        labels=["DOWN", "SIDEWAYS", "UP"],
        average=None,
        zero_division=0,
    )

    directional_mask = np.isin(
        np.asarray(y_true),
        directional_classes,
    )

    directional_correct = (
        (np.asarray(y_true) == np.asarray(predictions))
        & directional_mask
    ).sum()

    directional_capture = (
        directional_correct / directional_mask.sum()
        if directional_mask.sum() else 0
    )

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "DOWN_recall": recalls[0],
        "SIDEWAYS_recall": recalls[1],
        "UP_recall": recalls[2],
        "directional_capture": directional_capture,
        "directional_prediction_pct": np.isin(
            predictions, directional_classes
        ).mean(),
        "log_loss": log_loss(
            y_true, probabilities, labels=classes
        ),
    }


# ----------------------------------------------------------
# CHRONOLOGICAL CROSS-VALIDATION
# ----------------------------------------------------------

print("\nSTARTING CHRONOLOGICAL CROSS-VALIDATION")
print("-" * 75)

train = train.sort_values("timestamp").reset_index(drop=True)

unique_dates = np.array(sorted(train["timestamp"].dt.date.unique()))

splitter = TimeSeriesSplit(n_splits=3)

feature_sets = {
    "Original_22": ORIGINAL_FEATURES,
    "Normalized_22": NORMALIZED_FEATURES,
    "Combined_30": COMBINED_FEATURES,
}

results = []

for name, features in feature_sets.items():

    print(f"\nMODEL: {name}")
    print("Feature count:", len(features))

    fold_metrics = []

    for fold, (train_day_idx, val_day_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):

        fold_train_dates = set(unique_dates[train_day_idx])
        fold_val_dates = set(unique_dates[val_day_idx])

        fold_train = train[
            train["timestamp"].dt.date.isin(fold_train_dates)
        ]

        fold_val = train[
            train["timestamp"].dt.date.isin(fold_val_dates)
        ]

        X_train = fold_train[features]
        y_train = fold_train[TARGET]

        X_val = fold_val[features]
        y_val = fold_val[TARGET]

        model = create_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)

        metrics = calculate_metrics(
            y_val,
            predictions,
            probabilities,
            model.classes_,
        )

        metrics["fold"] = fold

        fold_metrics.append(metrics)

        print(
            f"Fold {fold}: "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Balanced Acc={metrics['balanced_accuracy']:.4f}, "
            f"Directional Capture={metrics['directional_capture']:.4f}"
        )

    cv_macro = np.mean([
        x["macro_f1"] for x in fold_metrics
    ])

    cv_macro_std = np.std([
        x["macro_f1"] for x in fold_metrics
    ])

    cv_balanced = np.mean([
        x["balanced_accuracy"] for x in fold_metrics
    ])

    cv_directional = np.mean([
        x["directional_capture"] for x in fold_metrics
    ])

    # ------------------------------------------------------
    # FULL TRAIN -> VALIDATION
    # ------------------------------------------------------

    model = create_model()

    model.fit(
        train[features],
        train[TARGET],
    )

    val_predictions = model.predict(val[features])
    val_probabilities = model.predict_proba(val[features])

    validation_metrics = calculate_metrics(
        val[TARGET],
        val_predictions,
        val_probabilities,
        model.classes_,
    )

    result = {
        "model": name,
        "feature_count": len(features),
        "cv_macro_f1_mean": cv_macro,
        "cv_macro_f1_std": cv_macro_std,
        "cv_balanced_accuracy_mean": cv_balanced,
        "cv_directional_capture_mean": cv_directional,
    }

    result.update({
        f"validation_{key}": value
        for key, value in validation_metrics.items()
    })

    results.append(result)

    print("\nValidation:")
    print("Macro F1:", round(validation_metrics["macro_f1"], 4))
    print("Balanced accuracy:", round(validation_metrics["balanced_accuracy"], 4))
    print("Directional capture:", round(validation_metrics["directional_capture"], 4))
    print("Directional predictions:", round(validation_metrics["directional_prediction_pct"], 4))
    print("Log loss:", round(validation_metrics["log_loss"], 4))


# ----------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("FINAL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:", OUTPUT_FILE)
print("\nEXPERIMENT COMPLETED")
