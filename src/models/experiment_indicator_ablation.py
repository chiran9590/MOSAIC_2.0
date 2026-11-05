
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)

TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

OUTPUT_FILE = Path("reports/indicator_ablation.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET = "label"

ORIGINAL_FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14",
    "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day",
]

SMA_EMA = [
    "SMA_5", "SMA_10", "EMA_5", "EMA_10"
]

MACD_FEATURES = [
    "MACD", "MACD_signal", "MACD_histogram"
]

CONFIGURATIONS = {
    "A_Original_22": ORIGINAL_FEATURES,

    "B_Without_SMA_EMA": [
        f for f in ORIGINAL_FEATURES if f not in SMA_EMA
    ],

    "C_Without_MACD": [
        f for f in ORIGINAL_FEATURES if f not in MACD_FEATURES
    ],

    "D_Without_SMA_EMA_MACD": [
        f for f in ORIGINAL_FEATURES
        if f not in SMA_EMA + MACD_FEATURES
    ],

    "E_Without_SMA_EMA_MACD_ATR": [
        f for f in ORIGINAL_FEATURES
        if f not in SMA_EMA + MACD_FEATURES + ["ATR_14"]
    ],
}


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


def evaluate(y_true, predictions):

    labels = ["DOWN", "SIDEWAYS", "UP"]

    recalls = recall_score(
        y_true,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    y_true_array = np.asarray(y_true)
    pred_array = np.asarray(predictions)

    directional_mask = np.isin(
        y_true_array, ["DOWN", "UP"]
    )

    directional_capture = (
        np.sum(
            (y_true_array == pred_array) & directional_mask
        ) / directional_mask.sum()
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
            pred_array, ["DOWN", "UP"]
        ).mean(),
    }


print("\nMOSAIC 2.0 - INDICATOR ABLATION EXPERIMENT")
print("=" * 75)

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)

for df in (train, val):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

train = train.sort_values("timestamp").reset_index(drop=True)
val = val.sort_values("timestamp").reset_index(drop=True)

if train["timestamp"].max() >= val["timestamp"].min():
    raise ValueError("Training and validation dates overlap.")

unique_dates = np.array(
    sorted(train["timestamp"].dt.date.unique())
)

splitter = TimeSeriesSplit(n_splits=3)

results = []

for config_name, features in CONFIGURATIONS.items():

    print("\n" + "-" * 75)
    print(config_name)
    print("Feature count:", len(features))

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):

        train_dates = set(unique_dates[train_idx])
        fold_dates = set(unique_dates[val_idx])

        fold_train = train[
            train["timestamp"].dt.date.isin(train_dates)
        ]

        fold_val = train[
            train["timestamp"].dt.date.isin(fold_dates)
        ]

        model = create_model()

        model.fit(
            fold_train[features],
            fold_train[TARGET],
        )

        predictions = model.predict(fold_val[features])

        metrics = evaluate(
            fold_val[TARGET],
            predictions,
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Balanced Acc={metrics['balanced_accuracy']:.4f}, "
            f"Directional Capture={metrics['directional_capture']:.4f}"
        )

    # Full training -> validation
    model = create_model()

    model.fit(
        train[features],
        train[TARGET],
    )

    val_predictions = model.predict(val[features])

    validation_metrics = evaluate(
        val[TARGET],
        val_predictions,
    )

    result = {
        "configuration": config_name,
        "feature_count": len(features),
    }

    for metric in fold_results[0]:

        values = [x[metric] for x in fold_results]

        result[f"cv_{metric}_mean"] = np.mean(values)
        result[f"cv_{metric}_std"] = np.std(values)

        result[f"validation_{metric}"] = (
            validation_metrics[metric]
        )

    results.append(result)

    print("\nValidation:")
    print(
        "Macro F1:",
        round(validation_metrics["macro_f1"], 4)
    )
    print(
        "Balanced accuracy:",
        round(validation_metrics["balanced_accuracy"], 4)
    )
    print(
        "Directional capture:",
        round(validation_metrics["directional_capture"], 4)
    )
    print(
        "Directional predictions:",
        round(validation_metrics["directional_prediction_pct"], 4)
    )


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("FINAL ABLATION COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:", OUTPUT_FILE)
print("\nEXPERIMENT COMPLETED")
