
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score
)

print("\nMOSAIC 2.0 - MACD TEMPORAL COMPARISON")
print("=" * 85)

TRAIN_PATH = "data/processed/splits/train.csv"
VAL_PATH = "data/processed/splits/validation.csv"

TARGET = "label"
DATE_COL = "timestamp"

MACD_FEATURES = [
    "MACD",
    "MACD_signal",
    "MACD_histogram"
]

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

train[DATE_COL] = pd.to_datetime(train[DATE_COL])
val[DATE_COL] = pd.to_datetime(val[DATE_COL])

# Original 22 features, excluding metadata and target.
excluded = {
    TARGET, DATE_COL,
    "open", "high", "low", "close", "volume",
    "future_return"
}

features = [
    col for col in train.columns
    if col not in excluded
]

features = [
    col for col in features
    if col in val.columns
]

baseline_features = features.copy()

candidate_features = [
    col for col in baseline_features
    if col not in MACD_FEATURES
]

print(f"\nBaseline feature count: {len(baseline_features)}")
print(f"No-MACD feature count: {len(candidate_features)}")

print("\nValidation dates:")
print(val[DATE_COL].min(), "to", val[DATE_COL].max())

# Five chronological blocks based on unique trading dates.
dates = np.array(sorted(val[DATE_COL].dt.date.unique()))
date_blocks = np.array_split(dates, 5)

def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ])

def directional_capture(y_true, y_pred):
    actual_directional = np.isin(y_true, ["DOWN", "UP"])

    correct_directional = (
        actual_directional &
        (y_true == y_pred) &
        np.isin(y_pred, ["DOWN", "UP"])
    )

    total = actual_directional.sum()

    if total == 0:
        return 0.0

    return correct_directional.sum() / total

def evaluate_configuration(name, selected_features, train, block):
    model = create_model()

    X_train = train[selected_features]
    y_train = train[TARGET]

    X_val = block[selected_features]
    y_val = block[TARGET]

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return {
        "configuration": name,
        "records": len(block),
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "DOWN_recall": recall_score(
            y_val, predictions,
            labels=["DOWN"],
            average="macro",
            zero_division=0
        ),
        "SIDEWAYS_recall": recall_score(
            y_val, predictions,
            labels=["SIDEWAYS"],
            average="macro",
            zero_division=0
        ),
        "UP_recall": recall_score(
            y_val, predictions,
            labels=["UP"],
            average="macro",
            zero_division=0
        ),
        "directional_capture": directional_capture(
            y_val.to_numpy(),
            predictions
        )
    }

results = []

for index, block_dates in enumerate(date_blocks, start=1):

    block = val[
        val[DATE_COL].dt.date.isin(block_dates)
    ].copy()

    print("\n" + "-" * 85)
    print(f"VALIDATION BLOCK {index}")
    print(
        f"Dates: {block[DATE_COL].min().date()} "
        f"to {block[DATE_COL].max().date()}"
    )
    print(f"Records: {len(block)}")

    for name, selected_features in [
        ("Baseline_22", baseline_features),
        ("No_MACD_19", candidate_features)
    ]:

        result = evaluate_configuration(
            name,
            selected_features,
            train,
            block
        )

        result["block"] = index
        results.append(result)

        print(
            f"\n{name}: "
            f"Macro F1={result['macro_f1']:.4f}, "
            f"Balanced Acc={result['balanced_accuracy']:.4f}, "
            f"Directional Capture={result['directional_capture']:.4f}"
        )

results_df = pd.DataFrame(results)

print("\n" + "=" * 85)
print("TEMPORAL COMPARISON SUMMARY")
print("=" * 85)

summary = results_df.pivot(
    index="block",
    columns="configuration",
    values=[
        "macro_f1",
        "balanced_accuracy",
        "directional_capture"
    ]
)

print(summary.round(4).to_string())

print("\nAVERAGE ACROSS FIVE BLOCKS")
print(
    results_df.groupby("configuration")[
        [
            "macro_f1",
            "balanced_accuracy",
            "directional_capture"
        ]
    ].mean().round(4)
)

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/macd_temporal_comparison.csv",
    index=False
)

print("\nSaved: reports/macd_temporal_comparison.csv")
print("\nEXPERIMENT COMPLETED")
