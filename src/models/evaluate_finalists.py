
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier


print("\nMOSAIC 2.0 - FINALIST VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

TRAIN_PATH = "data/processed/advanced_splits/train.csv"
VAL_PATH = "data/processed/advanced_splits/validation.csv"

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

print("\nTraining records:", len(train))
print("Validation records:", len(val))

# --------------------------------------------------
# 2. DEFINE FEATURE SETS
# --------------------------------------------------

original_features = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

selected_features = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_10", "RSI_14", "MACD_histogram", "ATR_14",
    "volume_ratio", "high_low_range_pct", "upper_wick_pct",
    "lower_wick_pct", "time_of_day", "bb_width_pct",
    "close_vwap_pct", "adx_14", "stochastic_d", "obv_ratio"
]

# --------------------------------------------------
# 3. PREPARE LABELS
# --------------------------------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(train["label"])
y_val = encoder.transform(val["label"])

print("\nClass encoding:")
for i, name in enumerate(encoder.classes_):
    print(f"{i} = {name}")

# --------------------------------------------------
# 4. DEFINE FINALISTS
# --------------------------------------------------

finalists = [
    {
        "name": "Original_LR_C1",
        "features": original_features,
        "model_type": "LR",
        "C": 1.0
    },
    {
        "name": "Original_LR_C001",
        "features": original_features,
        "model_type": "LR",
        "C": 0.01
    },
    {
        "name": "Selected_LR_C01",
        "features": selected_features,
        "model_type": "LR",
        "C": 0.1
    },
    {
        "name": "Original_LightGBM",
        "features": original_features,
        "model_type": "LGBM",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1
    },
    {
        "name": "Selected_LightGBM",
        "features": selected_features,
        "model_type": "LGBM",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 10
    }
]

# --------------------------------------------------
# 5. TRAIN AND EVALUATE
# --------------------------------------------------

results = []

os.makedirs("reports", exist_ok=True)

for finalist in finalists:

    name = finalist["name"]
    features = finalist["features"]

    print("\n" + "=" * 70)
    print("MODEL:", name)
    print("=" * 70)

    X_train = train[features]
    X_val = val[features]

    if finalist["model_type"] == "LR":

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=finalist["C"],
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            ))
        ])

    else:

        model = LGBMClassifier(
            n_estimators=finalist["n_estimators"],
            learning_rate=finalist["learning_rate"],
            num_leaves=finalist["num_leaves"],
            max_depth=finalist["max_depth"],
            min_child_samples=30,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )
    balanced_acc = balanced_accuracy_score(
        y_val, predictions
    )

    down_recall = recall_score(
        y_val, predictions, labels=[0], average="macro",
        zero_division=0
    )

    sideways_recall = recall_score(
        y_val, predictions, labels=[1], average="macro",
        zero_division=0
    )

    up_recall = recall_score(
        y_val, predictions, labels=[2], average="macro",
        zero_division=0
    )

    direction_recall = (down_recall + up_recall) / 2

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"DOWN Recall:        {down_recall:.4f}")
    print(f"SIDEWAYS Recall:    {sideways_recall:.4f}")
    print(f"UP Recall:          {up_recall:.4f}")
    print(f"Direction Recall:   {direction_recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1, 2],
        target_names=encoder.classes_,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1, 2]
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    cm_path = f"reports/confusion_{name}.csv"
    cm_df.to_csv(cm_path)

    print("\nConfusion Matrix:")
    print(cm_df)

    print("Saved:", cm_path)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "down_recall": down_recall,
        "sideways_recall": sideways_recall,
        "up_recall": up_recall,
        "direction_recall": direction_recall
    })

# --------------------------------------------------
# 6. FINAL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["macro_f1", "direction_recall"],
    ascending=False
)

print("\n\n" + "=" * 70)
print("FINALIST VALIDATION COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/finalist_validation_results.csv",
    index=False
)

print("\nSaved: reports/finalist_validation_results.csv")

print("\n" + "=" * 70)
print("STAGE 6 COMPLETED")
print("=" * 70)

print("Training data used:", len(train))
print("Validation data used:", len(val))
print("Test data was NOT used.")
