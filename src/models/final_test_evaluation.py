
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# MOSAIC 2.0 - FINAL TEST EVALUATION
# ============================================================

print("\nMOSAIC 2.0 - FINAL TEST EVALUATION")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed/advanced_splits/train.csv"
VALIDATION_PATH = "data/processed/advanced_splits/validation.csv"
TEST_PATH = "data/processed/advanced_splits/test.csv"

LABELS_PATH = "data/processed/reliance_30min_labels.csv"

MODEL_DIR = "models/saved"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COLUMN = "label"

FEATURES = [
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
    "time_of_day"
]

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["timestamp"]
)

validation = pd.read_csv(
    VALIDATION_PATH,
    parse_dates=["timestamp"]
)

test = pd.read_csv(
    TEST_PATH,
    parse_dates=["timestamp"]
)

labels_df = pd.read_csv(
    LABELS_PATH,
    parse_dates=["timestamp"]
)

print(f"Training records:   {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Test records:       {len(test)}")
print(f"Original labels:    {len(labels_df)}")


# ============================================================
# 3. VALIDATE DATA
# ============================================================

print("\nValidating datasets...")

for name, dataframe in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:

    if dataframe["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps found in {name} dataset.")

    missing_features = [
        feature for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {name}: {missing_features}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"Missing labels found in {name} dataset.")


if labels_df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found in original labels file.")

if "future_return" not in labels_df.columns:
    raise ValueError("future_return column missing from original labels file.")


# Check chronological separation

train_dates = set(train["timestamp"].dt.date)
validation_dates = set(validation["timestamp"].dt.date)
test_dates = set(test["timestamp"].dt.date)

if train_dates.intersection(validation_dates):
    raise ValueError("Train and validation dates overlap.")

if train_dates.intersection(test_dates):
    raise ValueError("Train and test dates overlap.")

if validation_dates.intersection(test_dates):
    raise ValueError("Validation and test dates overlap.")

print("No duplicate timestamps.")
print("No missing model features.")
print("No chronological date overlap.")


# ============================================================
# 4. PREPARE TRAINING DATA
# ============================================================

print("\nPreparing training data...")

development = pd.concat(
    [train, validation],
    ignore_index=True
)

development = development.sort_values(
    "timestamp"
).reset_index(drop=True)

X_train = development[FEATURES].copy()
y_train = development[TARGET_COLUMN].copy()

X_test = test[FEATURES].copy()
y_test = test[TARGET_COLUMN].copy()

print(f"Combined development records: {len(development)}")
print(f"Number of features: {len(FEATURES)}")


# ============================================================
# 5. ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

encoder.fit(["DOWN", "SIDEWAYS", "UP"])

y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("\nClass encoding:")

for index, class_name in enumerate(encoder.classes_):
    print(f"{index} = {class_name}")


# ============================================================
# 6. TRAIN FROZEN MODEL
# ============================================================

print("\nTraining frozen Logistic Regression model...")

print("Features: 22")
print("C = 1.0")
print("Class weights = balanced")

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    ))
])

model.fit(
    X_train,
    y_train_encoded
)

print("Training completed.")


# ============================================================
# 7. SAVE FINAL MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "mosaic_final_model.joblib"
)

encoder_path = os.path.join(
    MODEL_DIR,
    "mosaic_label_encoder.joblib"
)

joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)

print("\nSaved final model:")
print(model_path)

print("Saved label encoder:")
print(encoder_path)


# ============================================================
# 8. FINAL TEST PREDICTIONS
# ============================================================

print("\nEvaluating on untouched test dataset...")

predictions = model.predict(X_test)

predicted_labels = encoder.inverse_transform(
    predictions
)

actual_labels = encoder.inverse_transform(
    y_test_encoded
)


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_encoded,
    predictions
)

macro_f1 = f1_score(
    y_test_encoded,
    predictions,
    average="macro"
)

balanced_accuracy = balanced_accuracy_score(
    y_test_encoded,
    predictions
)

class_recalls = classification_report(
    y_test_encoded,
    predictions,
    labels=[0, 1, 2],
    output_dict=True,
    zero_division=0
)

down_recall = class_recalls["0"]["recall"]
sideways_recall = class_recalls["1"]["recall"]
up_recall = class_recalls["2"]["recall"]

direction_recall = (
    down_recall + up_recall
) / 2


# ============================================================
# 10. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
print(f"DOWN Recall:        {down_recall:.4f}")
print(f"SIDEWAYS Recall:    {sideways_recall:.4f}")
print(f"UP Recall:          {up_recall:.4f}")
print(f"Direction Recall:   {direction_recall:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test_encoded,
        predictions,
        labels=[0, 1, 2],
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_test_encoded,
    predictions,
    labels=[0, 1, 2]
)

print("\nCONFUSION MATRIX")

print(
    pd.DataFrame(
        matrix,
        index=["DOWN", "SIDEWAYS", "UP"],
        columns=["DOWN", "SIDEWAYS", "UP"]
    )
)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_output = pd.DataFrame([{
    "model": "Original_22_Feature_LR_C1",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_accuracy,
    "down_recall": down_recall,
    "sideways_recall": sideways_recall,
    "up_recall": up_recall,
    "direction_recall": direction_recall,
    "train_records": len(train),
    "validation_records": len(validation),
    "development_records": len(development),
    "test_records": len(test),
    "feature_count": len(FEATURES)
}])

metrics_output.to_csv(
    os.path.join(REPORT_DIR, "final_test_results.csv"),
    index=False
)

confusion_output = pd.DataFrame(
    matrix,
    index=["DOWN", "SIDEWAYS", "UP"],
    columns=["DOWN", "SIDEWAYS", "UP"]
)

confusion_output.to_csv(
    os.path.join(REPORT_DIR, "final_test_confusion_matrix.csv")
)


# ============================================================
# 14. RECOVER FUTURE RETURNS
# ============================================================

print("\nMatching future returns using timestamps...")

prediction_output = test[
    ["timestamp", "label"]
].merge(
    labels_df[["timestamp", "future_return"]],
    on="timestamp",
    how="left",
    validate="one_to_one"
)

if len(prediction_output) != len(test):
    raise ValueError(
        "Prediction record count changed during future_return merge."
    )

if prediction_output["future_return"].isna().any():
    missing_count = prediction_output["future_return"].isna().sum()

    raise ValueError(
        f"{missing_count} test records have no matching future_return."
    )

print("Future returns successfully matched.")


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_output["actual_encoded"] = y_test_encoded
prediction_output["predicted_encoded"] = predictions

prediction_output["actual_label"] = actual_labels
prediction_output["predicted_label"] = predicted_labels

# Save model probabilities for later analysis.
probabilities = model.predict_proba(X_test)

for index, class_name in enumerate(encoder.classes_):
    prediction_output[
        f"probability_{class_name.lower()}"
    ] = probabilities[:, index]

prediction_output = prediction_output.sort_values(
    "timestamp"
).reset_index(drop=True)

prediction_path = os.path.join(
    REPORT_DIR,
    "final_test_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 16. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nSaved reports:")

print("1. reports/final_test_results.csv")
print("2. reports/final_test_confusion_matrix.csv")
print("3. reports/final_test_predictions.csv")

print("\nSaved models:")

print("1. models/saved/mosaic_final_model.joblib")
print("2. models/saved/mosaic_label_encoder.joblib")

print("\nPrediction records:", len(prediction_output))

print("\nMOSAIC 2.0 final evaluation finished successfully.")
