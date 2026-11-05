
import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, timezone
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

print("\nMOSAIC 2.0 - FREEZE CANDIDATE MODEL")
print("=" * 75)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

TRAIN_PATH = "data/processed/splits/train.csv"
VAL_PATH = "data/processed/splits/validation.csv"

OUTPUT_DIR = "models/saved"
REPORT_DIR = "reports"

MODEL_NAME = "mosaic_candidate_no_macd_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. EXACT SELECTED FEATURES
# --------------------------------------------------

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

TARGET = "label"

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

print(f"Training records:   {len(train)}")
print(f"Validation records: {len(val)}")

# --------------------------------------------------
# 4. VALIDATE INPUT DATA
# --------------------------------------------------

missing_train = [
    col for col in FEATURES + [TARGET]
    if col not in train.columns
]

missing_val = [
    col for col in FEATURES + [TARGET]
    if col not in val.columns
]

if missing_train:
    raise ValueError(
        f"Missing training columns: {missing_train}"
    )

if missing_val:
    raise ValueError(
        f"Missing validation columns: {missing_val}"
    )

if len(FEATURES) != 19:
    raise ValueError("Expected exactly 19 features.")

if len(set(FEATURES)) != len(FEATURES):
    raise ValueError("Duplicate features detected.")

if any("MACD" in col for col in FEATURES):
    raise ValueError("MACD feature detected in candidate.")

X_train = train[FEATURES].copy()
X_val = val[FEATURES].copy()

y_train = train[TARGET].astype(str)
y_val = val[TARGET].astype(str)

if X_train.isnull().any().any():
    raise ValueError("Training features contain missing values.")

if X_val.isnull().any().any():
    raise ValueError("Validation features contain missing values.")

if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    raise ValueError("Training features contain infinite values.")

if not np.isfinite(X_val.to_numpy(dtype=float)).all():
    raise ValueError("Validation features contain infinite values.")

if set(y_train.unique()) != {"DOWN", "SIDEWAYS", "UP"}:
    raise ValueError(
        f"Unexpected training classes: {sorted(y_train.unique())}"
    )

if not set(y_val.unique()).issubset(set(y_train.unique())):
    raise ValueError("Validation contains an unknown class.")

print("\nFeature validation passed.")
print(f"Selected features: {len(FEATURES)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

# --------------------------------------------------
# 5. ENCODE LABELS
# --------------------------------------------------

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

print("\nClass encoding:")
for index, class_name in enumerate(encoder.classes_):
    print(f"{index} = {class_name}")

# --------------------------------------------------
# 6. BUILD MODEL
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ))
])

# --------------------------------------------------
# 7. TRAIN ON TRAINING DATA ONLY
# --------------------------------------------------

print("\nTraining candidate model...")

model.fit(X_train, y_train_encoded)

print("Training completed.")

# --------------------------------------------------
# 8. VALIDATION EVALUATION
# --------------------------------------------------

print("\nEvaluating on validation data...")

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val_encoded,
    predictions
)

macro_f1 = f1_score(
    y_val_encoded,
    predictions,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_val_encoded,
    predictions
)

report = classification_report(
    y_val_encoded,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_val_encoded,
    predictions,
    labels=np.arange(len(encoder.classes_))
)

print(f"\nValidation accuracy:          {accuracy:.4f}")
print(f"Validation Macro F1:          {macro_f1:.4f}")
print(f"Validation balanced accuracy: {balanced_acc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_val_encoded,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(pd.DataFrame(
    matrix,
    index=[f"Actual_{c}" for c in encoder.classes_],
    columns=[f"Predicted_{c}" for c in encoder.classes_]
))

# --------------------------------------------------
# 9. SAVE VERSIONED ARTIFACTS
# --------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_NAME}.joblib"
)

encoder_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_NAME}_encoder.joblib"
)

features_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_NAME}_features.json"
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_NAME}_metadata.json"
)

report_path = os.path.join(
    REPORT_DIR,
    f"{MODEL_NAME}_validation.json"
)

joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)

with open(features_path, "w") as file:
    json.dump(FEATURES, file, indent=4)

metadata = {
    "model_name": MODEL_NAME,
    "model_type": "LogisticRegression",
    "version": "1.0",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "target": TARGET,
    "prediction_horizon": "30 minutes",
    "features": FEATURES,
    "feature_count": len(FEATURES),
    "removed_features": [
        "MACD",
        "MACD_signal",
        "MACD_histogram"
    ],
    "class_names": encoder.classes_.tolist(),
    "training_records": int(len(train)),
    "validation_records": int(len(val)),
    "training_start": str(
        train["timestamp"].min()
    ),
    "training_end": str(
        train["timestamp"].max()
    ),
    "validation_start": str(
        val["timestamp"].min()
    ),
    "validation_end": str(
        val["timestamp"].max()
    ),
    "validation_metrics": {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc)
    },
    "training_configuration": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 3000,
        "scaler": "StandardScaler"
    },
    "status": "experimental_candidate_not_production_validated"
}

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

validation_report = {
    "model_name": MODEL_NAME,
    "metrics": {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc)
    },
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_order": encoder.classes_.tolist()
}

with open(report_path, "w") as file:
    json.dump(validation_report, file, indent=4)

# --------------------------------------------------
# 10. RELOAD AND VERIFY ARTIFACTS
# --------------------------------------------------

print("\nVerifying saved artifacts...")

loaded_model = joblib.load(model_path)
loaded_encoder = joblib.load(encoder_path)

loaded_predictions = loaded_model.predict(X_val)

if not np.array_equal(predictions, loaded_predictions):
    raise RuntimeError("Saved model predictions do not match.")

decoded = loaded_encoder.inverse_transform(
    loaded_predictions
)

if len(decoded) != len(val):
    raise RuntimeError("Prediction count verification failed.")

print("Model reload verification passed.")

print("\nSAVED FILES")
print("-" * 75)

for path in [
    model_path,
    encoder_path,
    features_path,
    metadata_path,
    report_path
]:
    print(path)

print("\n" + "=" * 75)
print("CANDIDATE MODEL FROZEN SUCCESSFULLY")
print("Status: Experimental candidate")
print("Original baseline model: Untouched")
print("=" * 75)
