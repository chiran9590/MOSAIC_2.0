
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - FEATURE SET COMPARISON
# ============================================================

print("\nMOSAIC 2.0 - FEATURE SET COMPARISON")
print("=" * 65)

ORIGINAL_DIR = "data/processed/splits"
ADVANCED_DIR = "data/processed/advanced_splits"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. DEFINE FEATURES
# ------------------------------------------------------------

original_features = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

advanced_features = original_features + [
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

assert len(original_features) == 22
assert len(advanced_features) == 33

# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading datasets...")

original_train = pd.read_csv(
    f"{ORIGINAL_DIR}/train.csv",
    parse_dates=["timestamp"]
)

original_validation = pd.read_csv(
    f"{ORIGINAL_DIR}/validation.csv",
    parse_dates=["timestamp"]
)

advanced_train = pd.read_csv(
    f"{ADVANCED_DIR}/train.csv",
    parse_dates=["timestamp"]
)

advanced_validation = pd.read_csv(
    f"{ADVANCED_DIR}/validation.csv",
    parse_dates=["timestamp"]
)

print("Original training records:", len(original_train))
print("Advanced training records:", len(advanced_train))
print("Original validation records:", len(original_validation))
print("Advanced validation records:", len(advanced_validation))

# ------------------------------------------------------------
# 3. MATCH EXACT TIMESTAMPS
# ------------------------------------------------------------

print("\nMatching timestamps for a fair comparison...")

common_train_timestamps = sorted(
    set(original_train["timestamp"])
    & set(advanced_train["timestamp"])
)

common_validation_timestamps = sorted(
    set(original_validation["timestamp"])
    & set(advanced_validation["timestamp"])
)

original_train = (
    original_train[
        original_train["timestamp"].isin(common_train_timestamps)
    ]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

advanced_train = (
    advanced_train[
        advanced_train["timestamp"].isin(common_train_timestamps)
    ]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

original_validation = (
    original_validation[
        original_validation["timestamp"].isin(common_validation_timestamps)
    ]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

advanced_validation = (
    advanced_validation[
        advanced_validation["timestamp"].isin(common_validation_timestamps)
    ]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Verify timestamp alignment
assert original_train["timestamp"].equals(
    advanced_train["timestamp"]
)

assert original_validation["timestamp"].equals(
    advanced_validation["timestamp"]
)

# Verify labels are identical for matched records
assert original_train["label"].equals(advanced_train["label"])
assert original_validation["label"].equals(
    advanced_validation["label"]
)

print("\nMATCHED DATASET SUMMARY")
print("-" * 45)
print("Common training records:", len(original_train))
print("Common validation records:", len(original_validation))

# ------------------------------------------------------------
# 4. PREPARE LABELS
# ------------------------------------------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(original_train["label"])
y_validation = encoder.transform(original_validation["label"])

print("\nClass encoding:")
for index, class_name in enumerate(encoder.classes_):
    print(f"{index} = {class_name}")

# ------------------------------------------------------------
# 5. PREPARE FEATURE MATRICES
# ------------------------------------------------------------

X_train_original = original_train[original_features]
X_validation_original = original_validation[original_features]

X_train_advanced = advanced_train[advanced_features]
X_validation_advanced = advanced_validation[advanced_features]

# Ensure features are numeric and complete
for name, dataframe in [
    ("Original training", X_train_original),
    ("Original validation", X_validation_original),
    ("Advanced training", X_train_advanced),
    ("Advanced validation", X_validation_advanced),
]:
    if dataframe.isna().any().any():
        raise ValueError(f"Missing values found in {name} features.")

    if not np.isfinite(dataframe.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values found in {name} features.")

# ------------------------------------------------------------
# 6. DEFINE MODELS
# ------------------------------------------------------------

def create_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=10,
            num_leaves=15,
            min_child_samples=40,
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        )
    }


# ------------------------------------------------------------
# 7. EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(model, X_train, X_validation, feature_set, model_name):

    # Balanced sample weights for XGBoost
    if model_name == "XGBoost":
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)

    macro_precision = precision_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        predictions
    )

    class_recalls = recall_score(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    down_recall = class_recalls[0]
    sideways_recall = class_recalls[1]
    up_recall = class_recalls[2]

    direction_recall = (down_recall + up_recall) / 2

    result = {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "down_recall": down_recall,
        "sideways_recall": sideways_recall,
        "up_recall": up_recall,
        "direction_recall": direction_recall
    }

    print(
        f"{model_name:<22} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Direction Recall: {direction_recall:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1, 2]
        )
    )

    return result


# ------------------------------------------------------------
# 8. TRAIN AND COMPARE BOTH FEATURE SETS
# ------------------------------------------------------------

results = []

feature_sets = [
    (
        "22 original",
        X_train_original,
        X_validation_original
    ),
    (
        "33 advanced",
        X_train_advanced,
        X_validation_advanced
    )
]

print("\nSTARTING MODEL COMPARISON")
print("=" * 65)

for feature_set_name, X_train, X_validation in feature_sets:

    print(f"\nFEATURE SET: {feature_set_name}")
    print("-" * 65)

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        result = evaluate_model(
            model,
            X_train,
            X_validation,
            feature_set_name,
            model_name
        )

        results.append(result)

# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["macro_f1", "direction_recall"],
    ascending=False
)

output_path = f"{REPORT_DIR}/feature_set_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("FEATURE SET COMPARISON RESULTS")
print("=" * 65)

columns_to_display = [
    "feature_set",
    "model",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "down_recall",
    "sideways_recall",
    "up_recall",
    "direction_recall"
]

print(
    results_df[columns_to_display].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# 10. COMPARE BEST RESULTS
# ------------------------------------------------------------

print("\nBEST MODEL FOR EACH FEATURE SET")
print("-" * 65)

for feature_set_name in ["22 original", "33 advanced"]:

    subset = results_df[
        results_df["feature_set"] == feature_set_name
    ]

    best = subset.sort_values(
        by=["macro_f1", "direction_recall"],
        ascending=False
    ).iloc[0]

    print(f"\n{feature_set_name}")
    print("Best model:", best["model"])
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"DOWN Recall: {best['down_recall']:.4f}")
    print(f"SIDEWAYS Recall: {best['sideways_recall']:.4f}")
    print(f"UP Recall: {best['up_recall']:.4f}")
    print(f"Direction Recall: {best['direction_recall']:.4f}")

print("\nComparison completed successfully.")
print("Results saved to:", output_path)
print("\nTest dataset was NOT used.")
