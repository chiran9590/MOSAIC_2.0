
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
    f1_score,
    recall_score,
    balanced_accuracy_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

print("\nMOSAIC 2.0 - SELECTED FEATURE EXPERIMENT")
print("=" * 70)

os.makedirs("reports", exist_ok=True)

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. FEATURE SETS
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
    "bb_position", "bb_width_pct", "close_vwap_pct",
    "adx_14", "plus_di_14", "minus_di_14", "di_spread",
    "stochastic_k", "stochastic_d",
    "obv_change", "obv_ratio"
]

# One representative from several highly correlated groups.
selected_features = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "SMA_10",
    "RSI_14",
    "MACD_histogram",
    "ATR_14",
    "volume_ratio",
    "high_low_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "time_of_day",
    "bb_width_pct",
    "close_vwap_pct",
    "adx_14",
    "stochastic_d",
    "obv_ratio"
]

compact_features = [
    "ATR_14",
    "bb_width_pct",
    "high_low_range_pct",
    "SMA_10",
    "MACD_histogram",
    "return_12",
    "return_1",
    "time_of_day",
    "stochastic_d",
    "adx_14",
    "close_vwap_pct",
    "obv_ratio"
]

feature_sets = {
    "22 Original": original_features,
    "33 Advanced": advanced_features,
    "18 Selected": selected_features,
    "12 Compact": compact_features
}

# ------------------------------------------------------------
# 2. LOAD MATCHED DATA
# ------------------------------------------------------------

print("\nLoading datasets...")

train = pd.read_csv(
    "data/processed/advanced_splits/train.csv",
    parse_dates=["timestamp"]
)

validation = pd.read_csv(
    "data/processed/advanced_splits/validation.csv",
    parse_dates=["timestamp"]
)

print("Training records:", len(train))
print("Validation records:", len(validation))

# Check feature availability
for name, features in feature_sets.items():
    missing = set(features) - set(train.columns)

    if missing:
        raise ValueError(f"{name} missing features: {missing}")

# ------------------------------------------------------------
# 3. PREPARE LABELS
# ------------------------------------------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(train["label"])
y_validation = encoder.transform(validation["label"])

print("\nClass encoding:")
for i, name in enumerate(encoder.classes_):
    print(i, "=", name)

print("\nFeature set sizes:")
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} features")

# ------------------------------------------------------------
# 4. DEFINE MODELS
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
            num_class=3,
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
# 5. EVALUATION
# ------------------------------------------------------------

def evaluate(model, X_train, X_val, model_name):

    if model_name == "XGBoost":

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )

    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    recalls = recall_score(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    down_recall = recalls[0]
    sideways_recall = recalls[1]
    up_recall = recalls[2]

    return {
        "accuracy": accuracy_score(y_validation, predictions),
        "macro_f1": f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation,
            predictions
        ),
        "down_recall": down_recall,
        "sideways_recall": sideways_recall,
        "up_recall": up_recall,
        "direction_recall": (
            down_recall + up_recall
        ) / 2
    }

# ------------------------------------------------------------
# 6. RUN EXPERIMENTS
# ------------------------------------------------------------

results = []

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print("FEATURE SET:", feature_set_name)
    print("Number of features:", len(features))
    print("=" * 70)

    X_train = train[features]
    X_val = validation[features]

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid training values in {feature_set_name}")

    if not np.isfinite(X_val.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid validation values in {feature_set_name}")

    for model_name, model in create_models().items():

        print("\nTraining:", model_name)

        metrics = evaluate(
            model,
            X_train,
            X_val,
            model_name
        )

        result = {
            "feature_set": feature_set_name,
            "feature_count": len(features),
            "model": model_name,
            **metrics
        }

        results.append(result)

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Direction Recall: {metrics['direction_recall']:.4f}"
        )

# ------------------------------------------------------------
# 7. SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["macro_f1", "direction_recall"],
    ascending=False
)

results_df.to_csv(
    "reports/selected_feature_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL FEATURE SELECTION RESULTS")
print("=" * 70)

columns = [
    "feature_set",
    "feature_count",
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
    results_df[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nBEST RESULT FOR EACH FEATURE SET")
print("-" * 70)

for name in feature_sets:

    subset = results_df[
        results_df["feature_set"] == name
    ]

    best = subset.sort_values(
        by=["macro_f1", "direction_recall"],
        ascending=False
    ).iloc[0]

    print(f"\n{name}")
    print("Model:", best["model"])
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Direction Recall: {best['direction_recall']:.4f}")

print("\nSaved: reports/selected_feature_results.csv")
print("Experiment completed.")
print("Test dataset was NOT used.")
