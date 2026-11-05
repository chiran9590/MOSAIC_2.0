
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================================
# MOSAIC 2.0 - TIME SERIES HYPERPARAMETER TUNING
# ==========================================================

print("\nMOSAIC 2.0 - HYPERPARAMETER TUNING")
print("=" * 65)

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
validation_df["timestamp"] = pd.to_datetime(
    validation_df["timestamp"]
)

train_df = train_df.sort_values("timestamp").reset_index(drop=True)
validation_df = validation_df.sort_values("timestamp").reset_index(drop=True)

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_df["label"])
y_validation = encoder.transform(validation_df["label"])

X_train = train_df[FEATURES]
X_validation = validation_df[FEATURES]

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Features:", len(FEATURES))
print("Classes:", list(encoder.classes_))

# ----------------------------------------------------------
# 2. Time-series cross-validation
# ----------------------------------------------------------

# Use trading days as groups so folds never split a trading day.
train_df["trading_date"] = train_df["timestamp"].dt.date

unique_days = np.array(sorted(train_df["trading_date"].unique()))

tscv = TimeSeriesSplit(n_splits=3)

folds = []

for fold_number, (train_day_idx, val_day_idx) in enumerate(
    tscv.split(unique_days), start=1
):
    fold_train_days = set(unique_days[train_day_idx])
    fold_val_days = set(unique_days[val_day_idx])

    train_indices = np.flatnonzero(
        train_df["trading_date"].isin(fold_train_days).to_numpy()
    )

    val_indices = np.flatnonzero(
        train_df["trading_date"].isin(fold_val_days).to_numpy()
    )

    folds.append((train_indices, val_indices))

    print(
        f"Fold {fold_number}: "
        f"Train days={len(train_day_idx)}, "
        f"Validation days={len(val_day_idx)}"
    )

# ----------------------------------------------------------
# 3. Define models and search spaces
# ----------------------------------------------------------

models = {
    "Logistic Regression": (
        lambda params: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
                **params
            ))
        ]),
        {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"]
        }
    ),

    "Random Forest": (
        lambda params: RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            **params
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [10, 15, 20],
            "min_samples_leaf": [5, 10],
            "max_features": ["sqrt"]
        }
    ),

    "XGBoost": (
        lambda params: XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **params
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }
    ),

    "LightGBM": (
        lambda params: LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
            **params
        ),
        {
            "n_estimators": [200, 400],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [15, 31],
            "max_depth": [-1, 10],
            "min_child_samples": [20, 40]
        }
    )
}

# ----------------------------------------------------------
# 4. Evaluate parameter combinations
# ----------------------------------------------------------

results = []
best_models = {}

for model_name, (model_factory, param_grid) in models.items():

    print("\n" + "=" * 65)
    print("TUNING:", model_name)
    print("=" * 65)

    best_score = -1
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):

        fold_scores = []
        fold_down_recall = []
        fold_up_recall = []
        fold_accuracy = []

        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]

            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            model = model_factory(params)

            if model_name == "XGBoost":
                sample_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_fold_train
                )
                model.fit(
                    X_fold_train,
                    y_fold_train,
                    sample_weight=sample_weights
                )
            else:
                model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)

            fold_scores.append(
                f1_score(
                    y_fold_val,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )

            fold_accuracy.append(
                accuracy_score(y_fold_val, predictions)
            )

            recalls = recall_score(
                y_fold_val,
                predictions,
                labels=[0, 1, 2],
                average=None,
                zero_division=0
            )

            # LabelEncoder: DOWN=0, SIDEWAYS=1, UP=2
            fold_down_recall.append(recalls[0])
            fold_up_recall.append(recalls[2])

        mean_f1 = np.mean(fold_scores)
        mean_accuracy = np.mean(fold_accuracy)
        mean_down_recall = np.mean(fold_down_recall)
        mean_up_recall = np.mean(fold_up_recall)

        # Primary objective: Macro F1.
        # Tie-break: balanced directional recall.
        direction_score = (
            mean_down_recall + mean_up_recall
        ) / 2

        results.append({
            "model": model_name,
            "parameters": str(params),
            "cv_macro_f1": mean_f1,
            "cv_accuracy": mean_accuracy,
            "down_recall": mean_down_recall,
            "up_recall": mean_up_recall,
            "direction_recall": direction_score
        })

        if (
            mean_f1 > best_score
            or (
                np.isclose(mean_f1, best_score)
                and best_params is not None
                and direction_score > best_direction_score
            )
        ):
            best_score = mean_f1
            best_direction_score = direction_score
            best_params = params

    # Refit the best configuration on the full training split.
    best_model = model_factory(best_params)

    if model_name == "XGBoost":
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )
        best_model.fit(X_train, y_train, sample_weight=weights)
    else:
        best_model.fit(X_train, y_train)

    best_models[model_name] = best_model

    print("Best parameters:", best_params)
    print(f"Best CV Macro F1: {best_score:.4f}")
    print(f"Directional recall: {best_direction_score:.4f}")

# ----------------------------------------------------------
# 5. Compare configurations
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["cv_macro_f1", "direction_recall"],
    ascending=False
)

results_df.to_csv(
    os.path.join(REPORT_DIR, "hyperparameter_results.csv"),
    index=False
)

print("\n" + "=" * 65)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 65)

print(
    results_df.groupby("model")[
        [
            "cv_macro_f1",
            "cv_accuracy",
            "down_recall",
            "up_recall",
            "direction_recall"
        ]
    ].max().sort_values("cv_macro_f1", ascending=False)
)

# ----------------------------------------------------------
# 6. Evaluate tuned models on existing validation data
# ----------------------------------------------------------

validation_results = []

for model_name, model in best_models.items():

    predictions = model.predict(X_validation)

    recalls = recall_score(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    validation_results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_validation, predictions),
        "macro_f1": f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation,
            predictions
        ),
        "down_recall": recalls[0],
        "sideways_recall": recalls[1],
        "up_recall": recalls[2],
        "direction_recall": (recalls[0] + recalls[2]) / 2
    })

validation_results_df = pd.DataFrame(validation_results)

validation_results_df = validation_results_df.sort_values(
    ["macro_f1", "direction_recall"],
    ascending=False
)

print("\n" + "=" * 65)
print("TUNED MODEL VALIDATION COMPARISON")
print("=" * 65)

print(validation_results_df.to_string(index=False))

validation_results_df.to_csv(
    os.path.join(REPORT_DIR, "tuned_validation_results.csv"),
    index=False
)

# ----------------------------------------------------------
# 7. Save best tuned model based on validation Macro F1
# ----------------------------------------------------------

best_name = validation_results_df.iloc[0]["model"]
best_tuned_model = best_models[best_name]

os.makedirs("models/saved", exist_ok=True)

joblib.dump(
    best_tuned_model,
    "models/saved/best_tuned_model.joblib"
)

joblib.dump(
    encoder,
    "models/saved/tuned_label_encoder.joblib"
)

print("\nBest tuned model:", best_name)
print("Saved: models/saved/best_tuned_model.joblib")
print("Saved: models/saved/tuned_label_encoder.joblib")
print("Saved: reports/hyperparameter_results.csv")
print("Saved: reports/tuned_validation_results.csv")

print("\nHYPERPARAMETER TUNING COMPLETED")
