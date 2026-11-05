
import os
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    balanced_accuracy_score
)

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - SELECTED MODEL HYPERPARAMETER TUNING
# ============================================================

print("\nMOSAIC 2.0 - SELECTED MODEL TUNING")
print("=" * 70)

RANDOM_STATE = 42
DATA_PATH = "data/processed/advanced_splits/train.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. DEFINE FEATURE SETS
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

feature_sets = {
    "22 Original": original_features,
    "18 Selected": selected_features
}

# ------------------------------------------------------------
# 2. LOAD TRAINING DATA ONLY
# ------------------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

print("Training records:", len(df))
print("Unique trading days:", df["timestamp"].dt.date.nunique())

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

print("\nClass encoding:")
for i, class_name in enumerate(encoder.classes_):
    print(i, "=", class_name)

# ------------------------------------------------------------
# 3. CREATE CHRONOLOGICAL FOLDS
# ------------------------------------------------------------

print("\nCreating 3 chronological folds...")

df["trading_day"] = df["timestamp"].dt.date

unique_days = np.array(
    sorted(df["trading_day"].unique())
)

time_splitter = TimeSeriesSplit(n_splits=3)

folds = []

for fold_number, (train_day_idx, val_day_idx) in enumerate(
    time_splitter.split(unique_days),
    start=1
):

    train_days = set(unique_days[train_day_idx])
    val_days = set(unique_days[val_day_idx])

    train_indices = np.flatnonzero(
        df["trading_day"].isin(train_days).to_numpy()
    )

    val_indices = np.flatnonzero(
        df["trading_day"].isin(val_days).to_numpy()
    )

    folds.append((train_indices, val_indices))

    print(f"\nFold {fold_number}")
    print("Training days:", len(train_days))
    print("Validation days:", len(val_days))
    print("Training records:", len(train_indices))
    print("Validation records:", len(val_indices))
    print("Training period:", min(train_days), "to", max(train_days))
    print("Validation period:", min(val_days), "to", max(val_days))

# ------------------------------------------------------------
# 4. DEFINE HYPERPARAMETER GRIDS
# ------------------------------------------------------------

lr_parameter_grid = {
    "C": [0.01, 0.1, 1.0, 10.0]
}

lgbm_parameter_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [15, 31],
    "max_depth": [-1, 10]
}

lgbm_combinations = list(
    itertools.product(
        lgbm_parameter_grid["n_estimators"],
        lgbm_parameter_grid["learning_rate"],
        lgbm_parameter_grid["num_leaves"],
        lgbm_parameter_grid["max_depth"]
    )
)

print("\nSEARCH SPACE")
print("-" * 50)
print("Logistic Regression configurations:", len(lr_parameter_grid["C"]))
print("LightGBM configurations:", len(lgbm_combinations))
print("Total configurations:", 2 * (
    len(lr_parameter_grid["C"]) + len(lgbm_combinations)
))
print("Total model fits:", 2 * (
    len(lr_parameter_grid["C"]) + len(lgbm_combinations)
) * len(folds))

# ------------------------------------------------------------
# 5. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(y_true, predictions):

    recalls = recall_score(
        y_true,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    down_recall = recalls[0]
    sideways_recall = recalls[1]
    up_recall = recalls[2]

    return {
        "accuracy": accuracy_score(y_true, predictions),

        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
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
# 6. MODEL BUILDERS
# ------------------------------------------------------------

def build_lr(C):

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])


def build_lgbm(n_estimators, learning_rate, num_leaves, max_depth):

    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=30,
        objective="multiclass",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    )

# ------------------------------------------------------------
# 7. RUN CROSS-VALIDATION
# ------------------------------------------------------------

all_results = []

total_experiments = 2 * (
    len(lr_parameter_grid["C"]) + len(lgbm_combinations)
)

experiment_number = 0

for feature_set_name, features in feature_sets.items():

    X = df[features]

    print("\n" + "=" * 70)
    print("FEATURE SET:", feature_set_name)
    print("Features:", len(features))
    print("=" * 70)

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    print("\nTUNING LOGISTIC REGRESSION")

    for C in lr_parameter_grid["C"]:

        experiment_number += 1
        fold_results = []

        print(
            f"\nExperiment {experiment_number}/{total_experiments}"
        )
        print("C =", C)

        for fold_number, (train_idx, val_idx) in enumerate(
            folds,
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            model = build_lr(C)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            metrics = calculate_metrics(y_val, predictions)

            fold_results.append(metrics)

            print(
                f"Fold {fold_number}: "
                f"Macro F1={metrics['macro_f1']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Direction Recall={metrics['direction_recall']:.4f}"
            )

        average_metrics = pd.DataFrame(fold_results).mean().to_dict()

        all_results.append({
            "feature_set": feature_set_name,
            "model": "Logistic Regression",
            "parameters": f"C={C}",
            "C": C,
            "n_estimators": np.nan,
            "learning_rate": np.nan,
            "num_leaves": np.nan,
            "max_depth": np.nan,
            **average_metrics
        })

    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    print("\nTUNING LIGHTGBM")

    for n_estimators, learning_rate, num_leaves, max_depth in lgbm_combinations:

        experiment_number += 1
        fold_results = []

        print(
            f"\nExperiment {experiment_number}/{total_experiments}"
        )

        print(
            f"Trees={n_estimators}, "
            f"Learning Rate={learning_rate}, "
            f"Leaves={num_leaves}, "
            f"Depth={max_depth}"
        )

        for fold_number, (train_idx, val_idx) in enumerate(
            folds,
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            model = build_lgbm(
                n_estimators,
                learning_rate,
                num_leaves,
                max_depth
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            metrics = calculate_metrics(y_val, predictions)

            fold_results.append(metrics)

            print(
                f"Fold {fold_number}: "
                f"Macro F1={metrics['macro_f1']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Direction Recall={metrics['direction_recall']:.4f}"
            )

        average_metrics = pd.DataFrame(fold_results).mean().to_dict()

        all_results.append({
            "feature_set": feature_set_name,
            "model": "LightGBM",
            "parameters": (
                f"trees={n_estimators}, "
                f"lr={learning_rate}, "
                f"leaves={num_leaves}, "
                f"depth={max_depth}"
            ),
            "C": np.nan,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            **average_metrics
        })

# ------------------------------------------------------------
# 8. SAVE ALL RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["macro_f1", "direction_recall"],
    ascending=False
)

output_path = f"{REPORT_DIR}/selected_model_cv_results.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

display_columns = [
    "feature_set",
    "model",
    "parameters",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "down_recall",
    "sideways_recall",
    "up_recall",
    "direction_recall"
]

print(
    results_df[display_columns].head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# 9. BEST CONFIGURATION FOR EACH COMBINATION
# ------------------------------------------------------------

print("\nBEST CONFIGURATION FOR EACH MODEL AND FEATURE SET")
print("-" * 70)

for feature_set_name in feature_sets:

    for model_name in ["Logistic Regression", "LightGBM"]:

        subset = results_df[
            (results_df["feature_set"] == feature_set_name)
            & (results_df["model"] == model_name)
        ]

        best = subset.sort_values(
            by=["macro_f1", "direction_recall"],
            ascending=False
        ).iloc[0]

        print(f"\n{feature_set_name} - {model_name}")
        print("Parameters:", best["parameters"])
        print(f"CV Accuracy: {best['accuracy']:.4f}")
        print(f"CV Macro F1: {best['macro_f1']:.4f}")
        print(f"CV Balanced Accuracy: {best['balanced_accuracy']:.4f}")
        print(f"DOWN Recall: {best['down_recall']:.4f}")
        print(f"SIDEWAYS Recall: {best['sideways_recall']:.4f}")
        print(f"UP Recall: {best['up_recall']:.4f}")
        print(f"Direction Recall: {best['direction_recall']:.4f}")

print("\n" + "=" * 70)
print("TUNING COMPLETED")
print("=" * 70)

print("Results saved:", output_path)
print("Only training data was used.")
print("Validation data was NOT used.")
print("Test data was NOT used.")
