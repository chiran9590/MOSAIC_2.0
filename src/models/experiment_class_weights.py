
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score
)

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - CLASS WEIGHT EXPERIMENT
# ============================================================

TRAIN_PATH = "data/processed/splits/train.csv"
VAL_PATH = "data/processed/splits/validation.csv"
REPORT_DIR = "reports"

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

CLASSES = ["DOWN", "SIDEWAYS", "UP"]

EXPERIMENTS = {
    "Unweighted": None,

    "Balanced": "balanced",

    "Directional_1.25": {
        "DOWN": 1.25,
        "SIDEWAYS": 1.0,
        "UP": 1.25
    },

    "Directional_1.50": {
        "DOWN": 1.50,
        "SIDEWAYS": 1.0,
        "UP": 1.50
    },

    "Directional_2.00": {
        "DOWN": 2.00,
        "SIDEWAYS": 1.0,
        "UP": 2.00
    }
}


def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = FEATURES + ["timestamp", "label"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps found in {path}"
        )

    X = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    if X.isna().any().any():
        raise ValueError(
            f"Missing or infinite feature values in {path}"
        )

    y = df["label"].astype(str)

    unexpected = set(y.unique()) - set(CLASSES)

    if unexpected:
        raise ValueError(
            f"Unexpected labels: {unexpected}"
        )

    return df, X, y


def build_model(class_weight):

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                class_weight=class_weight,
                max_iter=3000,
                solver="lbfgs",
                random_state=42
            )
        )
    ])


def calculate_metrics(y_true, predictions):

    recalls = recall_score(
        y_true,
        predictions,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(
            y_true, predictions
        ),

        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),

        "down_recall": recalls[0],

        "sideways_recall": recalls[1],

        "up_recall": recalls[2],

        "direction_recall": (
            recalls[0] + recalls[2]
        ) / 2
    }


def main():

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\nMOSAIC 2.0 - CLASS WEIGHT EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_df, X_train, y_train = load_data(
        TRAIN_PATH
    )

    val_df, X_val, y_val = load_data(
        VAL_PATH
    )

    print("\nDATASET INFORMATION")
    print("-" * 70)

    print("Training records:", len(train_df))
    print("Validation records:", len(val_df))
    print("Features:", len(FEATURES))

    train_df["trading_date"] = (
        train_df["timestamp"].dt.date
    )

    unique_dates = np.array(
        sorted(train_df["trading_date"].unique())
    )

    print("Training trading days:", len(unique_dates))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # CHRONOLOGICAL CROSS VALIDATION
    # --------------------------------------------------------

    splitter = TimeSeriesSplit(
        n_splits=3
    )

    results = []

    print("\nSTARTING CHRONOLOGICAL CROSS VALIDATION")
    print("=" * 70)

    for name, weights in EXPERIMENTS.items():

        print(f"\nEXPERIMENT: {name}")
        print("-" * 70)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(unique_dates),
            start=1
        ):

            train_dates = set(
                unique_dates[train_idx]
            )

            validation_dates = set(
                unique_dates[val_idx]
            )

            train_mask = (
                train_df["trading_date"]
                .isin(train_dates)
                .values
            )

            validation_mask = (
                train_df["trading_date"]
                .isin(validation_dates)
                .values
            )

            X_fold_train = X_train.loc[
                train_mask
            ]

            y_fold_train = y_train.loc[
                train_mask
            ]

            X_fold_validation = X_train.loc[
                validation_mask
            ]

            y_fold_validation = y_train.loc[
                validation_mask
            ]

            model = build_model(weights)

            model.fit(
                X_fold_train,
                y_fold_train
            )

            predictions = model.predict(
                X_fold_validation
            )

            metrics = calculate_metrics(
                y_fold_validation,
                predictions
            )

            fold_results.append(metrics)

            print(
                f"Fold {fold} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Balanced Accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"DOWN: {metrics['down_recall']:.4f} | "
                f"SIDEWAYS: {metrics['sideways_recall']:.4f} | "
                f"UP: {metrics['up_recall']:.4f}"
            )

        cv_metrics = (
            pd.DataFrame(fold_results)
            .mean()
            .to_dict()
        )

        # ----------------------------------------------------
        # SECONDARY VALIDATION EVALUATION
        # ----------------------------------------------------

        final_model = build_model(weights)

        final_model.fit(
            X_train,
            y_train
        )

        val_predictions = final_model.predict(
            X_val
        )

        validation_metrics = calculate_metrics(
            y_val,
            val_predictions
        )

        row = {
            "experiment": name
        }

        for metric, value in cv_metrics.items():
            row[f"cv_{metric}"] = value

        for metric, value in validation_metrics.items():
            row[f"validation_{metric}"] = value

        results.append(row)

        print("\nCross-validation mean Macro F1:",
              round(cv_metrics["macro_f1"], 4))

        print("Validation Macro F1:",
              round(validation_metrics["macro_f1"], 4))

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "cv_macro_f1",
        ascending=False
    )

    output_path = os.path.join(
        REPORT_DIR,
        "class_weight_experiment.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\nFINAL EXPERIMENT COMPARISON")
    print("=" * 110)

    columns = [
        "experiment",
        "cv_macro_f1",
        "cv_balanced_accuracy",
        "cv_direction_recall",
        "cv_down_recall",
        "cv_sideways_recall",
        "cv_up_recall",
        "validation_macro_f1"
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nREPORT SAVED:")
    print(output_path)

    print("\nIMPORTANT:")
    print("- Existing test data was not loaded.")
    print("- No existing model was overwritten.")
    print("- No model has been selected for deployment.")
    print("- Results are experimental, not proof of profitability.")


if __name__ == "__main__":
    main()
