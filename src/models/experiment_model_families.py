
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - MODEL FAMILY EXPERIMENT
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 3

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"
REPORT_PATH = "reports/model_family_experiment.csv"

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram", "ATR_14",
    "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

CLASSES = ["DOWN", "SIDEWAYS", "UP"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}


def load_dataset(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required = FEATURES + ["label", "timestamp"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.dropna(subset=required).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df[FEATURES] = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    unknown = set(df["label"].unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    df["target"] = df["label"].map(CLASS_TO_ID).astype(int)

    return df


def make_models():
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0
        ),

        "LightGBM": LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=-1,
            reg_lambda=2.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1
        )
    }


def fit_model(model_name, model, X, y):
    if model_name == "LogisticRegression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline.fit(X, y)
        return pipeline

    if model_name == "RandomForest":
        model.fit(X, y)
        return model

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y
    )

    model.fit(X, y, sample_weight=weights)
    return model


def calculate_metrics(y_true, y_pred):
    recalls = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "down_recall": recalls[0],
        "sideways_recall": recalls[1],
        "up_recall": recalls[2],
        "direction_recall": (recalls[0] + recalls[2]) / 2
    }


def main():
    print("\nMOSAIC 2.0 - MODEL FAMILY EXPERIMENT")
    print("=" * 65)

    print("\nLoading datasets...")

    train_df = load_dataset(TRAIN_PATH)
    validation_df = load_dataset(VALIDATION_PATH)

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df["target"]

    train_dates = train_df["timestamp"].dt.date
    unique_dates = np.array(sorted(train_dates.unique()))

    print(f"Training records:   {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")
    print(f"Features:           {len(FEATURES)}")
    print(f"Training days:      {len(unique_dates)}")

    print("\nTraining class distribution:")
    for class_name in CLASSES:
        count = (train_df["label"] == class_name).sum()
        print(f"{class_name:12}: {count:7} ({count / len(train_df):.2%})")

    # Create chronological folds using whole trading days.
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []

    for fold_number, (train_day_idx, val_day_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        fold_train_dates = set(unique_dates[train_day_idx])
        fold_val_dates = set(unique_dates[val_day_idx])

        train_indices = np.flatnonzero(
            train_dates.isin(fold_train_dates).to_numpy()
        )
        val_indices = np.flatnonzero(
            train_dates.isin(fold_val_dates).to_numpy()
        )

        folds.append((train_indices, val_indices))

        print(
            f"Fold {fold_number}: "
            f"{len(train_indices)} train / "
            f"{len(val_indices)} validation records"
        )

    results = []
    models = make_models()

    print("\nStarting chronological cross-validation...")
    print("=" * 65)

    for model_name, base_model in models.items():
        print(f"\nMODEL: {model_name}")
        fold_metrics = []

        for fold_number, (train_idx, val_idx) in enumerate(
            folds, start=1
        ):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]

            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Create a fresh model for every fold.
            fold_model = make_models()[model_name]

            try:
                fold_model = fit_model(
                    model_name,
                    fold_model,
                    X_fold_train,
                    y_fold_train
                )

                predictions = fold_model.predict(X_fold_val)
                metrics = calculate_metrics(
                    y_fold_val,
                    predictions
                )

                fold_metrics.append(metrics)

                print(
                    f"Fold {fold_number}: "
                    f"Macro F1={metrics['macro_f1']:.4f} | "
                    f"Balanced Acc={metrics['balanced_accuracy']:.4f} | "
                    f"Direction Recall={metrics['direction_recall']:.4f}"
                )

            except Exception as error:
                print(f"Fold {fold_number} FAILED: {error}")
                fold_metrics = []
                break

        if not fold_metrics:
            print(f"Skipping {model_name} due to experiment failure.")
            continue

        # Cross-validation mean and standard deviation.
        summary = {"model": model_name}

        for metric_name in fold_metrics[0]:
            values = [m[metric_name] for m in fold_metrics]

            summary[f"cv_{metric_name}_mean"] = np.mean(values)
            summary[f"cv_{metric_name}_std"] = np.std(values)

        # Secondary validation evaluation using all training records.
        print("Fitting on full training set for secondary validation...")

        try:
            final_model = make_models()[model_name]

            final_model = fit_model(
                model_name,
                final_model,
                X_train,
                y_train
            )

            validation_predictions = final_model.predict(
                X_validation
            )

            validation_metrics = calculate_metrics(
                y_validation,
                validation_predictions
            )

            for metric_name, value in validation_metrics.items():
                summary[f"validation_{metric_name}"] = value

            print(
                f"Validation Macro F1: "
                f"{validation_metrics['macro_f1']:.4f}"
            )

            results.append(summary)

        except Exception as error:
            print(f"Validation evaluation FAILED: {error}")

    if not results:
        raise RuntimeError("No model completed successfully.")

    report = pd.DataFrame(results)

    report = report.sort_values(
        by="cv_macro_f1_mean",
        ascending=False
    )

    os.makedirs("reports", exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\n\nFINAL MODEL FAMILY COMPARISON")
    print("=" * 100)

    display_columns = [
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_balanced_accuracy_mean",
        "cv_direction_recall_mean",
        "cv_down_recall_mean",
        "cv_sideways_recall_mean",
        "cv_up_recall_mean",
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_direction_recall"
    ]

    print(
        report[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nReport saved to:", REPORT_PATH)

    print("\nIMPORTANT:")
    print("- Existing MOSAIC model artifacts were not modified.")
    print("- The test dataset was not loaded or evaluated.")
    print("- Validation is for model development, not final proof.")
    print("- Higher Macro F1 alone does not establish trading profitability.")


if __name__ == "__main__":
    main()
