
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - SIGNAL CONFIDENCE EXPERIMENT (CORRECTED)
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 3

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"

REPORT_PATH = "reports/signal_confidence_thresholds.csv"

THRESHOLDS = [
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75
]

CLASSES = ["DOWN", "SIDEWAYS", "UP"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required = FEATURES + ["label", "timestamp"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=required).copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    unknown = set(df["label"].unique()) - set(CLASSES)

    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    df["target"] = df["label"].map(CLASS_TO_ID).astype(int)

    return df


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name):

    if model_name == "LogisticRegression":

        return Pipeline([
            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ])

    if model_name == "XGBoost":

        return XGBClassifier(
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
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# MODEL TRAINING
# ============================================================

def fit_model(model_name, model, X, y):

    if model_name == "LogisticRegression":

        model.fit(X, y)

    else:

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y
        )

        model.fit(
            X,
            y,
            sample_weight=weights
        )

    return model


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def get_probabilities(model, X):

    probabilities = model.predict_proba(X)

    ordered = np.zeros((len(X), 3))

    for column, class_id in enumerate(model.classes_):

        ordered[:, int(class_id)] = probabilities[:, column]

    return ordered


# ============================================================
# SIGNAL CONFIDENCE EVALUATION
# ============================================================

def evaluate_thresholds(
    model_name,
    dataset_name,
    y_true,
    probabilities
):

    predicted_classes = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    rows = []

    actual_directional_mask = np.isin(
        y_true,
        [
            CLASS_TO_ID["DOWN"],
            CLASS_TO_ID["UP"]
        ]
    )

    actual_directional_count = int(
        actual_directional_mask.sum()
    )

    for threshold in THRESHOLDS:

        # Only UP/DOWN predictions can generate alerts.
        alert_mask = (
            (predicted_classes != CLASS_TO_ID["SIDEWAYS"]) &
            (confidence >= threshold)
        )

        alert_indices = np.where(alert_mask)[0]

        alert_count = len(alert_indices)

        if alert_count == 0:
            continue

        actual = y_true[alert_indices]

        predicted = predicted_classes[alert_indices]

        correct_mask = actual == predicted

        correct_alerts = int(correct_mask.sum())

        incorrect_alerts = alert_count - correct_alerts

        alert_accuracy = (
            correct_alerts / alert_count
        )

        # Correct directional capture:
        # Actual UP/DOWN records correctly identified
        # by an UP/DOWN prediction.
        captured_directional_count = int(
            (
                actual_directional_mask[alert_indices] &
                correct_mask
            ).sum()
        )

        directional_capture = (
            captured_directional_count / actual_directional_count
            if actual_directional_count else 0
        )

        # UP metrics
        up_mask = predicted == CLASS_TO_ID["UP"]

        up_count = int(up_mask.sum())

        up_correct = int(
            (
                actual[up_mask] == CLASS_TO_ID["UP"]
            ).sum()
        )

        up_precision = (
            up_correct / up_count
            if up_count else np.nan
        )

        # DOWN metrics
        down_mask = predicted == CLASS_TO_ID["DOWN"]

        down_count = int(down_mask.sum())

        down_correct = int(
            (
                actual[down_mask] == CLASS_TO_ID["DOWN"]
            ).sum()
        )

        down_precision = (
            down_correct / down_count
            if down_count else np.nan
        )

        # Actual directional alerts, regardless of correctness.
        captured_directional_predictions = int(
            actual_directional_mask[alert_indices].sum()
        )

        rows.append({

            "model": model_name,
            "dataset": dataset_name,

            "threshold": threshold,

            "alert_count": alert_count,

            "correct_alerts": correct_alerts,

            "incorrect_alerts": incorrect_alerts,

            "alert_coverage_pct": (
                alert_count / len(y_true) * 100
            ),

            "alert_accuracy": alert_accuracy,

            "up_alerts": up_count,

            "down_alerts": down_count,

            "up_correct": up_correct,

            "down_correct": down_correct,

            "up_precision": up_precision,

            "down_precision": down_precision,

            "actual_directional_count": actual_directional_count,

            "captured_directional_predictions": (
                captured_directional_predictions
            ),

            "correctly_captured_directional_count": (
                captured_directional_count
            ),

            "directional_capture_pct": (
                directional_capture * 100
            ),

            "mean_alert_confidence": float(
                confidence[alert_indices].mean()
            )
        })

    return rows


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\nMOSAIC 2.0 - CORRECTED SIGNAL CONFIDENCE EXPERIMENT")
    print("=" * 75)

    print("\nLoading datasets...")

    train_df = load_data(TRAIN_PATH)

    validation_df = load_data(VALIDATION_PATH)

    X_train = train_df[FEATURES]

    y_train = train_df["target"].to_numpy()

    X_validation = validation_df[FEATURES]

    y_validation = validation_df["target"].to_numpy()

    unique_dates = np.array(
        sorted(
            train_df["timestamp"].dt.date.unique()
        )
    )

    print("Training records:", len(train_df))

    print("Validation records:", len(validation_df))

    print("Features:", len(FEATURES))

    print("Training days:", len(unique_dates))

    print("\nGenerating chronological out-of-fold predictions...")

    print("=" * 75)

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    model_names = [
        "LogisticRegression",
        "XGBoost"
    ]

    oof_probabilities = {

        name: np.full(
            (len(train_df), 3),
            np.nan
        )

        for name in model_names
    }

    # --------------------------------------------------------
    # CHRONOLOGICAL CROSS-VALIDATION
    # --------------------------------------------------------

    for fold_number, (
        train_day_idx,
        val_day_idx
    ) in enumerate(
        splitter.split(unique_dates),
        start=1
    ):

        train_dates = set(
            unique_dates[train_day_idx]
        )

        val_dates = set(
            unique_dates[val_day_idx]
        )

        train_indices = np.flatnonzero(
            train_df["timestamp"]
            .dt.date
            .isin(train_dates)
            .to_numpy()
        )

        val_indices = np.flatnonzero(
            train_df["timestamp"]
            .dt.date
            .isin(val_dates)
            .to_numpy()
        )

        print(
            f"\nFold {fold_number}: "
            f"{len(train_indices)} training / "
            f"{len(val_indices)} held-out records"
        )

        X_fold_train = X_train.iloc[train_indices]

        y_fold_train = y_train[train_indices]

        X_fold_val = X_train.iloc[val_indices]

        for model_name in model_names:

            print(f"Training {model_name}...")

            model = create_model(model_name)

            model = fit_model(
                model_name,
                model,
                X_fold_train,
                y_fold_train
            )

            probabilities = get_probabilities(
                model,
                X_fold_val
            )

            oof_probabilities[
                model_name
            ][val_indices] = probabilities

    # --------------------------------------------------------
    # OUT-OF-FOLD EVALUATION
    # --------------------------------------------------------

    all_results = []

    print("\n\nOUT-OF-FOLD SIGNAL EVALUATION")

    print("=" * 75)

    for model_name in model_names:

        valid_mask = ~np.isnan(
            oof_probabilities[model_name]
        ).any(axis=1)

        y_oof = y_train[valid_mask]

        p_oof = oof_probabilities[
            model_name
        ][valid_mask]

        print(
            f"\n{model_name}: "
            f"{len(y_oof)} evaluated records"
        )

        rows = evaluate_thresholds(
            model_name,
            "OutOfFold",
            y_oof,
            p_oof
        )

        all_results.extend(rows)

    # --------------------------------------------------------
    # SECONDARY VALIDATION EVALUATION
    # --------------------------------------------------------

    print("\n\nSECONDARY VALIDATION EVALUATION")

    print("=" * 75)

    for model_name in model_names:

        print(f"\nTraining full model: {model_name}")

        model = create_model(model_name)

        model = fit_model(
            model_name,
            model,
            X_train,
            y_train
        )

        probabilities = get_probabilities(
            model,
            X_validation
        )

        rows = evaluate_thresholds(
            model_name,
            "Validation",
            y_validation,
            probabilities
        )

        all_results.extend(rows)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = pd.DataFrame(all_results)

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\n\nCORRECTED SIGNAL CONFIDENCE SUMMARY")

    print("=" * 120)

    if not report.empty:

        display_columns = [
            "model",
            "dataset",
            "threshold",
            "alert_count",
            "alert_coverage_pct",
            "alert_accuracy",
            "up_alerts",
            "down_alerts",
            "up_precision",
            "down_precision",
            "correctly_captured_directional_count",
            "directional_capture_pct",
            "mean_alert_confidence"
        ]

        print(
            report[display_columns]
            .round(4)
            .to_string(index=False)
        )

    print("\nReport saved to:", REPORT_PATH)

    print("\nEXPERIMENT SAFETY CHECK")

    print("- Training and validation data only.")

    print("- Locked test dataset was not loaded.")

    print("- Existing model artifacts were not modified.")

    print("- No trading thresholds have been selected.")

    print("- No transaction costs or slippage are included.")

    print("- Results do not establish profitability.")


if __name__ == "__main__":

    main()
