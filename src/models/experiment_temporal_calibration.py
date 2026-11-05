
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    recall_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - TEMPORAL PROBABILITY CALIBRATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/splits/train.csv"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.csv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.20

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day"
]

CLASS_NAMES = {
    0: "DOWN",
    1: "SIDEWAYS",
    2: "UP"
}

CLASSES = np.array([0, 1, 2])


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    target_col = find_column(
        df,
        ["target", "label", "direction", "class", "target_class"],
        "target"
    )

    date_col = find_column(
        df,
        ["timestamp", "datetime", "date", "trading_date"],
        "date/timestamp"
    )

    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    # Normalize labels to the established MOSAIC encoding.
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        mapping = {
            "DOWN": 0,
            "SIDEWAYS": 1,
            "UP": 2
        }

        df[target_col] = (
            df[target_col].astype(str).str.upper().map(mapping)
        )

    df[target_col] = pd.to_numeric(
        df[target_col], errors="coerce"
    )

    if df[target_col].isna().any():
        raise ValueError("Invalid target labels found.")

    df[target_col] = df[target_col].astype(int)

    if not set(df[target_col].unique()).issubset({0, 1, 2}):
        raise ValueError(
            f"Unexpected target encoding: {sorted(df[target_col].unique())}"
        )

    df = df.sort_values(date_col).reset_index(drop=True)

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(
            "Missing or infinite feature values detected. "
            "Resolve them before calibration."
        )

    y = df[target_col].to_numpy()

    # Use calendar trading dates for the chronological split.
    dates = df[date_col].dt.date.to_numpy()

    return X, y, dates


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])


def fit_sigmoid_calibrators(probabilities, y):
    calibrators = []

    for class_id in CLASSES:
        binary_target = (y == class_id).astype(int)

        if len(np.unique(binary_target)) < 2:
            raise ValueError(
                f"Calibration period does not contain both outcomes "
                f"for class {CLASS_NAMES[class_id]}."
            )

        calibrator = LogisticRegression(
            C=1e6,
            max_iter=2000,
            random_state=RANDOM_STATE
        )

        calibrator.fit(
            probabilities[:, [class_id]],
            binary_target
        )

        calibrators.append(calibrator)

    return calibrators


def apply_sigmoid_calibrators(probabilities, calibrators):
    calibrated = np.column_stack([
        calibrators[i].predict_proba(
            probabilities[:, [i]]
        )[:, 1]
        for i in range(len(CLASSES))
    ])

    return normalize_probabilities(calibrated)


def fit_isotonic_calibrators(probabilities, y):
    calibrators = []

    for class_id in CLASSES:
        binary_target = (y == class_id).astype(int)

        if len(np.unique(binary_target)) < 2:
            raise ValueError(
                f"Calibration period does not contain both outcomes "
                f"for class {CLASS_NAMES[class_id]}."
            )

        calibrator = IsotonicRegression(
            out_of_bounds="clip"
        )

        calibrator.fit(
            probabilities[:, class_id],
            binary_target
        )

        calibrators.append(calibrator)

    return calibrators


def apply_isotonic_calibrators(probabilities, calibrators):
    calibrated = np.column_stack([
        calibrators[i].predict(probabilities[:, i])
        for i in range(len(CLASSES))
    ])

    return normalize_probabilities(calibrated)


def normalize_probabilities(probabilities):
    probabilities = np.clip(probabilities, 1e-10, None)

    row_sums = probabilities.sum(axis=1, keepdims=True)

    return probabilities / row_sums


def calculate_ece(y_true, probabilities, bins=10):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(float)

    ece = 0.0

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for i in range(bins):
        if i == bins - 1:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence < bin_edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        average_confidence = confidence[mask].mean()
        average_accuracy = correctness[mask].mean()

        ece += (
            mask.mean() *
            abs(average_confidence - average_accuracy)
        )

    return ece


def evaluate_model(name, y_true, probabilities):
    predictions = probabilities.argmax(axis=1)

    directional_mask = np.isin(y_true, [0, 2])
    directional_predictions = np.isin(predictions, [0, 2])

    directional_precision = (
        (predictions[directional_mask] == y_true[directional_mask]).mean()
        if directional_mask.any()
        else 0.0
    )

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "log_loss": log_loss(
            y_true, probabilities, labels=CLASSES
        ),
        "multiclass_brier": np.mean(
            np.sum(
                (probabilities - np.eye(3)[y_true]) ** 2,
                axis=1
            )
        ),
        "ece": calculate_ece(y_true, probabilities),
        "DOWN_recall": recall_score(
            y_true, predictions, labels=[0],
            average=None, zero_division=0
        )[0],
        "SIDEWAYS_recall": recall_score(
            y_true, predictions, labels=[1],
            average=None, zero_division=0
        )[0],
        "UP_recall": recall_score(
            y_true, predictions, labels=[2],
            average=None, zero_division=0
        )[0],
        "directional_precision": directional_precision,
        "directional_prediction_pct": directional_predictions.mean() * 100
    }


def main():
    print("\nMOSAIC 2.0 - TEMPORAL CALIBRATION EXPERIMENT")
    print("=" * 75)

    print("\nLoading datasets...")

    X_train, y_train, train_dates = load_dataset(TRAIN_PATH)
    X_val, y_val, _ = load_dataset(VALIDATION_PATH)

    unique_dates = np.sort(np.unique(train_dates))

    split_index = int(
        len(unique_dates) * (1 - CALIBRATION_FRACTION)
    )

    model_dates = unique_dates[:split_index]
    calibration_dates = unique_dates[split_index:]

    model_mask = np.isin(train_dates, model_dates)
    calibration_mask = np.isin(train_dates, calibration_dates)

    X_model = X_train.loc[model_mask]
    y_model = y_train[model_mask]

    X_calibration = X_train.loc[calibration_mask]
    y_calibration = y_train[calibration_mask]

    print(f"Training days:       {len(model_dates)}")
    print(f"Calibration days:    {len(calibration_dates)}")
    print(f"Validation records:  {len(y_val)}")

    print(
        f"\nModel training period: "
        f"{model_dates[0]} to {model_dates[-1]}"
    )

    print(
        f"Calibration period:   "
        f"{calibration_dates[0]} to {calibration_dates[-1]}"
    )

    print("\nTraining base Logistic Regression...")

    model = create_model()
    model.fit(X_model, y_model)

    print("Generating calibration-period probabilities...")

    calibration_probabilities = model.predict_proba(X_calibration)
    validation_probabilities = model.predict_proba(X_val)

    # Reorder probability columns to the established class order.
    model_classes = model.named_steps["classifier"].classes_

    def reorder_probabilities(probabilities):
        ordered = np.zeros((len(probabilities), len(CLASSES)))

        for index, class_id in enumerate(model_classes):
            ordered[:, int(class_id)] = probabilities[:, index]

        return ordered

    calibration_probabilities = reorder_probabilities(
        calibration_probabilities
    )

    validation_probabilities = reorder_probabilities(
        validation_probabilities
    )

    print("\nFitting sigmoid calibrators...")

    sigmoid_calibrators = fit_sigmoid_calibrators(
        calibration_probabilities,
        y_calibration
    )

    print("Fitting isotonic calibrators...")

    isotonic_calibrators = fit_isotonic_calibrators(
        calibration_probabilities,
        y_calibration
    )

    sigmoid_validation = apply_sigmoid_calibrators(
        validation_probabilities,
        sigmoid_calibrators
    )

    isotonic_validation = apply_isotonic_calibrators(
        validation_probabilities,
        isotonic_calibrators
    )

    print("\nEvaluating validation predictions...")

    results = [
        evaluate_model(
            "Uncalibrated",
            y_val,
            validation_probabilities
        ),
        evaluate_model(
            "Temporal_Sigmoid",
            y_val,
            sigmoid_validation
        ),
        evaluate_model(
            "Temporal_Isotonic",
            y_val,
            isotonic_validation
        )
    ]

    results_df = pd.DataFrame(results)

    print("\nVALIDATION COMPARISON")
    print("=" * 110)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    output_path = REPORT_DIR / "temporal_calibration.csv"

    results_df.to_csv(output_path, index=False)

    print("\nPrediction distribution:")

    for name, probabilities in [
        ("Uncalibrated", validation_probabilities),
        ("Temporal_Sigmoid", sigmoid_validation),
        ("Temporal_Isotonic", isotonic_validation)
    ]:
        predictions = probabilities.argmax(axis=1)

        distribution = pd.Series(predictions).value_counts(
            normalize=True
        ).reindex(CLASSES, fill_value=0)

        print(f"\n{name}")

        for class_id in CLASSES:
            print(
                f"{CLASS_NAMES[class_id]:10s}: "
                f"{distribution[class_id] * 100:.2f}%"
            )

    print("\nCalibration-period class distribution:")

    for class_id in CLASSES:
        count = int((y_calibration == class_id).sum())
        print(f"{CLASS_NAMES[class_id]:10s}: {count}")

    print(f"\nReport saved: {output_path}")

    print("\nIMPORTANT:")
    print("- Model training and calibration used separate chronological periods.")
    print("- Validation was not used to fit the calibrators.")
    print("- Test data was not loaded.")
    print("- No existing model artifacts were overwritten.")
    print("- Results are development evidence, not proof of trading profitability.")


if __name__ == "__main__":
    main()
