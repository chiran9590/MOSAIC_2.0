
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# ============================================================
# MOSAIC 2.0 - TEMPORAL STABILITY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/splits/train.csv"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.csv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

CLASSES = [0, 1, 2]


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    target_col = find_column(
        df,
        ["target", "label", "direction", "class", "target_class"],
        "target"
    )

    date_col = find_column(
        df,
        ["timestamp", "datetime", "date", "trading_date"],
        "timestamp"
    )

    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    if df[date_col].isna().any():
        raise ValueError("Invalid timestamps detected.")

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
        df[target_col],
        errors="coerce"
    )

    if df[target_col].isna().any():
        raise ValueError("Invalid target labels detected.")

    df[target_col] = df[target_col].astype(int)

    if not set(df[target_col].unique()).issubset(set(CLASSES)):
        raise ValueError("Unexpected target encoding.")

    df = df.sort_values(date_col).reset_index(drop=True)

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Missing or infinite feature values detected.")

    y = df[target_col].to_numpy()
    dates = df[date_col].dt.date.to_numpy()

    return X, y, dates


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ])


def evaluate_block(name, y_true, predictions):
    result = {
        "period": name,
        "records": len(y_true),
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
    }

    for class_id, class_name in CLASS_NAMES.items():
        result[f"{class_name}_recall"] = recall_score(
            y_true,
            predictions,
            labels=[class_id],
            average=None,
            zero_division=0
        )[0]

        result[f"{class_name}_precision"] = precision_score(
            y_true,
            predictions,
            labels=[class_id],
            average=None,
            zero_division=0
        )[0]

        result[f"actual_{class_name}_pct"] = (
            np.mean(y_true == class_id) * 100
        )

        result[f"predicted_{class_name}_pct"] = (
            np.mean(predictions == class_id) * 100
        )

    return result


def main():
    print("\nMOSAIC 2.0 - TEMPORAL STABILITY AUDIT")
    print("=" * 85)

    print("\nLoading datasets...")

    X_train, y_train, _ = load_dataset(TRAIN_PATH)
    X_val, y_val, val_dates = load_dataset(VALIDATION_PATH)

    unique_dates = np.sort(np.unique(val_dates))

    print(f"Training records:   {len(y_train)}")
    print(f"Validation records: {len(y_val)}")
    print(f"Validation days:    {len(unique_dates)}")

    print("\nTraining balanced Logistic Regression on full training data...")

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    # Divide validation trading dates into five chronological blocks.
    date_blocks = np.array_split(unique_dates, 5)

    results = []
    prediction_records = []

    print("\nCHRONOLOGICAL VALIDATION BLOCKS")
    print("=" * 85)

    for index, block_dates in enumerate(date_blocks, start=1):
        mask = np.isin(val_dates, block_dates)

        block_y = y_val[mask]
        block_predictions = predictions[mask]

        period_name = (
            f"Block_{index}: "
            f"{block_dates[0]} to {block_dates[-1]}"
        )

        result = evaluate_block(
            period_name,
            block_y,
            block_predictions
        )

        results.append(result)

        print(f"\n{period_name}")
        print(f"Records:             {len(block_y)}")
        print(f"Accuracy:            {result['accuracy']:.4f}")
        print(f"Macro F1:            {result['macro_f1']:.4f}")
        print(f"Balanced accuracy:   {result['balanced_accuracy']:.4f}")

        for class_name in ["DOWN", "SIDEWAYS", "UP"]:
            print(
                f"{class_name:10s} "
                f"Recall: {result[f'{class_name}_recall']:.4f} | "
                f"Actual: {result[f'actual_{class_name}_pct']:.2f}% | "
                f"Predicted: {result[f'predicted_{class_name}_pct']:.2f}%"
            )

        prediction_records.append(
            pd.DataFrame({
                "date": np.array(val_dates)[mask],
                "actual": block_y,
                "predicted": block_predictions,
                "block": index
            })
        )

    results_df = pd.DataFrame(results)

    results_path = REPORT_DIR / "temporal_stability_metrics.csv"
    predictions_path = REPORT_DIR / "temporal_stability_predictions.csv"

    results_df.to_csv(results_path, index=False)

    pd.concat(
        prediction_records,
        ignore_index=True
    ).to_csv(predictions_path, index=False)

    print("\nOVERALL BLOCK COMPARISON")
    print("=" * 85)

    columns = [
        "period",
        "records",
        "macro_f1",
        "balanced_accuracy",
        "DOWN_recall",
        "SIDEWAYS_recall",
        "UP_recall"
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nCONFUSION MATRIX - FULL VALIDATION")

    print(
        pd.DataFrame(
            confusion_matrix(
                y_val,
                predictions,
                labels=CLASSES
            ),
            index=["Actual_DOWN", "Actual_SIDEWAYS", "Actual_UP"],
            columns=["Pred_DOWN", "Pred_SIDEWAYS", "Pred_UP"]
        )
    )

    print("\nREPORTS SAVED")
    print(results_path)
    print(predictions_path)

    print("\nIMPORTANT")
    print("- Only training and validation data were used.")
    print("- Test data was not loaded.")
    print("- No existing model artifacts were modified.")
    print("- This is a development stability diagnostic.")


if __name__ == "__main__":
    main()
