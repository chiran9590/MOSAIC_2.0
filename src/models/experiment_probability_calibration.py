
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ============================================================
# MOSAIC 2.0 - PROBABILITY CALIBRATION EXPERIMENT
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 3

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"

REPORT_PATH = "reports/probability_calibration.csv"
PLOT_PATH = "reports/probability_reliability.png"

CLASSES = ["DOWN", "SIDEWAYS", "UP"]

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASSES)
}

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
# LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required = FEATURES + ["label", "timestamp"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=required).copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["target"] = df["label"].map(CLASS_TO_ID)

    if df["target"].isna().any():
        raise ValueError("Unexpected target labels found.")

    df["target"] = df["target"].astype(int)

    return df


# ============================================================
# BASE MODEL
# ============================================================

def create_base_model():

    return Pipeline([
        ("scaler", StandardScaler()),

        ("model", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_multiclass_brier(y_true, probabilities):

    one_hot = np.eye(3)[y_true]

    return np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1
        )
    )


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):

    predicted = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    correctness = (predicted == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
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

        accuracy = correctness[mask].mean()

        avg_confidence = confidence[mask].mean()

        ece += (
            mask.mean() *
            abs(accuracy - avg_confidence)
        )

    return ece


def evaluate_model(name, y_true, probabilities):

    predictions = np.argmax(probabilities, axis=1)

    return {
        "model": name,

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),

        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1, 2]
        ),

        "multiclass_brier": calculate_multiclass_brier(
            y_true,
            probabilities
        ),

        "ece": expected_calibration_error(
            y_true,
            probabilities
        )
    }


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

def plot_reliability(y_true, probability_sets):

    plt.figure(figsize=(9, 7))

    for name, probabilities in probability_sets.items():

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        correctness = (
            predictions == y_true
        ).astype(float)

        bin_edges = np.linspace(0, 1, 11)

        avg_confidence = []
        avg_accuracy = []

        for i in range(10):

            mask = (
                (confidence >= bin_edges[i]) &
                (confidence < bin_edges[i + 1])
            )

            if mask.sum() == 0:
                continue

            avg_confidence.append(
                confidence[mask].mean()
            )

            avg_accuracy.append(
                correctness[mask].mean()
            )

        plt.plot(
            avg_confidence,
            avg_accuracy,
            marker="o",
            label=name
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean predicted confidence")

    plt.ylabel("Observed accuracy")

    plt.title("MOSAIC 2.0 - Probability Reliability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=150)

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nMOSAIC 2.0 - PROBABILITY CALIBRATION")
    print("=" * 70)

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

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    # Out-of-fold probabilities from the uncalibrated model.
    oof_probabilities = np.full(
        (len(train_df), 3),
        np.nan
    )

    print("\nGenerating chronological OOF predictions...")

    for fold, (train_day_idx, val_day_idx) in enumerate(
        splitter.split(unique_dates),
        start=1
    ):

        train_dates = set(unique_dates[train_day_idx])

        val_dates = set(unique_dates[val_day_idx])

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

        model = create_base_model()

        model.fit(
            X_train.iloc[train_indices],
            y_train[train_indices]
        )

        oof_probabilities[val_indices] = model.predict_proba(
            X_train.iloc[val_indices]
        )

        print(
            f"Fold {fold} completed: "
            f"{len(val_indices)} predictions"
        )

    valid_mask = ~np.isnan(
        oof_probabilities
    ).any(axis=1)

    y_oof = y_train[valid_mask]

    p_oof = oof_probabilities[valid_mask]

    # --------------------------------------------------------
    # FIT BASE MODEL ON FULL TRAINING DATA
    # --------------------------------------------------------

    print("\nTraining base model on full training set...")

    base_model = create_base_model()

    base_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # CALIBRATION EXPERIMENT
    # --------------------------------------------------------

    print("\nFitting calibration candidates...")

    sigmoid_model = CalibratedClassifierCV(
        estimator=create_base_model(),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3)
    )

    isotonic_model = CalibratedClassifierCV(
        estimator=create_base_model(),
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=3)
    )

    sigmoid_model.fit(
        X_train,
        y_train
    )

    isotonic_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    probability_sets = {

        "Uncalibrated": base_model.predict_proba(
            X_validation
        ),

        "Sigmoid": sigmoid_model.predict_proba(
            X_validation
        ),

        "Isotonic": isotonic_model.predict_proba(
            X_validation
        )
    }

    results = []

    for name, probabilities in probability_sets.items():

        metrics = evaluate_model(
            name,
            y_validation,
            probabilities
        )

        results.append(metrics)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SAVE RELIABILITY PLOT
    # --------------------------------------------------------

    plot_reliability(
        y_validation,
        probability_sets
    )

    print("\nVALIDATION CALIBRATION COMPARISON")

    print("=" * 85)

    print(
        report.round(5).to_string(index=False)
    )

    print("\nOOF BASELINE METRICS")

    print(
        pd.DataFrame([
            evaluate_model(
                "OOF Uncalibrated",
                y_oof,
                p_oof
            )
        ]).round(5).to_string(index=False)
    )

    print("\nReports saved:")

    print(REPORT_PATH)

    print(PLOT_PATH)

    print("\nIMPORTANT:")

    print("- Test data was not loaded.")

    print("- No saved models were overwritten.")

    print("- Calibration results are development results.")

    print("- Do not select a trading threshold from these results alone.")


if __name__ == "__main__":

    main()
