
import pandas as pd
import numpy as np
import joblib
import importlib.util

from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models/saved")
REPORT_DIR = Path("reports")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "SMA_5",
    "SMA_10",
    "EMA_5",
    "EMA_10",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "ATR_14",
    "volume_change",
    "volume_ratio",
    "high_low_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "hour",
    "minute",
    "time_of_day"
]

TARGET = "label"

CLASS_ORDER = ["UP", "DOWN", "SIDEWAYS"]


def load_data():

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    # Check required columns
    required_columns = FEATURES + [TARGET]

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"Missing training column: {column}"
            )

        if column not in validation_df.columns:
            raise ValueError(
                f"Missing validation column: {column}"
            )

    X_train = train_df[FEATURES].copy()
    X_validation = validation_df[FEATURES].copy()

    # Replace infinite values
    X_train = X_train.replace(
        [np.inf, -np.inf], np.nan
    )

    X_validation = X_validation.replace(
        [np.inf, -np.inf], np.nan
    )

    if X_train.isnull().any().any():
        raise ValueError("Training features contain missing values.")

    if X_validation.isnull().any().any():
        raise ValueError("Validation features contain missing values.")

    # Encode target classes
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        train_df[TARGET]
    )

    y_validation = encoder.transform(
        validation_df[TARGET]
    )

    print("Training records:", len(X_train))
    print("Validation records:", len(X_validation))
    print("Number of features:", len(FEATURES))

    print("\nEncoded classes:")
    for index, class_name in enumerate(encoder.classes_):
        print(index, "=", class_name)

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        encoder
    )


def evaluate_model(
    name,
    model,
    X_validation,
    y_validation,
    encoder
):

    print("\n" + "=" * 65)
    print("MODEL:", name)
    print("=" * 65)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation, predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
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

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_validation,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=np.arange(len(encoder.classes_))
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def main():

    print("\nMOSAIC 2.0 - MODEL TRAINING")
    print("=" * 65)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        encoder
    ) = load_data()

    # Balanced sample weights for training
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    results = []
    trained_models = {}

    # --------------------------------------------------
    # 1. MAJORITY CLASS BASELINE
    # --------------------------------------------------

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    results.append(
        evaluate_model(
            "Majority Baseline",
            baseline,
            X_validation,
            y_validation,
            encoder
        )
    )

    # --------------------------------------------------
    # 2. LOGISTIC REGRESSION
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    logistic_model.fit(
        X_train,
        y_train
    )

    trained_models["Logistic Regression"] = logistic_model

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_validation,
            y_validation,
            encoder
        )
    )

    # --------------------------------------------------
    # 3. RANDOM FOREST
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    trained_models["Random Forest"] = rf_model

    results.append(
        evaluate_model(
            "Random Forest",
            rf_model,
            X_validation,
            y_validation,
            encoder
        )
    )

    # --------------------------------------------------
    # 4. XGBOOST
    # --------------------------------------------------

    if importlib.util.find_spec("xgboost"):

        from xgboost import XGBClassifier

        print("\nTraining XGBoost...")

        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

        trained_models["XGBoost"] = xgb_model

        results.append(
            evaluate_model(
                "XGBoost",
                xgb_model,
                X_validation,
                y_validation,
                encoder
            )
        )

    else:

        print("\nXGBoost not installed. Skipping.")

    # --------------------------------------------------
    # 5. LIGHTGBM
    # --------------------------------------------------

    if importlib.util.find_spec("lightgbm"):

        from lightgbm import LGBMClassifier

        print("\nTraining LightGBM...")

        lgbm_model = LGBMClassifier(
            objective="multiclass",
            num_class=len(encoder.classes_),
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        lgbm_model.fit(
            X_train,
            y_train
        )

        trained_models["LightGBM"] = lgbm_model

        results.append(
            evaluate_model(
                "LightGBM",
                lgbm_model,
                X_validation,
                y_validation,
                encoder
            )
        )

    else:

        print("\nLightGBM not installed. Skipping.")

    # --------------------------------------------------
    # 6. MODEL COMPARISON
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="macro_f1",
        ascending=False
    )

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        comparison.to_string(index=False)
    )

    # --------------------------------------------------
    # 7. SAVE BEST MODEL
    # --------------------------------------------------

    best_model_name = comparison.iloc[0]["model"]

    if best_model_name == "Majority Baseline":

        print(
            "\nWARNING: No trained model outperformed "
            "the majority baseline on validation macro F1."
        )

    else:

        best_model = trained_models[best_model_name]

        joblib.dump(
            best_model,
            MODEL_DIR / "best_model.joblib"
        )

        joblib.dump(
            encoder,
            MODEL_DIR / "label_encoder.joblib"
        )

        print("\nBest model:", best_model_name)

        print(
            "Saved model:",
            MODEL_DIR / "best_model.joblib"
        )

        print(
            "Saved encoder:",
            MODEL_DIR / "label_encoder.joblib"
        )

    print(
        "\nComparison report:",
        REPORT_DIR / "model_comparison.csv"
    )

    print("\nMODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
