
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# MOSAIC 2.0 - FEATURE AND MARKET DISTRIBUTION DRIFT AUDIT
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

EPSILON = 1e-6
NUMBER_OF_BINS = 10


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description} column. "
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
        raise ValueError("Invalid timestamps found.")

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
        raise ValueError("Invalid target labels found.")

    df[target_col] = df[target_col].astype(int)

    if not set(df[target_col].unique()).issubset({0, 1, 2}):
        raise ValueError("Unexpected class encoding.")

    df = df.sort_values(date_col).reset_index(drop=True)

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Missing or infinite feature values found.")

    df["_trading_date"] = df[date_col].dt.date

    return df, target_col


def calculate_psi(reference, current):
    """
    Population Stability Index using quantile boundaries
    learned exclusively from the reference/training data.
    """

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(
        0,
        1,
        NUMBER_OF_BINS + 1
    )

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    # Constant feature: no meaningful distribution bins.
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference,
        bins=edges,
        include_lowest=True
    )

    current_bins = pd.cut(
        current,
        bins=edges,
        include_lowest=True
    )

    reference_counts = pd.Series(
        reference_bins
    ).value_counts(sort=False)

    current_counts = pd.Series(
        current_bins
    ).value_counts(sort=False)

    reference_pct = (
        reference_counts.to_numpy() / len(reference)
    )

    current_pct = (
        current_counts.to_numpy() / len(current)
    )

    reference_pct = np.clip(
        reference_pct,
        EPSILON,
        None
    )

    current_pct = np.clip(
        current_pct,
        EPSILON,
        None
    )

    psi = np.sum(
        (current_pct - reference_pct) *
        np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_feature_drift(reference_df, current_df):
    results = []

    for feature in FEATURES:
        reference = reference_df[feature].to_numpy(dtype=float)
        current = current_df[feature].to_numpy(dtype=float)

        reference_mean = np.mean(reference)
        current_mean = np.mean(current)

        reference_std = np.std(reference)

        if reference_std < 1e-12:
            standardized_shift = 0.0
        else:
            standardized_shift = (
                current_mean - reference_mean
            ) / reference_std

        psi = calculate_psi(reference, current)

        results.append({
            "feature": feature,
            "training_mean": reference_mean,
            "validation_mean": current_mean,
            "training_std": reference_std,
            "standardized_mean_shift": standardized_shift,
            "absolute_mean_shift": abs(standardized_shift),
            "psi": psi
        })

    return pd.DataFrame(results).sort_values(
        "psi",
        ascending=False
    )


def classify_psi(psi):
    if psi < 0.10:
        return "LOW"

    if psi < 0.25:
        return "MODERATE"

    return "HIGH"


def calculate_class_distribution(df, target_col):
    counts = df[target_col].value_counts().reindex(
        [0, 1, 2],
        fill_value=0
    )

    percentages = counts / len(df) * 100

    return {
        f"{CLASS_NAMES[class_id]}_count": int(counts[class_id])
        for class_id in [0, 1, 2]
    } | {
        f"{CLASS_NAMES[class_id]}_pct": percentages[class_id]
        for class_id in [0, 1, 2]
    }


def main():
    print("\nMOSAIC 2.0 - FEATURE DRIFT AUDIT")
    print("=" * 90)

    print("\nLoading datasets...")

    train_df, train_target = load_dataset(TRAIN_PATH)
    validation_df, validation_target = load_dataset(
        VALIDATION_PATH
    )

    train_dates = np.sort(
        train_df["_trading_date"].unique()
    )

    validation_dates = np.sort(
        validation_df["_trading_date"].unique()
    )

    print(f"Training records:   {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")

    print(
        f"Training period: {train_dates[0]} to {train_dates[-1]}"
    )

    print(
        f"Validation period: "
        f"{validation_dates[0]} to {validation_dates[-1]}"
    )

    # Split validation dates into five chronological blocks.
    date_blocks = np.array_split(
        validation_dates,
        5
    )

    all_drift_results = []
    block_summary = []

    for index, block_dates in enumerate(
        date_blocks,
        start=1
    ):
        block_df = validation_df[
            validation_df["_trading_date"].isin(block_dates)
        ].copy()

        block_name = f"Block_{index}"

        print("\n" + "=" * 90)
        print(
            f"{block_name}: {block_dates[0]} to {block_dates[-1]}"
        )
        print("=" * 90)

        drift_df = calculate_feature_drift(
            train_df,
            block_df
        )

        drift_df["block"] = block_name
        drift_df["drift_level"] = drift_df["psi"].apply(
            classify_psi
        )

        all_drift_results.append(drift_df)

        print("\nTOP 10 FEATURES BY PSI")

        display_columns = [
            "feature",
            "psi",
            "drift_level",
            "standardized_mean_shift"
        ]

        print(
            drift_df[display_columns]
            .head(10)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        low_count = int(
            (drift_df["drift_level"] == "LOW").sum()
        )

        moderate_count = int(
            (drift_df["drift_level"] == "MODERATE").sum()
        )

        high_count = int(
            (drift_df["drift_level"] == "HIGH").sum()
        )

        distribution = calculate_class_distribution(
            block_df,
            validation_target
        )

        summary = {
            "block": block_name,
            "start_date": str(block_dates[0]),
            "end_date": str(block_dates[-1]),
            "records": len(block_df),
            "low_drift_features": low_count,
            "moderate_drift_features": moderate_count,
            "high_drift_features": high_count,
            "mean_psi": drift_df["psi"].mean(),
            "max_psi": drift_df["psi"].max(),
            **distribution
        }

        block_summary.append(summary)

        print("\nCLASS DISTRIBUTION")

        for class_id, class_name in CLASS_NAMES.items():
            print(
                f"{class_name:10s}: "
                f"{distribution[f'{class_name}_pct']:.2f}%"
            )

        print("\nDRIFT SUMMARY")
        print(f"Low drift features:      {low_count}")
        print(f"Moderate drift features: {moderate_count}")
        print(f"High drift features:     {high_count}")
        print(f"Mean PSI:                {drift_df['psi'].mean():.4f}")
        print(f"Maximum PSI:             {drift_df['psi'].max():.4f}")

    all_drift_df = pd.concat(
        all_drift_results,
        ignore_index=True
    )

    summary_df = pd.DataFrame(block_summary)

    drift_path = REPORT_DIR / "feature_drift_details.csv"
    summary_path = REPORT_DIR / "feature_drift_summary.csv"

    all_drift_df.to_csv(
        drift_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # Identify features repeatedly showing high drift.
    repeated_high_drift = (
        all_drift_df[
            all_drift_df["drift_level"] == "HIGH"
        ]
        .groupby("feature")
        .size()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 90)
    print("FEATURES WITH REPEATED HIGH DRIFT")
    print("=" * 90)

    if repeated_high_drift.empty:
        print("No features crossed the high-drift threshold.")
    else:
        print(repeated_high_drift.to_string())

    print("\nFINAL BLOCK SUMMARY")
    print("=" * 90)

    print(
        summary_df[
            [
                "block",
                "records",
                "low_drift_features",
                "moderate_drift_features",
                "high_drift_features",
                "mean_psi",
                "max_psi"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nREPORTS SAVED")
    print(drift_path)
    print(summary_path)

    print("\nIMPORTANT")
    print("- Training data is the reference distribution.")
    print("- Validation blocks are compared chronologically.")
    print("- No test data was loaded.")
    print("- PSI thresholds are screening heuristics, not proof of model failure.")
    print("- No model or feature artifacts were modified.")


if __name__ == "__main__":
    main()
