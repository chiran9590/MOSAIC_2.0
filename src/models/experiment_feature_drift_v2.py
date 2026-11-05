
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, wasserstein_distance

# ============================================================
# MOSAIC 2.0 - CORRECTED FEATURE DRIFT AUDIT
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

NUMBER_OF_BINS = 10
PSEUDOCOUNT = 0.5


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot identify {description} column. "
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
        "date"
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
        raise ValueError("Invalid target labels.")

    df[target_col] = df[target_col].astype(int)

    if not set(df[target_col].unique()).issubset({0, 1, 2}):
        raise ValueError("Unexpected class encoding.")

    df = df.sort_values(date_col).reset_index(drop=True)

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing feature values found.")

    df["_trading_date"] = df[date_col].dt.date

    return df, target_col


def calculate_smoothed_psi(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Reference quantiles define fixed comparison bins.
    quantiles = np.linspace(
        0,
        1,
        NUMBER_OF_BINS + 1
    )

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges
    )

    # Apply additive smoothing to prevent zero proportions.
    reference_pct = (
        reference_counts + PSEUDOCOUNT
    ) / (
        len(reference_counts) * PSEUDOCOUNT +
        len(reference)
    )

    current_pct = (
        current_counts + PSEUDOCOUNT
    ) / (
        len(current_counts) * PSEUDOCOUNT +
        len(current)
    )

    psi = np.sum(
        (current_pct - reference_pct) *
        np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_feature_metrics(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference_std = np.std(reference)

    if reference_std < 1e-12:
        standardized_shift = 0.0
        normalized_wasserstein = 0.0
    else:
        standardized_shift = (
            np.mean(current) - np.mean(reference)
        ) / reference_std

        normalized_wasserstein = (
            wasserstein_distance(reference, current)
            / reference_std
        )

    ks_result = ks_2samp(
        reference,
        current
    )

    return {
        "psi": calculate_smoothed_psi(reference, current),
        "ks_statistic": ks_result.statistic,
        "ks_pvalue": ks_result.pvalue,
        "standardized_mean_shift": standardized_shift,
        "normalized_wasserstein": normalized_wasserstein,
        "training_median": np.median(reference),
        "validation_median": np.median(current),
        "training_std": reference_std,
        "validation_std": np.std(current)
    }


def calculate_class_distribution(df, target_col):
    result = {}

    for class_id, class_name in CLASS_NAMES.items():
        result[f"{class_name}_pct"] = (
            (df[target_col] == class_id).mean() * 100
        )

    return result


def main():
    print("\nMOSAIC 2.0 - CORRECTED FEATURE DRIFT AUDIT")
    print("=" * 95)

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

    date_blocks = np.array_split(
        validation_dates,
        5
    )

    all_results = []
    summary_results = []

    for index, block_dates in enumerate(
        date_blocks,
        start=1
    ):
        block_name = f"Block_{index}"

        block_df = validation_df[
            validation_df["_trading_date"].isin(block_dates)
        ]

        print("\n" + "=" * 95)
        print(
            f"{block_name}: {block_dates[0]} to {block_dates[-1]}"
        )
        print("=" * 95)

        block_results = []

        for feature in FEATURES:
            metrics = calculate_feature_metrics(
                train_df[feature].to_numpy(),
                block_df[feature].to_numpy()
            )

            metrics["feature"] = feature
            metrics["block"] = block_name

            block_results.append(metrics)

        block_results_df = pd.DataFrame(block_results)

        all_results.append(block_results_df)

        print("\nTOP 10 FEATURES BY SMOOTHED PSI")

        print(
            block_results_df.sort_values(
                "psi",
                ascending=False
            )[
                [
                    "feature",
                    "psi",
                    "ks_statistic",
                    "standardized_mean_shift",
                    "normalized_wasserstein"
                ]
            ].head(10).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        high_psi = int(
            (block_results_df["psi"] >= 0.25).sum()
        )

        moderate_psi = int(
            (
                (block_results_df["psi"] >= 0.10) &
                (block_results_df["psi"] < 0.25)
            ).sum()
        )

        low_psi = int(
            (block_results_df["psi"] < 0.10).sum()
        )

        distribution = calculate_class_distribution(
            block_df,
            validation_target
        )

        summary_results.append({
            "block": block_name,
            "records": len(block_df),
            "low_psi_features": low_psi,
            "moderate_psi_features": moderate_psi,
            "high_psi_features": high_psi,
            "mean_psi": block_results_df["psi"].mean(),
            "max_psi": block_results_df["psi"].max(),
            "mean_ks": block_results_df["ks_statistic"].mean(),
            **distribution
        })

        print("\nCLASS DISTRIBUTION")

        for class_name in CLASS_NAMES.values():
            print(
                f"{class_name:10s}: "
                f"{distribution[f'{class_name}_pct']:.2f}%"
            )

        print("\nDRIFT COUNTS USING SMOOTHED PSI")
        print(f"Low:      {low_psi}")
        print(f"Moderate: {moderate_psi}")
        print(f"High:     {high_psi}")

    details_df = pd.concat(
        all_results,
        ignore_index=True
    )

    summary_df = pd.DataFrame(summary_results)

    details_path = REPORT_DIR / "feature_drift_v2_details.csv"
    summary_path = REPORT_DIR / "feature_drift_v2_summary.csv"

    details_df.to_csv(
        details_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 95)
    print("FINAL DRIFT SUMMARY")
    print("=" * 95)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFEATURES WITH HIGHEST AVERAGE PSI")

    average_psi = (
        details_df.groupby("feature")["psi"]
        .mean()
        .sort_values(ascending=False)
    )

    print(average_psi.head(10).to_string())

    print("\nREPORTS SAVED")
    print(details_path)
    print(summary_path)

    print("\nIMPORTANT")
    print("- PSI uses additive smoothing to reduce empty-bin inflation.")
    print("- KS and Wasserstein provide additional distribution comparisons.")
    print("- KS p-values are descriptive and sensitive to sample size.")
    print("- Drift does not establish that a feature causes prediction errors.")
    print("- No test data was loaded.")
    print("- No model artifacts were modified.")


if __name__ == "__main__":
    main()
