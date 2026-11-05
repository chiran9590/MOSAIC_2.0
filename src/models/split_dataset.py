
import pandas as pd
from pathlib import Path

# Input file
INPUT_FILE = Path("data/processed/reliance_features.csv")

# Output directory
OUTPUT_DIR = Path("data/processed/splits")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def show_summary(name, df):

    print(f"\n{name} DATASET")
    print("-" * 55)

    print("Records:", len(df))

    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )

    print("\nClass counts:")
    print(
        df["label"].value_counts()
        .reindex(["UP", "DOWN", "SIDEWAYS"], fill_value=0)
    )

    print("\nClass percentages:")
    print(
        (
            df["label"].value_counts(normalize=True) * 100
        )
        .reindex(["UP", "DOWN", "SIDEWAYS"], fill_value=0)
        .round(2)
    )


def main():

    print("\nMOSAIC 2.0 - CHRONOLOGICAL DATA SPLITTING")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    if abs(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
    ) > 1e-9:
        raise ValueError("Split ratios must add up to 1.")

    # Load feature dataset
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print("Total records:", len(df))

    # Extract unique trading dates in chronological order
    trading_dates = sorted(
        df["timestamp"].dt.date.unique()
    )

    total_days = len(trading_dates)

    # Calculate split boundaries using trading days
    train_end = int(total_days * TRAIN_RATIO)

    validation_end = int(
        total_days * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_dates = trading_dates[:train_end]

    validation_dates = trading_dates[
        train_end:validation_end
    ]

    test_dates = trading_dates[validation_end:]

    # Create datasets
    train_df = df[
        df["timestamp"].dt.date.isin(train_dates)
    ].copy()

    validation_df = df[
        df["timestamp"].dt.date.isin(validation_dates)
    ].copy()

    test_df = df[
        df["timestamp"].dt.date.isin(test_dates)
    ].copy()

    # Save datasets
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # Display summaries
    print("\nSPLITTING SUMMARY")
    print("=" * 60)

    print("Total trading days:", total_days)

    print(
        f"Training days: {len(train_dates)} "
        f"({TRAIN_RATIO * 100:.0f}%)"
    )

    print(
        f"Validation days: {len(validation_dates)} "
        f"({VALIDATION_RATIO * 100:.0f}%)"
    )

    print(
        f"Testing days: {len(test_dates)} "
        f"({TEST_RATIO * 100:.0f}%)"
    )

    show_summary("TRAINING", train_df)
    show_summary("VALIDATION", validation_df)
    show_summary("TESTING", test_df)

    # Verify date separation
    assert max(train_dates) < min(validation_dates)
    assert max(validation_dates) < min(test_dates)

    assert (
        set(train_dates).isdisjoint(validation_dates)
    )

    assert (
        set(train_dates).isdisjoint(test_dates)
    )

    assert (
        set(validation_dates).isdisjoint(test_dates)
    )

    print("\nDATASET VERIFICATION")
    print("-" * 55)
    print("Training and validation dates overlap: NO")
    print("Training and testing dates overlap: NO")
    print("Validation and testing dates overlap: NO")

    print("\nSaved files:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")

    print("\nCHRONOLOGICAL SPLITTING COMPLETED")


if __name__ == "__main__":
    main()
