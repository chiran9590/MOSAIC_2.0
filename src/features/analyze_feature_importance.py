
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# ============================================================
# MOSAIC 2.0 - FEATURE IMPORTANCE AND REDUNDANCY ANALYSIS
# ============================================================

print("\nMOSAIC 2.0 - FEATURE ANALYSIS")
print("=" * 65)

DATA_PATH = "data/processed/advanced_splits/train.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. LOAD TRAINING DATA ONLY
# ------------------------------------------------------------

print("\nLoading training data...")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

print("Training records:", len(df))

# ------------------------------------------------------------
# 2. DEFINE FEATURES
# ------------------------------------------------------------

features = [
    "return_1", "return_3", "return_6", "return_12",
    "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
    "ATR_14", "volume_change", "volume_ratio",
    "high_low_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "hour", "minute", "time_of_day",
    "bb_position", "bb_width_pct", "close_vwap_pct",
    "adx_14", "plus_di_14", "minus_di_14", "di_spread",
    "stochastic_k", "stochastic_d",
    "obv_change", "obv_ratio"
]

X = df[features].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

if X.isna().any().any():
    raise ValueError("Missing values detected in feature data.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Infinite values detected in feature data.")

print("Features:", len(features))
print("Classes:", list(encoder.classes_))

# ------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n1. CORRELATION ANALYSIS")
print("-" * 65)

correlation = X.corr(method="spearman").abs()

correlation_pairs = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        value = correlation.iloc[i, j]

        if value >= 0.90:
            correlation_pairs.append({
                "feature_1": features[i],
                "feature_2": features[j],
                "absolute_spearman_correlation": value
            })

correlation_df = pd.DataFrame(
    correlation_pairs,
    columns=[
        "feature_1",
        "feature_2",
        "absolute_spearman_correlation"
    ]
)

correlation_df = correlation_df.sort_values(
    by="absolute_spearman_correlation",
    ascending=False
)

correlation_path = f"{REPORT_DIR}/feature_correlations.csv"
correlation_df.to_csv(correlation_path, index=False)

if len(correlation_df) > 0:
    print("\nHighly correlated feature pairs (>= 0.90):")
    print(
        correlation_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
else:
    print("No feature pairs with correlation >= 0.90.")

# ------------------------------------------------------------
# 4. MUTUAL INFORMATION
# ------------------------------------------------------------

print("\n2. MUTUAL INFORMATION ANALYSIS")
print("-" * 65)

mi_values = mutual_info_classif(
    X,
    y,
    discrete_features=False,
    random_state=RANDOM_STATE
)

mi_df = pd.DataFrame({
    "feature": features,
    "mutual_information": mi_values
})

mi_df = mi_df.sort_values(
    by="mutual_information",
    ascending=False
)

print("\nTop 15 features by mutual information:")
print(
    mi_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# ------------------------------------------------------------
# 5. LOGISTIC REGRESSION COEFFICIENT IMPORTANCE
# ------------------------------------------------------------

print("\n3. LOGISTIC REGRESSION IMPORTANCE")
print("-" * 65)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    ))
])

lr_pipeline.fit(X, y)

lr_coefficients = np.abs(
    lr_pipeline.named_steps["model"].coef_
).mean(axis=0)

lr_df = pd.DataFrame({
    "feature": features,
    "lr_mean_absolute_coefficient": lr_coefficients
})

lr_df = lr_df.sort_values(
    by="lr_mean_absolute_coefficient",
    ascending=False
)

print("\nTop 15 features by Logistic Regression:")
print(
    lr_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# ------------------------------------------------------------
# 6. RANDOM FOREST IMPORTANCE
# ------------------------------------------------------------

print("\n4. RANDOM FOREST IMPORTANCE")
print("-" * 65)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X, y)

rf_df = pd.DataFrame({
    "feature": features,
    "rf_importance": rf.feature_importances_
})

rf_df = rf_df.sort_values(
    by="rf_importance",
    ascending=False
)

print("\nTop 15 features by Random Forest:")
print(
    rf_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# ------------------------------------------------------------
# 7. COMBINE FEATURE RANKINGS
# ------------------------------------------------------------

print("\n5. COMBINED FEATURE RANKING")
print("-" * 65)

combined = (
    mi_df
    .merge(lr_df, on="feature")
    .merge(rf_df, on="feature")
)

combined["mi_rank"] = combined[
    "mutual_information"
].rank(ascending=False, method="min")

combined["lr_rank"] = combined[
    "lr_mean_absolute_coefficient"
].rank(ascending=False, method="min")

combined["rf_rank"] = combined[
    "rf_importance"
].rank(ascending=False, method="min")

combined["average_rank"] = combined[
    ["mi_rank", "lr_rank", "rf_rank"]
].mean(axis=1)

combined = combined.sort_values("average_rank")

combined["final_rank"] = np.arange(1, len(combined) + 1)

combined_path = f"{REPORT_DIR}/feature_importance_analysis.csv"

combined.to_csv(combined_path, index=False)

print("\nFeature ranking:")
print(
    combined[
        [
            "final_rank",
            "feature",
            "mutual_information",
            "lr_mean_absolute_coefficient",
            "rf_importance",
            "average_rank"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# ------------------------------------------------------------
# 8. SAVE FEATURE CORRELATION MATRIX
# ------------------------------------------------------------

correlation.to_csv(
    f"{REPORT_DIR}/feature_correlation_matrix.csv"
)

# ------------------------------------------------------------
# 9. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated reports:")
print("1. reports/feature_correlations.csv")
print("2. reports/feature_correlation_matrix.csv")
print("3. reports/feature_importance_analysis.csv")

print("\nImportant:")
print("- Analysis used training data only.")
print("- Validation data was not used.")
print("- Test data was not used.")
print("- No features have been removed yet.")

print("\nNext step: Select candidate feature combinations.")
