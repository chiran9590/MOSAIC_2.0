
import pandas as pd
import numpy as np

print("\nMOSAIC 2.0 - BACKTEST V2 INTEGRITY AUDIT")
print("=" * 70)

PATH = "reports/backtest_v2_trade_results.csv"

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading V2 trades...")

trades = pd.read_csv(
    PATH,
    parse_dates=[
        "signal_timestamp",
        "entry_timestamp",
        "exit_timestamp"
    ]
)

if trades.empty:
    raise ValueError("No trades found in the V2 trade results.")

print("Total trades:", len(trades))

required_columns = [
    "signal_timestamp",
    "direction",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "gross_return"
]

for col in required_columns:
    if col not in trades.columns:
        raise ValueError(f"Missing required column: {col}")

# --------------------------------------------------
# 2. DUPLICATE CHECK
# --------------------------------------------------

print("\n1. DUPLICATE CHECK")

duplicates = trades.duplicated(
    subset=["signal_timestamp", "entry_timestamp"]
).sum()

print("Duplicate trades:", duplicates)

# --------------------------------------------------
# 3. TRADE TIMING CHECK
# --------------------------------------------------

print("\n2. TRADE TIMING CHECK")

entry_delay = (
    trades["entry_timestamp"]
    - trades["signal_timestamp"]
).dt.total_seconds() / 60

holding_period = (
    trades["exit_timestamp"]
    - trades["entry_timestamp"]
).dt.total_seconds() / 60

print("\nEntry delay values:")
print(entry_delay.value_counts().to_string())

print("\nHolding period values:")
print(holding_period.value_counts().to_string())

incorrect_entry_delay = (
    entry_delay != 5
).sum()

incorrect_holding_period = (
    holding_period != 30
).sum()

# --------------------------------------------------
# 4. OVERLAP CHECK
# --------------------------------------------------

print("\n3. OVERLAP CHECK")

trades = trades.sort_values(
    ["entry_timestamp", "exit_timestamp"]
).reset_index(drop=True)

trades["previous_exit"] = (
    trades["exit_timestamp"].shift(1)
)

trades["overlap_minutes"] = (
    trades["previous_exit"]
    - trades["entry_timestamp"]
).dt.total_seconds() / 60

overlaps = (
    trades["overlap_minutes"] > 0
).sum()

same_time_transitions = (
    trades["overlap_minutes"] == 0
).sum()

print("Overlapping trades:", overlaps)

print(
    "Trades entering at previous exit:",
    same_time_transitions
)

# --------------------------------------------------
# 5. RETURN RECONSTRUCTION
# --------------------------------------------------

print("\n4. RETURN RECONSTRUCTION")

trades["reconstructed_return"] = np.where(
    trades["direction"] == "UP",
    trades["exit_price"] / trades["entry_price"] - 1,
    1 - trades["exit_price"] / trades["entry_price"]
)

trades["return_difference"] = (
    trades["gross_return"]
    - trades["reconstructed_return"]
)

maximum_return_difference = (
    trades["return_difference"].abs().max()
)

mean_return_difference = (
    trades["return_difference"].abs().mean()
)

print(
    "Maximum return difference:",
    f"{maximum_return_difference:.12f}"
)

print(
    "Mean return difference:",
    f"{mean_return_difference:.12f}"
)

# --------------------------------------------------
# 6. EQUITY CURVE AUDIT
# --------------------------------------------------

print("\n5. EQUITY CURVE AUDIT")

trades["reconstructed_equity"] = (
    1 + trades["gross_return"]
).cumprod()

trades["running_peak"] = (
    trades["reconstructed_equity"].cummax()
)

trades["reconstructed_drawdown"] = (
    trades["reconstructed_equity"]
    / trades["running_peak"]
) - 1

final_equity = (
    trades["reconstructed_equity"].iloc[-1]
)

total_return = final_equity - 1

maximum_drawdown = (
    trades["reconstructed_drawdown"].min()
)

print(
    "Final compounded gross return:",
    f"{total_return:.4%}"
)

print(
    "Maximum drawdown:",
    f"{maximum_drawdown:.4%}"
)

# --------------------------------------------------
# 7. DIRECTION-WISE RETURN CHECK
# --------------------------------------------------

print("\n6. DIRECTION-WISE RETURN CHECK")

direction_summary = trades.groupby(
    "direction"
).agg(
    trades=("gross_return", "count"),
    average_return=("gross_return", "mean"),
    win_rate=(
        "gross_return",
        lambda x: (x > 0).mean()
    ),
    total_return=(
        "gross_return",
        lambda x: (1 + x).prod() - 1
    )
)

print(direction_summary.to_string())

# --------------------------------------------------
# 8. DAILY PERFORMANCE CHECK
# --------------------------------------------------

print("\n7. DAILY PERFORMANCE CHECK")

trades["trading_date"] = (
    trades["signal_timestamp"].dt.date
)

daily = trades.groupby(
    "trading_date"
).agg(
    trades=("gross_return", "count"),
    daily_return=(
        "gross_return",
        lambda x: (1 + x).prod() - 1
    )
)

daily_compounded_return = (
    (1 + daily["daily_return"]).prod() - 1
)

print("Trading days:", len(daily))

print(
    "Total daily compounded return:",
    f"{daily_compounded_return:.4%}"
)

print(
    "Difference between trade and daily compounding:",
    f"{abs(total_return - daily_compounded_return):.12f}"
)

# --------------------------------------------------
# 9. SAVE AUDIT RESULTS
# --------------------------------------------------

trades.to_csv(
    "reports/backtest_v2_integrity_details.csv",
    index=False
)

daily.to_csv(
    "reports/backtest_v2_integrity_daily.csv"
)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("INTEGRITY AUDIT SUMMARY")
print("=" * 70)

print("Total trades:", len(trades))
print("Duplicate trades:", duplicates)
print("Overlapping trades:", overlaps)
print("Same-time transitions:", same_time_transitions)

print(
    "Incorrect entry delays:",
    incorrect_entry_delay
)

print(
    "Incorrect holding periods:",
    incorrect_holding_period
)

print(
    "Maximum return difference:",
    f"{maximum_return_difference:.12f}"
)

print(
    "Final gross return:",
    f"{total_return:.4%}"
)

print(
    "Maximum drawdown:",
    f"{maximum_drawdown:.4%}"
)

# --------------------------------------------------
# 11. VALIDATION STATUS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

checks = {
    "No duplicate trades": duplicates == 0,
    "No overlapping trades": overlaps == 0,
    "Correct entry delay": incorrect_entry_delay == 0,
    "Correct holding period": incorrect_holding_period == 0,
    "Return reconstruction": maximum_return_difference < 1e-10,
    "Daily compounding consistency": abs(
        total_return - daily_compounded_return
    ) < 1e-10
}

for name, passed in checks.items():
    print(
        f"{'[PASS]' if passed else '[FAIL]'} {name}"
    )

print("\nSaved:")
print("reports/backtest_v2_integrity_details.csv")
print("reports/backtest_v2_integrity_daily.csv")

if all(checks.values()):
    print("\nALL INTEGRITY CHECKS PASSED.")
else:
    print("\nSOME CHECKS FAILED. INVESTIGATION REQUIRED.")

print("\nAudit completed.")
