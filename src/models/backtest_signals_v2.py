
import os
import pandas as pd
import numpy as np

print("\nMOSAIC 2.0 - CORRECTED TRADING BACKTEST V2")
print("=" * 70)

PREDICTIONS_PATH = "reports/final_test_predictions.csv"
CANDLES_PATH = "data/processed/reliance_5min.csv"

OUTPUT_DIR = "reports"

# Keep zero for gross-performance comparison.
# We will add realistic transaction costs later.
ROUND_TRIP_COST_BPS = 0.0

HOLDING_MINUTES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["timestamp"]
)

candles = pd.read_csv(
    CANDLES_PATH,
    parse_dates=["timestamp"]
)

predictions = predictions.sort_values(
    "timestamp"
).reset_index(drop=True)

candles = candles.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Predictions:", len(predictions))
print("Candles:", len(candles))

# --------------------------------------------------
# 2. VALIDATE DATA
# --------------------------------------------------

required_prediction_columns = [
    "timestamp",
    "predicted_label"
]

required_candle_columns = [
    "timestamp",
    "open"
]

for col in required_prediction_columns:
    if col not in predictions.columns:
        raise ValueError(f"Missing prediction column: {col}")

for col in required_candle_columns:
    if col not in candles.columns:
        raise ValueError(f"Missing candle column: {col}")

if predictions["timestamp"].duplicated().any():
    raise ValueError("Duplicate prediction timestamps found.")

if candles["timestamp"].duplicated().any():
    raise ValueError("Duplicate candle timestamps found.")

open_prices = candles.set_index(
    "timestamp"
)["open"].to_dict()

candle_timestamps = set(candles["timestamp"])

# --------------------------------------------------
# 3. GENERATE TRADE CANDIDATES
# --------------------------------------------------

print("\nPreparing trade candidates...")

candidates = predictions[
    predictions["predicted_label"].isin(["UP", "DOWN"])
].copy()

candidates["entry_timestamp"] = (
    candidates["timestamp"]
    + pd.Timedelta(minutes=5)
)

candidates["exit_timestamp"] = (
    candidates["entry_timestamp"]
    + pd.Timedelta(minutes=HOLDING_MINUTES)
)

candidates["entry_price"] = (
    candidates["entry_timestamp"].map(open_prices)
)

candidates["exit_price"] = (
    candidates["exit_timestamp"].map(open_prices)
)

candidates["same_trading_day"] = (
    candidates["timestamp"].dt.date
    == candidates["exit_timestamp"].dt.date
)

# --------------------------------------------------
# 4. CHECK PRICE AVAILABILITY
# --------------------------------------------------

missing_entry = 0
missing_exit = 0
wrong_day = 0

valid_candidates = []

for _, row in candidates.iterrows():

    if row["entry_timestamp"] not in candle_timestamps:
        missing_entry += 1
        continue

    if row["exit_timestamp"] not in candle_timestamps:
        missing_exit += 1
        continue

    if not row["same_trading_day"]:
        wrong_day += 1
        continue

    if pd.isna(row["entry_price"]) or pd.isna(row["exit_price"]):
        missing_entry += 1
        continue

    if row["entry_price"] <= 0 or row["exit_price"] <= 0:
        continue

    valid_candidates.append(row)

print("\nPRICE VALIDATION")
print("-" * 50)

print("Directional signals:", len(candidates))
print("Missing entry:", missing_entry)
print("Missing exit:", missing_exit)
print("Wrong trading day:", wrong_day)
print("Valid candidates:", len(valid_candidates))

# --------------------------------------------------
# 5. EXECUTE NON-OVERLAPPING TRADES
# --------------------------------------------------

print("\nExecuting trades...")

valid_candidates = sorted(
    valid_candidates,
    key=lambda x: x["entry_timestamp"]
)

executed_trades = []

next_available_entry = None
skipped_overlapping = 0

for row in valid_candidates:

    entry_time = row["entry_timestamp"]
    exit_time = row["exit_timestamp"]

    # IMPORTANT:
    # Equality is allowed. A new trade can enter at
    # the exact timestamp when the previous trade exits.
    if (
        next_available_entry is not None
        and entry_time < next_available_entry
    ):
        skipped_overlapping += 1
        continue

    entry_price = float(row["entry_price"])
    exit_price = float(row["exit_price"])

    direction = row["predicted_label"]

    market_return = (
        exit_price / entry_price
    ) - 1

    if direction == "UP":
        gross_return = market_return
    else:
        gross_return = -market_return

    cost = ROUND_TRIP_COST_BPS / 10000

    net_return = (
        (1 + gross_return) * (1 - cost)
    ) - 1

    executed_trades.append({
        "signal_timestamp": row["timestamp"],
        "direction": direction,
        "entry_timestamp": entry_time,
        "exit_timestamp": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "market_return": market_return,
        "gross_return": gross_return,
        "net_return": net_return
    })

    next_available_entry = exit_time

trades = pd.DataFrame(executed_trades)

if trades.empty:
    raise ValueError("No trades were executed.")

# --------------------------------------------------
# 6. CALCULATE TRADE METRICS
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

trades["equity"] = (
    1 + trades["net_return"]
).cumprod()

trades["running_peak"] = (
    trades["equity"].cummax()
)

trades["drawdown"] = (
    trades["equity"] / trades["running_peak"]
) - 1

total_trades = len(trades)

winning_trades = (
    trades["net_return"] > 0
).sum()

losing_trades = (
    trades["net_return"] < 0
).sum()

breakeven_trades = (
    trades["net_return"] == 0
).sum()

win_rate = winning_trades / total_trades

average_return = trades["net_return"].mean()

median_return = trades["net_return"].median()

gross_total_return = (
    (1 + trades["gross_return"]).prod() - 1
)

net_total_return = trades["equity"].iloc[-1] - 1

gross_profit = trades.loc[
    trades["net_return"] > 0,
    "net_return"
].sum()

gross_loss = abs(
    trades.loc[
        trades["net_return"] < 0,
        "net_return"
    ].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0 else np.inf
)

maximum_drawdown = trades["drawdown"].min()

print("Total trades:", total_trades)
print("Winning trades:", winning_trades)
print("Losing trades:", losing_trades)
print("Breakeven trades:", breakeven_trades)

print(f"Win rate: {win_rate:.2%}")
print(f"Average net return: {average_return:.4%}")
print(f"Median net return: {median_return:.4%}")
print(f"Gross compounded return: {gross_total_return:.4%}")
print(f"Net compounded return: {net_total_return:.4%}")
print(f"Profit factor: {profit_factor:.4f}")
print(f"Maximum drawdown: {maximum_drawdown:.4%}")

# --------------------------------------------------
# 7. DIRECTION-WISE PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 70)
print("DIRECTION-WISE PERFORMANCE")
print("=" * 70)

direction_summary = trades.groupby(
    "direction"
).agg(
    total_trades=("net_return", "count"),
    average_gross_return=("gross_return", "mean"),
    average_net_return=("net_return", "mean"),
    median_return=("net_return", "median"),
    total_gross_return=(
        "gross_return",
        lambda x: (1 + x).prod() - 1
    ),
    win_rate=(
        "net_return",
        lambda x: (x > 0).mean()
    )
)

print(direction_summary.to_string())

# --------------------------------------------------
# 8. DAILY PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 70)
print("DAILY PERFORMANCE")
print("=" * 70)

trades["trading_date"] = (
    trades["signal_timestamp"].dt.date
)

daily = trades.groupby(
    "trading_date"
).agg(
    trades=("net_return", "count"),
    daily_return=(
        "net_return",
        lambda x: (1 + x).prod() - 1
    )
)

daily["daily_equity"] = (
    1 + daily["daily_return"]
).cumprod()

daily["daily_peak"] = (
    daily["daily_equity"].cummax()
)

daily["drawdown"] = (
    daily["daily_equity"] / daily["daily_peak"]
) - 1

print(daily["daily_return"].describe())

print("\nTrading days:", len(daily))
print("Average trades/day:", daily["trades"].mean())

# --------------------------------------------------
# 9. SAVE REPORTS
# --------------------------------------------------

trades.to_csv(
    f"{OUTPUT_DIR}/backtest_v2_trade_results.csv",
    index=False
)

direction_summary.to_csv(
    f"{OUTPUT_DIR}/backtest_v2_direction_summary.csv"
)

daily.to_csv(
    f"{OUTPUT_DIR}/backtest_v2_daily_performance.csv"
)

overall = pd.DataFrame([{
    "total_trades": total_trades,
    "winning_trades": winning_trades,
    "losing_trades": losing_trades,
    "breakeven_trades": breakeven_trades,
    "win_rate": win_rate,
    "average_net_return": average_return,
    "median_net_return": median_return,
    "gross_total_return": gross_total_return,
    "net_total_return": net_total_return,
    "profit_factor": profit_factor,
    "maximum_drawdown": maximum_drawdown,
    "skipped_overlapping": skipped_overlapping,
    "missing_entry": missing_entry,
    "missing_exit": missing_exit,
    "wrong_trading_day": wrong_day
}])

overall.to_csv(
    f"{OUTPUT_DIR}/backtest_v2_overall_metrics.csv",
    index=False
)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("BACKTEST V2 COMPLETED")
print("=" * 70)

print("Executed trades:", total_trades)
print("Skipped overlapping:", skipped_overlapping)
print("Missing entry:", missing_entry)
print("Missing exit:", missing_exit)

print("\nSaved reports:")
print("reports/backtest_v2_trade_results.csv")
print("reports/backtest_v2_direction_summary.csv")
print("reports/backtest_v2_daily_performance.csv")
print("reports/backtest_v2_overall_metrics.csv")

print("\nBacktest V2 completed successfully.")
