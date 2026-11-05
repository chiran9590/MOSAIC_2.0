
import os
import pandas as pd
import numpy as np

# ============================================================
# MOSAIC 2.0 - EXECUTION-AWARE BACKTEST
# ============================================================

print("\nMOSAIC 2.0 - TRADING SIGNAL BACKTEST")
print("=" * 70)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PREDICTIONS_PATH = "reports/final_test_predictions.csv"
CANDLES_PATH = "data/processed/reliance_5min.csv"

REPORT_DIR = "reports"

# Illustrative configurable round-trip trading cost.
# Keep at zero initially to measure gross performance.
ROUND_TRIP_COST_BPS = 0.0

# 1 basis point = 0.01%
# 5 bps = 0.05%

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading predictions and 5-minute candles...")

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["timestamp"]
)

candles = pd.read_csv(
    CANDLES_PATH,
    parse_dates=["timestamp"]
)

print("Prediction records:", len(predictions))
print("5-minute candles:", len(candles))

# ============================================================
# 3. VALIDATE CANDLE DATA
# ============================================================

required_candle_columns = [
    "timestamp",
    "open",
    "close"
]

missing = [
    col for col in required_candle_columns
    if col not in candles.columns
]

if missing:
    raise ValueError(
        f"Missing candle columns: {missing}"
    )

if candles["timestamp"].duplicated().any():
    raise ValueError("Duplicate candle timestamps found.")

if candles[["open", "close"]].isna().any().any():
    raise ValueError("Missing OHLC prices found.")

if (candles[["open", "close"]] <= 0).any().any():
    raise ValueError("Invalid OHLC prices found.")

candles = candles.sort_values(
    "timestamp"
).reset_index(drop=True)

predictions = predictions.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# 4. CREATE TRADING DATE
# ============================================================

candles["trading_date"] = (
    candles["timestamp"].dt.date
)

predictions["trading_date"] = (
    predictions["timestamp"].dt.date
)

# Build timestamp-to-open-price lookup.
open_prices = candles.set_index("timestamp")["open"]

# ============================================================
# 5. PREPARE TRADING SIGNALS
# ============================================================

print("\nPreparing UP and DOWN signals...")

signals = predictions[
    predictions["predicted_label"].isin(
        ["UP", "DOWN"]
    )
].copy()

signals = signals.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Total directional signals:", len(signals))

# ============================================================
# 6. EXECUTION RULES
# ============================================================

print("\nApplying execution rules...")

trades = []

skipped_missing_prices = 0
skipped_wrong_day = 0
skipped_overlapping = 0

next_available_entry = None

for _, signal in signals.iterrows():

    signal_time = signal["timestamp"]
    trading_date = signal["trading_date"]

    # Signal becomes available after current candle closes.
    # Enter at the next candle's open.
    entry_time = signal_time + pd.Timedelta(minutes=5)

    # Exit exactly 30 minutes after entry.
    exit_time = entry_time + pd.Timedelta(minutes=30)

    # Do not carry trades overnight.
    if (
        entry_time.date() != trading_date
        or exit_time.date() != trading_date
    ):
        skipped_wrong_day += 1
        continue

    # Require exact candle timestamps.
    if (
        entry_time not in open_prices.index
        or exit_time not in open_prices.index
    ):
        skipped_missing_prices += 1
        continue

    # Avoid overlapping trades.
    # Require a strictly later entry than the previous exit.
    if (
        next_available_entry is not None
        and entry_time <= next_available_entry
    ):
        skipped_overlapping += 1
        continue

    entry_price = float(
        open_prices.loc[entry_time]
    )

    exit_price = float(
        open_prices.loc[exit_time]
    )

    if entry_price <= 0 or exit_price <= 0:
        skipped_missing_prices += 1
        continue

    direction = signal["predicted_label"]

    if direction == "UP":

        gross_return = (
            exit_price / entry_price
        ) - 1

    else:

        gross_return = (
            entry_price - exit_price
        ) / entry_price

    cost_return = (
        ROUND_TRIP_COST_BPS / 10000
    )

    net_return = gross_return - cost_return

    trades.append({

        "signal_time": signal_time,

        "entry_time": entry_time,

        "exit_time": exit_time,

        "trading_date": trading_date,

        "direction": direction,

        "entry_price": entry_price,

        "exit_price": exit_price,

        "gross_return": gross_return,

        "gross_return_pct": gross_return * 100,

        "cost_return_pct": cost_return * 100,

        "net_return": net_return,

        "net_return_pct": net_return * 100,

        "entry_confidence": signal[
            "probability_up"
            if direction == "UP"
            else "probability_down"
        ]

    })

    next_available_entry = exit_time


# ============================================================
# 7. CREATE TRADE DATAFRAME
# ============================================================

trades_df = pd.DataFrame(trades)

print("\nEXECUTION SUMMARY")
print("-" * 70)

print("Original directional signals:", len(signals))
print("Executed trades:", len(trades_df))
print("Skipped - missing prices:", skipped_missing_prices)
print("Skipped - wrong trading day:", skipped_wrong_day)
print("Skipped - overlapping:", skipped_overlapping)

if trades_df.empty:
    raise ValueError(
        "No valid trades generated. Check candle timestamps and data alignment."
    )


# ============================================================
# 8. CALCULATE OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL BACKTEST PERFORMANCE")
print("=" * 70)

gross_returns = trades_df["gross_return"]

net_returns = trades_df["net_return"]

winning_trades = (
    gross_returns > 0
).sum()

losing_trades = (
    gross_returns < 0
).sum()

win_rate = (
    winning_trades / len(trades_df)
) * 100

average_return = gross_returns.mean() * 100

median_return = gross_returns.median() * 100

total_gross_return = (
    (1 + gross_returns).prod() - 1
) * 100

total_net_return = (
    (1 + net_returns).prod() - 1
) * 100

gross_profit = gross_returns[
    gross_returns > 0
].sum()

gross_loss = abs(
    gross_returns[gross_returns < 0].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else np.inf
)

print(f"Total trades:       {len(trades_df)}")
print(f"Winning trades:     {winning_trades}")
print(f"Losing trades:      {losing_trades}")
print(f"Win rate:           {win_rate:.2f}%")
print(f"Average return:     {average_return:.4f}%")
print(f"Median return:      {median_return:.4f}%")
print(f"Gross total return: {total_gross_return:.4f}%")
print(f"Net total return:   {total_net_return:.4f}%")
print(f"Profit factor:      {profit_factor:.4f}")


# ============================================================
# 9. DRAWDOWN ANALYSIS
# ============================================================

equity_curve = (
    1 + net_returns
).cumprod()

running_peak = equity_curve.cummax()

drawdown = (
    equity_curve / running_peak
) - 1

max_drawdown = (
    drawdown.min() * 100
)

print("\nMAXIMUM DRAWDOWN")
print(f"{max_drawdown:.4f}%")


# ============================================================
# 10. DIRECTION-WISE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("DIRECTION-WISE PERFORMANCE")
print("=" * 70)

direction_summary = (
    trades_df.groupby("direction")
    .agg(
        trades=("gross_return", "count"),

        average_gross_return=(
            "gross_return_pct", "mean"
        ),

        median_gross_return=(
            "gross_return_pct", "median"
        ),

        winning_trades=(
            "gross_return",
            lambda x: (x > 0).sum()
        ),

        total_gross_return=(
            "gross_return",
            lambda x: ((1 + x).prod() - 1) * 100
        )
    )
)

direction_summary["win_rate"] = (
    trades_df.groupby("direction")["gross_return"]
    .apply(lambda x: (x > 0).mean() * 100)
)

print(direction_summary.round(4))

direction_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "backtest_direction_summary.csv"
    )
)


# ============================================================
# 11. DAILY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("DAILY PERFORMANCE")
print("=" * 70)

daily = (
    trades_df.groupby("trading_date")
    .agg(
        trades=("net_return", "count"),

        daily_return=(
            "net_return",
            lambda x: ((1 + x).prod() - 1) * 100
        )
    )
)

print(daily.describe().round(4))

daily.to_csv(
    os.path.join(
        REPORT_DIR,
        "backtest_daily_performance.csv"
    )
)


# ============================================================
# 12. BREAKEVEN TRANSACTION COST
# ============================================================

breakeven_cost_bps = (
    gross_returns.mean() * 10000
)

print("\n" + "=" * 70)
print("BREAKEVEN COST ANALYSIS")
print("=" * 70)

print(
    "Average gross return per trade:",
    f"{gross_returns.mean() * 100:.4f}%"
)

print(
    "Approximate breakeven round-trip cost:",
    f"{breakeven_cost_bps:.4f} basis points"
)

print(
    "Configured round-trip cost:",
    f"{ROUND_TRIP_COST_BPS:.2f} basis points"
)


# ============================================================
# 13. SAVE TRADE RECORDS
# ============================================================

trades_df["equity"] = equity_curve

trades_df["drawdown_pct"] = drawdown * 100

trades_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "backtest_trade_results.csv"
    ),
    index=False
)


# ============================================================
# 14. SAVE OVERALL METRICS
# ============================================================

metrics = pd.DataFrame([{

    "total_trades": len(trades_df),

    "winning_trades": winning_trades,

    "losing_trades": losing_trades,

    "win_rate": win_rate,

    "average_gross_return_pct": average_return,

    "median_gross_return_pct": median_return,

    "total_gross_return_pct": total_gross_return,

    "total_net_return_pct": total_net_return,

    "profit_factor": profit_factor,

    "maximum_drawdown_pct": max_drawdown,

    "breakeven_round_trip_cost_bps": breakeven_cost_bps,

    "configured_round_trip_cost_bps": ROUND_TRIP_COST_BPS

}])

metrics.to_csv(
    os.path.join(
        REPORT_DIR,
        "backtest_overall_metrics.csv"
    ),
    index=False
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("BACKTEST COMPLETED")
print("=" * 70)

print("\nGenerated reports:")

print("1. backtest_trade_results.csv")
print("2. backtest_overall_metrics.csv")
print("3. backtest_direction_summary.csv")
print("4. backtest_daily_performance.csv")

print("\nMOSAIC 2.0 - Stage 8 Step 2 completed.")
