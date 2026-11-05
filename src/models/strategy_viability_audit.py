
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# MOSAIC 2.0
# STAGE 8 STEP 7 - STRATEGY VIABILITY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRADES_FILE = ROOT / "reports" / "backtest_cost_adjusted_trades.csv"
CANDLES_FILE = ROOT / "data" / "processed" / "reliance_5min.csv"

OUTPUT_DIR = ROOT / "reports"

QUANTITIES = [1, 10, 25, 50, 100]
SLIPPAGE_SCENARIOS = [0.0, 0.5, 1.0, 2.0]

HOLDING_MINUTES = 30
ENTRY_DELAY_MINUTES = 5

# ============================================================
# GROWW NSE EQUITY INTRADAY CHARGES
# ============================================================

BROKERAGE_RATE = 0.001
BROKERAGE_MAX = 20.0
BROKERAGE_MIN = 5.0
BROKERAGE_MIN_RATE = 0.025

STT_RATE = 0.00025
STAMP_RATE = 0.00003
EXCHANGE_RATE = 0.0000297
SEBI_RATE = 0.000001
IPFT_RATE = 0.000001
GST_RATE = 0.18

# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, description):

    lookup = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def parse_timestamps(series):

    result = pd.to_datetime(
        series,
        errors="coerce"
    )

    # Convert timezone-aware timestamps to Indian time.
    if isinstance(result.dtype, pd.DatetimeTZDtype):

        result = (
            result.dt
            .tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )

    return result


def normalize_direction(value):

    value = str(value).strip().upper()

    if value in ["UP", "BUY", "LONG"]:
        return "UP"

    if value in ["DOWN", "SELL", "SHORT"]:
        return "DOWN"

    raise ValueError(
        f"Unexpected direction: {value}"
    )


def get_brokerage(order_value):

    calculated = min(
        BROKERAGE_MAX,
        BROKERAGE_RATE * order_value
    )

    if calculated < BROKERAGE_MIN:

        calculated = min(
            BROKERAGE_MIN,
            BROKERAGE_MIN_RATE * order_value
        )

    return calculated


def calculate_groww_charges(
    buy_value,
    sell_value
):

    buy_brokerage = get_brokerage(buy_value)
    sell_brokerage = get_brokerage(sell_value)

    brokerage = (
        buy_brokerage + sell_brokerage
    )

    stt = sell_value * STT_RATE

    stamp = buy_value * STAMP_RATE

    exchange = (
        buy_value + sell_value
    ) * EXCHANGE_RATE

    sebi = (
        buy_value + sell_value
    ) * SEBI_RATE

    ipft = (
        buy_value + sell_value
    ) * IPFT_RATE

    gst = GST_RATE * (
        brokerage + exchange + sebi + ipft
    )

    total = (
        brokerage
        + stt
        + stamp
        + exchange
        + sebi
        + ipft
        + gst
    )

    return {
        "brokerage": brokerage,
        "stt": stt,
        "stamp_duty": stamp,
        "exchange_charges": exchange,
        "sebi_charges": sebi,
        "ipft_charges": ipft,
        "gst": gst,
        "total_charges": total
    }


def calculate_profit_factor(values):

    values = np.asarray(values, dtype=float)

    gains = values[values > 0].sum()

    losses = abs(
        values[values < 0].sum()
    )

    if losses == 0:

        return (
            float("inf")
            if gains > 0
            else 0.0
        )

    return float(gains / losses)


def safe_mean(series):

    return (
        float(series.mean())
        if len(series) > 0
        else np.nan
    )


# ============================================================
# LOAD TRADES
# ============================================================

print("\nMOSAIC 2.0 - STRATEGY VIABILITY AUDIT")
print("=" * 75)

if not TRADES_FILE.exists():
    raise FileNotFoundError(TRADES_FILE)

if not CANDLES_FILE.exists():
    raise FileNotFoundError(CANDLES_FILE)

trades = pd.read_csv(TRADES_FILE)

candles = pd.read_csv(CANDLES_FILE)

print("\nINPUT DATA")
print("-" * 75)

print("Trade records:", len(trades))
print("Candle records:", len(candles))

print("\nTrade columns:")
print(list(trades.columns))

print("\nCandle columns:")
print(list(candles.columns))

# Identify trade columns

signal_col = find_column(
    trades,
    ["signal_timestamp", "timestamp"],
    "signal timestamp"
)

direction_col = find_column(
    trades,
    ["direction"],
    "direction"
)

entry_col = find_column(
    trades,
    ["entry_price"],
    "entry price"
)

exit_col = find_column(
    trades,
    ["exit_price"],
    "exit price"
)

# Identify candle columns

candle_time_col = find_column(
    candles,
    ["timestamp", "datetime", "date"],
    "candle timestamp"
)

open_col = find_column(
    candles,
    ["open"],
    "candle open"
)

high_col = find_column(
    candles,
    ["high"],
    "candle high"
)

low_col = find_column(
    candles,
    ["low"],
    "candle low"
)

close_col = find_column(
    candles,
    ["close"],
    "candle close"
)

# Normalize timestamps

trades["signal_time"] = parse_timestamps(
    trades[signal_col]
)

candles["candle_time"] = parse_timestamps(
    candles[candle_time_col]
)

# Normalize numerical columns

for col in [entry_col, exit_col]:

    trades[col] = pd.to_numeric(
        trades[col],
        errors="coerce"
    )

for col in [
    open_col,
    high_col,
    low_col,
    close_col
]:

    candles[col] = pd.to_numeric(
        candles[col],
        errors="coerce"
    )

trades["direction_clean"] = (
    trades[direction_col].apply(
        normalize_direction
    )
)

trades = trades.dropna(
    subset=[
        "signal_time",
        entry_col,
        exit_col
    ]
).copy()

candles = candles.dropna(
    subset=[
        "candle_time",
        open_col,
        high_col,
        low_col,
        close_col
    ]
).copy()

candles = candles.sort_values(
    "candle_time"
)

candles = candles.drop_duplicates(
    subset=["candle_time"]
)

candles = candles.set_index(
    "candle_time"
).sort_index()

print("\nCLEANED INPUT")
print("-" * 75)

print("Valid trades:", len(trades))
print("Valid candles:", len(candles))

print(
    "Candle date range:",
    candles.index.min(),
    "to",
    candles.index.max()
)

# ============================================================
# RECONSTRUCT EXECUTION TIMING
# ============================================================

trades["entry_time"] = (
    trades["signal_time"]
    + pd.Timedelta(
        minutes=ENTRY_DELAY_MINUTES
    )
)

trades["exit_time"] = (
    trades["entry_time"]
    + pd.Timedelta(
        minutes=HOLDING_MINUTES
    )
)

# Validate entry and exit prices against candle opens.

timing_checks = []

for _, trade in trades.iterrows():

    entry_time = trade["entry_time"]
    exit_time = trade["exit_time"]

    entry_exists = entry_time in candles.index
    exit_exists = exit_time in candles.index

    actual_entry_open = np.nan
    actual_exit_open = np.nan

    if entry_exists:

        actual_entry_open = float(
            candles.loc[entry_time, open_col]
        )

    if exit_exists:

        actual_exit_open = float(
            candles.loc[exit_time, open_col]
        )

    entry_difference = (
        abs(actual_entry_open - trade[entry_col])
        if entry_exists
        else np.nan
    )

    exit_difference = (
        abs(actual_exit_open - trade[exit_col])
        if exit_exists
        else np.nan
    )

    timing_checks.append({

        "signal_time": trade["signal_time"],

        "entry_time": entry_time,

        "exit_time": exit_time,

        "entry_candle_exists": entry_exists,

        "exit_candle_exists": exit_exists,

        "entry_price_difference": entry_difference,

        "exit_price_difference": exit_difference

    })

timing_df = pd.DataFrame(timing_checks)

valid_timing = (
    timing_df["entry_candle_exists"]
    & timing_df["exit_candle_exists"]
    & (timing_df["entry_price_difference"] < 0.000001)
    & (timing_df["exit_price_difference"] < 0.000001)
)

print("\nEXECUTION TIMING AUDIT")
print("-" * 75)

print(
    "Trades with matching entry and exit candles:",
    int(valid_timing.sum())
)

print(
    "Trades with timing or price mismatches:",
    int((~valid_timing).sum())
)

# Stop if execution prices cannot be reconstructed.
if not valid_timing.all():

    print(
        "\nWARNING: Some trade execution prices "
        "do not match the candle data."
    )

    print(
        "Inspect the timing report before trusting "
        "the excursion calculations."
    )

# ============================================================
# CALCULATE MFE AND MAE
# ============================================================

excursion_rows = []

for _, trade in trades.iterrows():

    direction = trade["direction_clean"]

    entry = float(trade[entry_col])

    exit_price = float(trade[exit_col])

    entry_time = trade["entry_time"]

    exit_time = trade["exit_time"]

    # Holding interval:
    # entry candle included; exit candle excluded.
    period = candles.loc[
        (candles.index >= entry_time)
        & (candles.index < exit_time)
    ]

    if len(period) == 0:

        excursion_rows.append({

            "signal_time": trade["signal_time"],

            "direction": direction,

            "mfe_bps": np.nan,

            "mae_bps": np.nan,

            "candles_in_holding_period": 0

        })

        continue

    highest = float(period[high_col].max())

    lowest = float(period[low_col].min())

    if direction == "UP":

        mfe = (highest / entry - 1) * 10000

        mae = (lowest / entry - 1) * 10000

        gross_return_bps = (
            exit_price / entry - 1
        ) * 10000

    else:

        mfe = (entry - lowest) / entry * 10000

        mae = (entry - highest) / entry * 10000

        gross_return_bps = (
            entry - exit_price
        ) / entry * 10000

    excursion_rows.append({

        "signal_time": trade["signal_time"],

        "entry_time": entry_time,

        "exit_time": exit_time,

        "direction": direction,

        "entry_price": entry,

        "exit_price": exit_price,

        "gross_return_bps": gross_return_bps,

        "mfe_bps": mfe,

        "mae_bps": mae,

        "candles_in_holding_period": len(period)

    })

excursions = pd.DataFrame(excursion_rows)

# ============================================================
# GROSS DIRECTIONAL ANALYSIS
# ============================================================

print("\nDIRECTIONAL PERFORMANCE")
print("-" * 75)

direction_rows = []

for direction in ["UP", "DOWN"]:

    subset = excursions[
        excursions["direction"] == direction
    ].dropna(
        subset=["gross_return_bps"]
    )

    gross_values = (
        subset["gross_return_bps"]
        .to_numpy()
    )

    if len(subset) == 0:
        continue

    direction_rows.append({

        "direction": direction,

        "trades": len(subset),

        "average_gross_return_bps":
            safe_mean(subset["gross_return_bps"]),

        "median_gross_return_bps":
            subset["gross_return_bps"].median(),

        "gross_win_rate_pct":
            (subset["gross_return_bps"] > 0).mean() * 100,

        "gross_profit_factor":
            calculate_profit_factor(gross_values),

        "average_mfe_bps":
            safe_mean(subset["mfe_bps"]),

        "average_mae_bps":
            safe_mean(subset["mae_bps"]),

        "mfe_positive_pct":
            (subset["mfe_bps"] > 0).mean() * 100

    })

direction_summary = pd.DataFrame(
    direction_rows
)

print(
    direction_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# GROWW COST AND BREAK-EVEN ANALYSIS
# ============================================================

print("\nGROWW COST AND BREAK-EVEN ANALYSIS")
print("-" * 75)

cost_rows = []

for quantity in QUANTITIES:

    for slippage_bps in SLIPPAGE_SCENARIOS:

        slippage_rate = (
            slippage_bps / 10000
        )

        pnl_values = []

        gross_values = []

        charge_values = []

        slippage_values = []

        breakeven_values = []

        for _, trade in trades.iterrows():

            entry = float(trade[entry_col])

            exit_price = float(trade[exit_col])

            direction = trade["direction_clean"]

            if direction == "UP":

                actual_entry = (
                    entry * (1 + slippage_rate)
                )

                actual_exit = (
                    exit_price * (1 - slippage_rate)
                )

                gross_pnl = (
                    quantity * (exit_price - entry)
                )

                execution_pnl = (
                    quantity *
                    (actual_exit - actual_entry)
                )

                buy_value = quantity * actual_entry

                sell_value = quantity * actual_exit

            else:

                actual_entry = (
                    entry * (1 - slippage_rate)
                )

                actual_exit = (
                    exit_price * (1 + slippage_rate)
                )

                gross_pnl = (
                    quantity * (entry - exit_price)
                )

                execution_pnl = (
                    quantity *
                    (actual_entry - actual_exit)
                )

                sell_value = quantity * actual_entry

                buy_value = quantity * actual_exit

            charges = calculate_groww_charges(
                buy_value,
                sell_value
            )

            total_charges = charges[
                "total_charges"
            ]

            slippage_impact = (
                gross_pnl - execution_pnl
            )

            net_pnl = (
                execution_pnl - total_charges
            )

            entry_notional = quantity * entry

            # Approximate gross movement required to
            # cover both slippage and transaction charges.
            required_bps = (
                (slippage_impact + total_charges)
                / entry_notional
                * 10000
            )

            pnl_values.append(net_pnl)

            gross_values.append(gross_pnl)

            charge_values.append(total_charges)

            slippage_values.append(slippage_impact)

            breakeven_values.append(required_bps)

        pnl_values = np.asarray(pnl_values)

        gross_values = np.asarray(gross_values)

        charge_values = np.asarray(charge_values)

        slippage_values = np.asarray(slippage_values)

        cost_rows.append({

            "quantity": quantity,

            "slippage_bps_per_side": slippage_bps,

            "trades": len(pnl_values),

            "gross_pnl_rs": gross_values.sum(),

            "slippage_cost_rs":
                slippage_values.sum(),

            "groww_charges_rs":
                charge_values.sum(),

            "net_pnl_rs":
                pnl_values.sum(),

            "net_profitable_trade_pct":
                (pnl_values > 0).mean() * 100,

            "profit_factor":
                calculate_profit_factor(pnl_values),

            "average_cost_per_trade_rs":
                charge_values.mean(),

            "average_break_even_bps":
                np.mean(breakeven_values),

            "median_break_even_bps":
                np.median(breakeven_values)

        })

cost_summary = pd.DataFrame(cost_rows)

zero_slippage_costs = cost_summary[
    cost_summary["slippage_bps_per_side"] == 0
]

print(
    zero_slippage_costs.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# TIME OF DAY ANALYSIS
# ============================================================

print("\nTIME OF DAY ANALYSIS")
print("-" * 75)

time_df = excursions.dropna(
    subset=["gross_return_bps"]
).copy()

time_df["entry_hour"] = (
    pd.to_datetime(time_df["entry_time"])
    .dt.hour
)

time_summary = (
    time_df.groupby(
        ["direction", "entry_hour"]
    )
    .agg(
        trades=("gross_return_bps", "size"),

        average_gross_return_bps=(
            "gross_return_bps",
            "mean"
        ),

        median_gross_return_bps=(
            "gross_return_bps",
            "median"
        ),

        gross_win_rate_pct=(
            "gross_return_bps",
            lambda x: (x > 0).mean() * 100
        ),

        average_mfe_bps=("mfe_bps", "mean"),

        average_mae_bps=("mae_bps", "mean")

    )
    .reset_index()
)

print(
    time_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# OVERALL EXCURSION SUMMARY
# ============================================================

print("\nMFE / MAE SUMMARY")
print("-" * 75)

valid_excursions = excursions.dropna(
    subset=["mfe_bps", "mae_bps"]
)

print(
    "Trades with excursion data:",
    len(valid_excursions)
)

print(
    "Average MFE:",
    round(valid_excursions["mfe_bps"].mean(), 4),
    "bps"
)

print(
    "Average MAE:",
    round(valid_excursions["mae_bps"].mean(), 4),
    "bps"
)

print(
    "Median MFE:",
    round(valid_excursions["mfe_bps"].median(), 4),
    "bps"
)

print(
    "Median MAE:",
    round(valid_excursions["mae_bps"].median(), 4),
    "bps"
)

# ============================================================
# SAVE REPORTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timing_path = (
    OUTPUT_DIR /
    "viability_timing_audit.csv"
)

excursion_path = (
    OUTPUT_DIR /
    "viability_mfe_mae.csv"
)

direction_path = (
    OUTPUT_DIR /
    "viability_direction_summary.csv"
)

cost_path = (
    OUTPUT_DIR /
    "viability_groww_break_even.csv"
)

time_path = (
    OUTPUT_DIR /
    "viability_time_of_day.csv"
)

timing_df.to_csv(
    timing_path,
    index=False
)

excursions.to_csv(
    excursion_path,
    index=False
)

direction_summary.to_csv(
    direction_path,
    index=False
)

cost_summary.to_csv(
    cost_path,
    index=False
)

time_summary.to_csv(
    time_path,
    index=False
)

# ============================================================
# FINAL INTERPRETATION CHECKS
# ============================================================

print("\n" + "=" * 75)

print("STAGE 8 STEP 7 - FINAL AUDIT SUMMARY")

print("=" * 75)

if len(direction_summary) > 0:

    for _, row in direction_summary.iterrows():

        print(
            f"{row['direction']}: "
            f"{row['trades']} trades | "
            f"Gross win rate: "
            f"{row['gross_win_rate_pct']:.2f}% | "
            f"Average gross return: "
            f"{row['average_gross_return_bps']:.4f} bps"
        )

print("\nExecution timing checks:")
print("Total trades:", len(timing_df))

print(
    "Exact execution matches:",
    int(valid_timing.sum())
)

print(
    "Execution mismatches:",
    int((~valid_timing).sum())
)

print("\nReports saved:")

for path in [
    timing_path,
    excursion_path,
    direction_path,
    cost_path,
    time_path
]:

    print(path)

print("\nIMPORTANT:")
print("- No model retraining was performed.")
print("- No thresholds were changed.")
print("- No new trading rules were optimized.")
print("- MFE/MAE use intraperiod candle highs and lows.")
print("- Intraperiod order of high and low is unknown.")
print("- Costs are estimated using Groww's published schedule.")
print("- This is historical research, not a live trading system.")

print("\nSTAGE 8 STEP 7 COMPLETE")
