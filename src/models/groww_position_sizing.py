
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# MOSAIC 2.0 - STAGE 8 STEP 6
# GROWW TRANSACTION COST & POSITION SIZING ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "reports" / "backtest_cost_adjusted_trades.csv"
OUTPUT_DIR = ROOT / "reports"

QUANTITIES = [1, 10, 25, 50, 100]
SLIPPAGE_SCENARIOS = [0.0, 0.5, 1.0, 2.0]

# Groww NSE equity intraday charges
BROKERAGE_RATE = 0.001       # 0.1% per order
BROKERAGE_MAX = 20.0
BROKERAGE_MIN = 5.0
BROKERAGE_MIN_RATE = 0.025   # 2.5% minimum-charge cap

STT_RATE = 0.00025           # Sell side only
STAMP_RATE = 0.00003         # Buy side only
EXCHANGE_RATE = 0.0000297    # Both sides
SEBI_RATE = 0.000001         # Both sides
IPFT_RATE = 0.000001         # Both sides
GST_RATE = 0.18

# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, description):
    lookup = {str(c).lower().strip(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def brokerage(order_value):
    """
    Groww brokerage:
    ₹20 or 0.1% per order, whichever is lower.
    If calculated brokerage is below ₹5, apply
    the lower of ₹5 or 2.5% of order value.
    """
    calculated = min(BROKERAGE_MAX, BROKERAGE_RATE * order_value)

    if calculated < BROKERAGE_MIN:
        calculated = min(
            BROKERAGE_MIN,
            BROKERAGE_MIN_RATE * order_value
        )

    return calculated


def calculate_charges(buy_value, sell_value):
    buy_brokerage = brokerage(buy_value)
    sell_brokerage = brokerage(sell_value)

    total_brokerage = buy_brokerage + sell_brokerage

    stt = sell_value * STT_RATE
    stamp = buy_value * STAMP_RATE

    exchange = (buy_value + sell_value) * EXCHANGE_RATE
    sebi = (buy_value + sell_value) * SEBI_RATE
    ipft = (buy_value + sell_value) * IPFT_RATE

    gst = GST_RATE * (
        total_brokerage + exchange + sebi + ipft
    )

    total = (
        total_brokerage
        + stt
        + stamp
        + exchange
        + sebi
        + ipft
        + gst
    )

    return {
        "brokerage": total_brokerage,
        "stt": stt,
        "stamp_duty": stamp,
        "exchange": exchange,
        "sebi": sebi,
        "ipft": ipft,
        "gst": gst,
        "total_charges": total
    }


def max_drawdown_from_pnl(pnl_series):
    """
    Drawdown of the cumulative fixed-quantity P&L curve.
    Starts at zero; does not assume reinvestment.
    """
    equity = np.concatenate([[0.0], np.cumsum(pnl_series)])
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity - running_peak

    return float(drawdown.min())


def profit_factor(pnl):
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())

    if losses == 0:
        return np.inf if gains > 0 else 0.0

    return float(gains / losses)


# ============================================================
# LOAD DATA
# ============================================================

print("\nMOSAIC 2.0 - GROWW POSITION SIZING ANALYSIS")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Expected reports/backtest_cost_adjusted_trades.csv"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded records: {len(df)}")
print("Available columns:")
print(list(df.columns))

# Identify required fields
entry_col = find_column(
    df,
    ["entry_price", "entry", "entry_open"],
    "entry price"
)

exit_col = find_column(
    df,
    ["exit_price", "exit", "exit_open"],
    "exit price"
)

direction_col = find_column(
    df,
    ["direction", "signal", "predicted_label",
     "predicted_direction", "label"],
    "trade direction"
)

# Optional timestamps
entry_time_col = next(
    (c for c in df.columns
     if c.lower() in ["entry_time", "entry_timestamp"]),
    None
)

exit_time_col = next(
    (c for c in df.columns
     if c.lower() in ["exit_time", "exit_timestamp"]),
    None
)

df[entry_col] = pd.to_numeric(df[entry_col], errors="coerce")
df[exit_col] = pd.to_numeric(df[exit_col], errors="coerce")

df = df.dropna(subset=[entry_col, exit_col]).copy()

# Normalize direction
def normalize_direction(value):
    value = str(value).strip().upper()

    if value in ["UP", "BUY", "LONG"]:
        return "UP"

    if value in ["DOWN", "SELL", "SHORT"]:
        return "DOWN"

    raise ValueError(f"Unknown trade direction: {value}")


df["direction_clean"] = df[direction_col].apply(
    normalize_direction
)

df = df[
    (df[entry_col] > 0) &
    (df[exit_col] > 0)
].copy()

if entry_time_col:
    df["sort_time"] = pd.to_datetime(
        df[entry_time_col], errors="coerce"
    )
    df = df.sort_values("sort_time")

df = df.reset_index(drop=True)

print(f"Valid trades loaded: {len(df)}")

# ============================================================
# SIMULATION
# ============================================================

trade_results = []
summary_results = []

for quantity in QUANTITIES:

    for slippage_bps in SLIPPAGE_SCENARIOS:

        slippage_rate = slippage_bps / 10000.0

        rows = []

        for _, trade in df.iterrows():

            entry = float(trade[entry_col])
            exit_price = float(trade[exit_col])
            direction = trade["direction_clean"]

            # Apply adverse execution slippage
            if direction == "UP":
                # Buy first, sell later
                actual_entry = entry * (1 + slippage_rate)
                actual_exit = exit_price * (1 - slippage_rate)

                gross_pnl = quantity * (exit_price - entry)

                net_execution_pnl = quantity * (
                    actual_exit - actual_entry
                )

                buy_value = quantity * actual_entry
                sell_value = quantity * actual_exit

            else:
                # Short sell first, buy back later
                actual_entry = entry * (1 - slippage_rate)
                actual_exit = exit_price * (1 + slippage_rate)

                gross_pnl = quantity * (entry - exit_price)

                net_execution_pnl = quantity * (
                    actual_entry - actual_exit
                )

                sell_value = quantity * actual_entry
                buy_value = quantity * actual_exit

            charges = calculate_charges(
                buy_value,
                sell_value
            )

            net_pnl = (
                net_execution_pnl
                - charges["total_charges"]
            )

            entry_notional = quantity * entry

            net_return = (
                net_pnl / entry_notional
                if entry_notional > 0 else 0
            )

            row = {
                "quantity": quantity,
                "slippage_bps_per_side": slippage_bps,
                "direction": direction,
                "entry_price": entry,
                "exit_price": exit_price,
                "actual_entry": actual_entry,
                "actual_exit": actual_exit,
                "entry_notional": entry_notional,
                "gross_pnl": gross_pnl,
                "execution_pnl_after_slippage": net_execution_pnl,
                **charges,
                "net_pnl": net_pnl,
                "net_return_pct": net_return * 100
            }

            if entry_time_col:
                row["entry_time"] = trade[entry_time_col]

            if exit_time_col:
                row["exit_time"] = trade[exit_time_col]

            rows.append(row)

        result = pd.DataFrame(rows)
        result["cumulative_net_pnl"] = result["net_pnl"].cumsum()

        total_gross = result["gross_pnl"].sum()
        total_net = result["net_pnl"].sum()
        total_charges = result["total_charges"].sum()
        total_notional = result["entry_notional"].sum()

        wins = (result["net_pnl"] > 0).sum()
        losses = (result["net_pnl"] < 0).sum()
        breakeven = (result["net_pnl"] == 0).sum()

        win_rate = wins / len(result) * 100

        average_net_return = result["net_return_pct"].mean()

        pf = profit_factor(result["net_pnl"].to_numpy())

        drawdown = max_drawdown_from_pnl(
            result["net_pnl"].to_numpy()
        )

        max_exposure = result["entry_notional"].max()

        drawdown_pct_of_max_exposure = (
            drawdown / max_exposure * 100
            if max_exposure > 0 else 0
        )

        summary = {
            "quantity": quantity,
            "slippage_bps_per_side": slippage_bps,
            "trades": len(result),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate_pct": win_rate,
            "gross_pnl_rs": total_gross,
            "slippage_impact_rs": (
                total_gross -
                result["execution_pnl_after_slippage"].sum()
            ),
            "total_charges_rs": total_charges,
            "net_pnl_rs": total_net,
            "average_net_return_pct": average_net_return,
            "profit_factor": pf,
            "max_drawdown_rs": drawdown,
            "max_exposure_rs": max_exposure,
            "drawdown_pct_of_max_exposure":
                drawdown_pct_of_max_exposure,
            "total_entry_notional_rs": total_notional,
            "average_cost_per_trade_rs":
                total_charges / len(result)
        }

        summary_results.append(summary)

        result["quantity"] = quantity
        result["slippage_bps_per_side"] = slippage_bps
        trade_results.append(result)

        print(
            f"\nQuantity: {quantity} shares | "
            f"Slippage: {slippage_bps} bps per side"
        )
        print(f"Gross P&L:       Rs. {total_gross:,.2f}")
        print(f"Groww charges:   Rs. {total_charges:,.2f}")
        print(f"Net P&L:         Rs. {total_net:,.2f}")
        print(f"Win rate:        {win_rate:.2f}%")
        print(f"Profit factor:   {pf:.4f}")
        print(f"Max drawdown:    Rs. {drawdown:,.2f}")

# ============================================================
# SAVE REPORTS
# ============================================================

summary_df = pd.DataFrame(summary_results)
trades_df = pd.concat(trade_results, ignore_index=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = OUTPUT_DIR / "groww_position_sizing_summary.csv"
trades_path = OUTPUT_DIR / "groww_position_sizing_trades.csv"

summary_df.to_csv(summary_path, index=False)
trades_df.to_csv(trades_path, index=False)

# Also create a quantity comparison at zero slippage
zero_slippage = summary_df[
    summary_df["slippage_bps_per_side"] == 0
].copy()

zero_path = OUTPUT_DIR / "groww_quantity_comparison.csv"
zero_slippage.to_csv(zero_path, index=False)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("GROWW POSITION SIZING - ZERO SLIPPAGE COMPARISON")
print("=" * 70)

display_columns = [
    "quantity",
    "trades",
    "gross_pnl_rs",
    "total_charges_rs",
    "net_pnl_rs",
    "win_rate_pct",
    "profit_factor",
    "max_drawdown_rs"
]

print(
    zero_slippage[display_columns].to_string(
        index=False,
        formatters={
            "gross_pnl_rs": "{:,.2f}".format,
            "total_charges_rs": "{:,.2f}".format,
            "net_pnl_rs": "{:,.2f}".format,
            "win_rate_pct": "{:.2f}".format,
            "profit_factor": "{:.4f}".format,
            "max_drawdown_rs": "{:,.2f}".format
        }
    )
)

print("\nREPORTS SAVED")
print(f"1. {summary_path}")
print(f"2. {trades_path}")
print(f"3. {zero_path}")

print("\nIMPORTANT NOTES")
print("- Fixed quantities are used; no reinvestment is assumed.")
print("- Drawdown is measured on cumulative fixed-quantity net P&L.")
print("- Short-position margin is not estimated.")
print("- Auto square-off penalties are excluded.")
print("- Slippage is applied adversely on both execution legs.")
print("- These are historical simulations, not live trading results.")

print("\nSTAGE 8 STEP 6 COMPLETE")
