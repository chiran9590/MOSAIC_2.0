
import os
import pandas as pd
import numpy as np

print("\nMOSAIC 2.0 - TRANSACTION COST ANALYSIS")
print("=" * 75)

INPUT_PATH = "reports/backtest_v2_trade_results.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Illustrative Zerodha-style equity intraday brokerage
BROKERAGE_RATE = 0.0003
MAX_BROKERAGE_PER_ORDER = 20.0

# Rates expressed as decimal fractions
STT_RATE = 0.00025
NSE_TRANSACTION_RATE = 0.000030699
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
IPFT_RATE = 0.000000001

GST_RATE = 0.18

# Slippage sensitivity: basis points PER SIDE
SLIPPAGE_SCENARIOS = [0.0, 0.5, 1.0, 2.0]

# --------------------------------------------------
# 2. LOAD TRADES
# --------------------------------------------------

print("\nLoading verified V2 trades...")

trades = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "signal_timestamp",
        "entry_timestamp",
        "exit_timestamp"
    ]
)

if trades.empty:
    raise ValueError("No trades found.")

required_columns = [
    "direction",
    "entry_price",
    "exit_price",
    "gross_return"
]

for col in required_columns:
    if col not in trades.columns:
        raise ValueError(f"Missing column: {col}")

if not trades["direction"].isin(["UP", "DOWN"]).all():
    raise ValueError("Unexpected trade direction found.")

if (
    (trades["entry_price"] <= 0).any()
    or (trades["exit_price"] <= 0).any()
):
    raise ValueError("Invalid entry or exit price.")

print("Loaded trades:", len(trades))

# --------------------------------------------------
# 3. CALCULATE BUY/SELL TURNOVER
# --------------------------------------------------

# One share is used as the reference position.
# Results are normalized by entry notional.
#
# UP:
#   Buy at entry, sell at exit.
#
# DOWN:
#   Sell at entry, buy at exit.

trades["entry_notional"] = trades["entry_price"]
trades["exit_notional"] = trades["exit_price"]

trades["buy_turnover"] = np.where(
    trades["direction"] == "UP",
    trades["entry_notional"],
    trades["exit_notional"]
)

trades["sell_turnover"] = np.where(
    trades["direction"] == "UP",
    trades["exit_notional"],
    trades["entry_notional"]
)

trades["total_turnover"] = (
    trades["buy_turnover"]
    + trades["sell_turnover"]
)

# Gross P&L per one-share position
trades["gross_pnl"] = (
    trades["gross_return"]
    * trades["entry_notional"]
)

# --------------------------------------------------
# 4. BROKERAGE
# --------------------------------------------------

def calculate_brokerage(turnover):

    return min(
        MAX_BROKERAGE_PER_ORDER,
        BROKERAGE_RATE * turnover
    )

trades["entry_brokerage"] = (
    trades["entry_notional"].apply(calculate_brokerage)
)

trades["exit_brokerage"] = (
    trades["exit_notional"].apply(calculate_brokerage)
)

trades["total_brokerage"] = (
    trades["entry_brokerage"]
    + trades["exit_brokerage"]
)

# --------------------------------------------------
# 5. STATUTORY CHARGES
# --------------------------------------------------

# STT: sell side only
trades["stt"] = (
    trades["sell_turnover"] * STT_RATE
)

# NSE transaction charges: both sides
trades["exchange_charges"] = (
    trades["total_turnover"]
    * NSE_TRANSACTION_RATE
)

# SEBI turnover charges: both sides
trades["sebi_charges"] = (
    trades["total_turnover"]
    * SEBI_RATE
)

# Stamp duty: buy side only
trades["stamp_duty"] = (
    trades["buy_turnover"]
    * STAMP_DUTY_RATE
)

# NSE IPFT contribution: both sides
trades["ipft_charges"] = (
    trades["total_turnover"]
    * IPFT_RATE
)

# GST on brokerage, exchange and SEBI charges
trades["gst"] = (
    trades["total_brokerage"]
    + trades["exchange_charges"]
    + trades["sebi_charges"]
) * GST_RATE

trades["statutory_and_brokerage_cost"] = (
    trades["total_brokerage"]
    + trades["stt"]
    + trades["exchange_charges"]
    + trades["sebi_charges"]
    + trades["stamp_duty"]
    + trades["ipft_charges"]
    + trades["gst"]
)

# --------------------------------------------------
# 6. SLIPPAGE SENSITIVITY
# --------------------------------------------------

print("\n" + "=" * 75)
print("SLIPPAGE SENSITIVITY ANALYSIS")
print("=" * 75)

scenario_results = []
scenario_trade_data = {}

for slippage_bps in SLIPPAGE_SCENARIOS:

    slippage_rate = slippage_bps / 10000

    # Slippage is applied to both entry and exit turnover.
    slippage_cost = (
        trades["total_turnover"]
        * slippage_rate
    )

    total_cost = (
        trades["statutory_and_brokerage_cost"]
        + slippage_cost
    )

    net_pnl = (
        trades["gross_pnl"]
        - total_cost
    )

    # Normalize by entry notional
    net_return = (
        net_pnl / trades["entry_notional"]
    )

    equity = (
        1 + net_return
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak
    ) - 1

    wins = (net_return > 0).sum()
    losses = (net_return < 0).sum()

    gross_profit = net_pnl[
        net_pnl > 0
    ].sum()

    gross_loss = abs(
        net_pnl[net_pnl < 0].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    compounded_return = equity.iloc[-1] - 1

    average_net_return = net_return.mean()

    total_cost_amount = total_cost.sum()

    scenario_results.append({
        "slippage_bps_per_side": slippage_bps,
        "total_trades": len(trades),
        "gross_pnl_per_share_sum": trades["gross_pnl"].sum(),
        "total_cost_per_share_sum": total_cost_amount,
        "net_pnl_per_share_sum": net_pnl.sum(),
        "average_net_return": average_net_return,
        "win_rate": (net_return > 0).mean(),
        "profit_factor": profit_factor,
        "compounded_net_return": compounded_return,
        "maximum_drawdown": drawdown.min(),
        "winning_trades": wins,
        "losing_trades": losses
    })

    scenario_trade_data[slippage_bps] = pd.DataFrame({
        "signal_timestamp": trades["signal_timestamp"],
        "direction": trades["direction"],
        "entry_price": trades["entry_price"],
        "exit_price": trades["exit_price"],
        "gross_pnl": trades["gross_pnl"],
        "brokerage": trades["total_brokerage"],
        "stt": trades["stt"],
        "exchange_charges": trades["exchange_charges"],
        "sebi_charges": trades["sebi_charges"],
        "stamp_duty": trades["stamp_duty"],
        "ipft_charges": trades["ipft_charges"],
        "gst": trades["gst"],
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "net_return": net_return
    })

    print(f"\nSlippage: {slippage_bps} bps per side")
    print(f"Total costs: Rs. {total_cost_amount:.6f}")
    print(f"Net P&L per reference share: Rs. {net_pnl.sum():.6f}")
    print(f"Average net return: {average_net_return:.4%}")
    print(f"Win rate: {(net_return > 0).mean():.2%}")
    print(f"Profit factor: {profit_factor:.4f}")
    print(f"Compounded return: {compounded_return:.4%}")
    print(f"Maximum drawdown: {drawdown.min():.4%}")

# --------------------------------------------------
# 7. SAVE SCENARIO RESULTS
# --------------------------------------------------

summary = pd.DataFrame(scenario_results)

summary.to_csv(
    f"{OUTPUT_DIR}/backtest_cost_sensitivity.csv",
    index=False
)

# Save the baseline 0.5 bps per side trade report
baseline_slippage = 0.5

scenario_trade_data[baseline_slippage].to_csv(
    f"{OUTPUT_DIR}/backtest_cost_adjusted_trades.csv",
    index=False
)

# --------------------------------------------------
# 8. BASELINE COST BREAKDOWN
# --------------------------------------------------

print("\n" + "=" * 75)
print("BASELINE COST BREAKDOWN")
print("=" * 75)

baseline_costs = pd.DataFrame({
    "Charge": [
        "Brokerage",
        "STT",
        "NSE transaction charges",
        "SEBI charges",
        "Stamp duty",
        "IPFT",
        "GST"
    ],
    "Total per reference share": [
        trades["total_brokerage"].sum(),
        trades["stt"].sum(),
        trades["exchange_charges"].sum(),
        trades["sebi_charges"].sum(),
        trades["stamp_duty"].sum(),
        trades["ipft_charges"].sum(),
        trades["gst"].sum()
    ]
})

print(baseline_costs.to_string(index=False))

# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 75)
print("COST ANALYSIS COMPLETED")
print("=" * 75)

print("\nSlippage sensitivity:")
print(
    summary[
        [
            "slippage_bps_per_side",
            "total_cost_per_share_sum",
            "average_net_return",
            "win_rate",
            "profit_factor",
            "compounded_net_return",
            "maximum_drawdown"
        ]
    ].to_string(index=False)
)

print("\nSaved reports:")
print("reports/backtest_cost_sensitivity.csv")
print("reports/backtest_cost_adjusted_trades.csv")

print("\nIMPORTANT:")
print("One reference share is used for cost normalization.")
print("Actual portfolio P&L depends on position size and capital.")
print("Brokerage and slippage assumptions must be verified.")
