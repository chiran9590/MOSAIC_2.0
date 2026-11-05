import json
from datetime import datetime, timezone

from src.paper_trading.database import get_connection


def create_paper_trade(
    strategy_result: dict,
    data_source: str,
    synthetic_data: bool,
) -> dict:
    """Save an accepted strategy candidate as a paper trade."""

    if strategy_result.get("decision") != "PAPER_TRADE_CANDIDATE":
        raise ValueError("Only accepted paper-trade candidates can be saved.")

    if strategy_result.get("risk_accepted") is not True:
        raise ValueError("Risk assessment must be accepted.")

    if strategy_result.get("paper_trading_only") is not True:
        raise ValueError("Paper-trading-only result required.")

    if strategy_result.get("real_order_placed") is not False:
        raise ValueError("Real-order results cannot be stored here.")

    required = [
        "symbol",
        "direction",
        "prediction",
        "candle_timestamp",
        "entry_price",
        "stop_loss",
        "quantity",
        "position_value",
        "estimated_total_risk",
        "probabilities",
        "model_version",
    ]

    missing = [
        field for field in required
        if strategy_result.get(field) is None
    ]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if strategy_result["direction"] not in ("LONG", "SHORT"):
        raise ValueError("Invalid trade direction.")

    if strategy_result["quantity"] <= 0:
        raise ValueError("Quantity must be positive.")

    entry_price = float(strategy_result["entry_price"])
    stop_loss = float(strategy_result["stop_loss"])

    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("Entry price and stop loss must be positive.")

    if strategy_result["direction"] == "LONG" and stop_loss >= entry_price:
        raise ValueError("LONG stop loss must be below entry price.")

    if strategy_result["direction"] == "SHORT" and stop_loss <= entry_price:
        raise ValueError("SHORT stop loss must be above entry price.")

    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        insert_sql = """
            INSERT INTO paper_trades (
                symbol,
                direction,
                prediction,
                candle_timestamp,
                entry_time,
                entry_price,
                stop_loss,
                quantity,
                position_value,
                estimated_risk,
                probabilities,
                model_version,
                data_source,
                synthetic_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                symbol,
                candle_timestamp,
                model_version,
                direction
            ) DO NOTHING
        """

        parameters = (
            strategy_result["symbol"],
            strategy_result["direction"],
            strategy_result["prediction"],
            strategy_result["candle_timestamp"],
            now,
            entry_price,
            stop_loss,
            int(strategy_result["quantity"]),
            float(strategy_result["position_value"]),
            float(strategy_result["estimated_total_risk"]),
            json.dumps(strategy_result["probabilities"]),
            strategy_result["model_version"],
            data_source,
            int(synthetic_data),
        )

        import os

        if os.getenv("DATABASE_URL"):
            cursor = connection.execute(
                insert_sql + " RETURNING id",
                parameters,
            )
            inserted_row = cursor.fetchone()
            inserted_id = inserted_row["id"] if inserted_row else None
        else:
            cursor = connection.execute(insert_sql, parameters)
            inserted_id = cursor.lastrowid if cursor.rowcount else None

        if inserted_id is None:
            row = connection.execute(
                """
                SELECT * FROM paper_trades
                WHERE symbol = ?
                  AND candle_timestamp = ?
                  AND model_version = ?
                  AND direction = ?
                """,
                (
                    strategy_result["symbol"],
                    strategy_result["candle_timestamp"],
                    strategy_result["model_version"],
                    strategy_result["direction"],
                ),
            ).fetchone()

            return {
                "created": False,
                "message": "Duplicate signal. Existing trade was not modified.",
                "trade_id": row["id"],
            }

        return {
            "created": True,
            "message": "Paper trade saved successfully.",
            "trade_id": inserted_id,
        }


def get_all_trades():
    """Return all paper trades, newest first."""

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM paper_trades ORDER BY id DESC"
        ).fetchall()

        return [dict(row) for row in rows]


def get_open_trades():
    """Return only currently open paper trades."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM paper_trades
            WHERE status = 'OPEN'
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def close_paper_trade(
    trade_id: int,
    exit_price: float,
    estimated_round_trip_cost_pct: float = 0.20,
) -> dict:
    """Close an open paper trade and calculate illustrative P&L."""

    if exit_price <= 0:
        raise ValueError("Exit price must be positive.")

    if not 0 <= estimated_round_trip_cost_pct <= 100:
        raise ValueError("Invalid estimated transaction-cost percentage.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM paper_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()

        if row is None:
            raise ValueError("Paper trade not found.")

        if row["status"] != "OPEN":
            raise ValueError("Only OPEN trades can be closed.")

        entry_price = row["entry_price"]
        quantity = row["quantity"]

        if row["direction"] == "LONG":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Illustrative cost estimate on entry and exit turnover.
        turnover = (entry_price + exit_price) * quantity

        estimated_cost = turnover * (
            estimated_round_trip_cost_pct / 100
        )

        net_pnl = gross_pnl - estimated_cost

        exit_time = datetime.now(timezone.utc).isoformat()

        connection.execute(
            """
            UPDATE paper_trades
            SET status = 'CLOSED',
                exit_price = ?,
                exit_time = ?,
                gross_pnl = ?,
                estimated_cost = ?,
                net_pnl = ?
            WHERE id = ? AND status = 'OPEN'
            """,
            (
                exit_price,
                exit_time,
                gross_pnl,
                estimated_cost,
                net_pnl,
                trade_id,
            ),
        )

        return {
            "trade_id": trade_id,
            "status": "CLOSED",
            "gross_pnl": round(gross_pnl, 2),
            "estimated_cost": round(estimated_cost, 2),
            "net_pnl": round(net_pnl, 2),
            "cost_assumption_pct": estimated_round_trip_cost_pct,
        }


def get_trade_performance(starting_capital: float = 100000.0) -> dict:
    """Calculate paper-trading performance and chronological equity curve."""

    if starting_capital <= 0:
        raise ValueError("Starting capital must be greater than zero.")

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_trades,

                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END)
                    AS open_trades,

                SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END)
                    AS closed_trades,

                SUM(CASE
                    WHEN status = 'CLOSED' AND net_pnl > 0 THEN 1
                    ELSE 0
                END) AS winning_trades,

                SUM(CASE
                    WHEN status = 'CLOSED' AND net_pnl < 0 THEN 1
                    ELSE 0
                END) AS losing_trades,

                SUM(CASE
                    WHEN status = 'CLOSED' AND net_pnl = 0 THEN 1
                    ELSE 0
                END) AS breakeven_trades,

                SUM(CASE WHEN status = 'CLOSED'
                    THEN gross_pnl ELSE 0 END) AS gross_pnl,

                SUM(CASE WHEN status = 'CLOSED'
                    THEN estimated_cost ELSE 0 END) AS estimated_cost,

                SUM(CASE WHEN status = 'CLOSED'
                    THEN net_pnl ELSE 0 END) AS net_pnl

            FROM paper_trades
            """
        ).fetchone()

        closed_trades = connection.execute(
            """
            SELECT
                id,
                exit_time,
                net_pnl
            FROM paper_trades
            WHERE status = 'CLOSED'
            ORDER BY exit_time ASC, id ASC
            """
        ).fetchall()

    result = dict(row)

    count_fields = [
        "total_trades",
        "open_trades",
        "closed_trades",
        "winning_trades",
        "losing_trades",
        "breakeven_trades",
    ]

    for field in count_fields:
        result[field] = int(result[field] or 0)

    money_fields = [
        "gross_pnl",
        "estimated_cost",
        "net_pnl",
    ]

    for field in money_fields:
        result[field] = float(round(float(result[field] or 0), 2))

    closed = result["closed_trades"]
    winning = result["winning_trades"]

    result["win_rate_pct"] = round(
        (winning / closed) * 100, 2
    ) if closed else 0.0

    result["average_net_pnl"] = round(
        result["net_pnl"] / closed, 2
    ) if closed else 0.0

    # ----------------------------------------
    # EQUITY CURVE AND DRAWDOWN
    # ----------------------------------------

    equity = float(starting_capital)
    peak_equity = equity

    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    equity_curve = []

    for trade in closed_trades:

        trade_pnl = float(trade["net_pnl"] or 0.0)
        equity += trade_pnl

        peak_equity = max(peak_equity, equity)

        drawdown = peak_equity - equity
        drawdown_pct = (
            (drawdown / peak_equity) * 100
            if peak_equity > 0
            else 0.0
        )

        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        equity_curve.append({
            "trade_id": trade["id"],
            "exit_time": trade["exit_time"],
            "net_pnl": round(trade_pnl, 2),
            "equity": round(equity, 2),
            "peak_equity": round(peak_equity, 2),
            "drawdown": round(drawdown, 2),
            "drawdown_pct": round(drawdown_pct, 2),
        })

    result["starting_capital"] = round(starting_capital, 2)
    result["current_equity"] = round(equity, 2)
    result["cumulative_pnl"] = result["net_pnl"]
    result["max_drawdown"] = round(max_drawdown, 2)
    result["max_drawdown_pct"] = round(max_drawdown_pct, 2)
    result["equity_curve"] = equity_curve

    return result
