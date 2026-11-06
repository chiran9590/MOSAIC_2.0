import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.mock_market_data import MockMarketDataProvider
from src.data.samco_market_data import SamcoMarketDataProvider
from datetime import datetime
from zoneinfo import ZoneInfo
from src.inference.live_candle_aggregator import aggregate_completed_5min
from src.inference.prediction_engine import MosaicPredictionEngine

from src.strategy.strategy_pipeline import StrategyPipeline
from src.strategy.risk_manager import RiskConfig

from src.paper_trading.database import initialize_database
from src.paper_trading.trade_store import (
    get_all_trades,
    get_open_trades,
    close_paper_trade,
    get_trade_performance,
)


app = FastAPI(
    title="MOSAIC 2.0 API",
    description="Experimental market analysis and paper-trading API",
    version="2.0.0",
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

frontend_url = os.getenv("FRONTEND_URL")

if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MOCK_TRADING_DATE = "2026-11-03"

DEFAULT_RISK_CONFIG = RiskConfig(
    capital=100000.0,
    risk_per_trade_pct=0.5,
    max_position_value_pct=20.0,
    stop_loss_pct=1.0,
    estimated_round_trip_cost_pct=0.20,
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class LivePredictionRequest(BaseModel):
    symbol: str = Field(
        default="RELIANCE",
        pattern=r"^RELIANCE$"
    )


class LiveStrategyRequest(BaseModel):
    symbol: str = Field(
        default="RELIANCE",
        pattern=r"^RELIANCE$"
    )


class StrategyMockRequest(BaseModel):
    entry_price: float | None = Field(default=None, gt=0)


class ClosePaperTradeRequest(BaseModel):
    exit_price: float = Field(gt=0)
    estimated_round_trip_cost_pct: float = Field(
        default=0.20,
        ge=0,
        le=100,
    )


# --------------------------------------------------
# CACHED ENGINES
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_prediction_engine():
    return MosaicPredictionEngine()


@lru_cache(maxsize=1)
def get_strategy_pipeline():
    return StrategyPipeline(risk_config=DEFAULT_RISK_CONFIG)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    initialize_database()


# --------------------------------------------------
# MOCK MARKET DATA HELPER
# --------------------------------------------------

def get_mock_candles():

    provider = MockMarketDataProvider(
        seed=42,
        initial_price=1400.0,
    )

    minute_candles = provider.fetch_minute_candles(
        symbol="RELIANCE",
        trading_date=MOCK_TRADING_DATE,
    )

    five_minute_candles = aggregate_completed_5min(
        minute_candles,
        now="2026-11-03 15:30:00",
    )

    if len(five_minute_candles) != 75:
        raise ValueError(
            f"Expected 75 completed candles, "
            f"received {len(five_minute_candles)}"
        )

    return minute_candles, five_minute_candles


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "MOSAIC 2.0",
        "message": "Market analysis API is running",
        "version": "2.0.0",
        "status": "EXPERIMENTAL",
        "documentation": "/docs",
        "paper_trading_enabled": True,
        "paper_trade_creation_enabled": False,
        "paper_trade_creation_status": "DISABLED_UNTIL_PRODUCTION_APPROVAL",
        "live_order_execution": False,
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MOSAIC 2.0 API",
        "environment": "development",
    }


# --------------------------------------------------
# MOCK PREDICTION
# --------------------------------------------------

@app.post("/api/v1/predict/mock")
def predict_mock():

    try:
        minute_candles, five_minute_candles = get_mock_candles()
        prediction = get_prediction_engine().predict(five_minute_candles)

        return {
            "success": True,
            "symbol": prediction.get("symbol", "RELIANCE"),
            "data_source": "MOCK",
            "synthetic_data": True,
            "trading_date": str(prediction["candle_timestamp"])[:10],
            "timeframe": "5minute",
            "input_candles": len(minute_candles),
            "completed_5min_candles": len(five_minute_candles),
            "last_candle_timestamp": prediction.get(
                "latest_market_candle_timestamp"
            ),
            "prediction_result": prediction,
            "execution": {
                "mode": "PREDICTION_ONLY",
                "real_order_placed": False,
                "broker_order_execution_enabled": False,
            },
            "warning": "Synthetic data only. This is not a live market signal.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Mock prediction failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# MOCK MARKET DATA
# --------------------------------------------------

@app.get("/api/v1/market/mock")
def get_mock_market_data():

    try:
        _, five_minute_candles = get_mock_candles()

        candles = []

        for _, row in five_minute_candles.iterrows():
            candles.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        return {
            "success": True,
            "symbol": "RELIANCE",
            "timeframe": "5minute",
            "data_source": "MOCK",
            "synthetic_data": True,
            "trading_date": MOCK_TRADING_DATE,
            "total_candles": len(candles),
            "candles": candles,
            "warning": "Synthetic data only. Not live market data.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Mock market data failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# MOCK STRATEGY
# --------------------------------------------------

@app.post("/api/v1/strategy/mock")
def evaluate_mock_strategy(request: StrategyMockRequest):

    try:
        _, five_minute_candles = get_mock_candles()

        prediction = get_prediction_engine().predict(
            five_minute_candles
        )

        entry_price = request.entry_price

        if entry_price is None:
            entry_price = float(five_minute_candles.iloc[-1]["close"])

        strategy_result = get_strategy_pipeline().evaluate(
            prediction=prediction,
            entry_price=entry_price,
        )

        return {
            "success": True,
            "symbol": prediction["symbol"],
            "data_source": "MOCK",
            "synthetic_data": True,
            "strategy_result": strategy_result.to_dict(),
            "execution": {
                "mode": "PAPER_TRADING_ONLY",
                "real_order_placed": False,
                "broker_connected": False,
            },
            "warning": (
                "Synthetic data and experimental model output. "
                "Risk estimates use illustrative configuration. "
                "This is not a live trading signal."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Mock strategy evaluation failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# CREATE PAPER TRADE
# --------------------------------------------------

@app.post("/api/v1/paper-trades")
def api_create_paper_trade():

    # Trade creation is deliberately disabled until a model has
    # passed validation and received explicit production approval.
    # Never trust client-supplied strategy or risk-approval fields.
    raise HTTPException(
        status_code=409,
        detail=(
            "Paper-trade creation is currently disabled. "
            "The active MOSAIC model is experimental and has not "
            "received production approval. Existing paper trades "
            "can still be viewed and closed."
        ),
    )


# --------------------------------------------------
# ALL PAPER TRADES
# --------------------------------------------------

@app.get("/api/v1/paper-trades")
def api_get_all_paper_trades():

    try:
        trades = get_all_trades()

        return {
            "success": True,
            "total": len(trades),
            "trades": trades,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve trades: {str(exc)}",
        ) from exc


# --------------------------------------------------
# OPEN PAPER TRADES
# --------------------------------------------------

@app.get("/api/v1/paper-trades/open")
def api_get_open_paper_trades():

    try:
        trades = get_open_trades()

        return {
            "success": True,
            "total": len(trades),
            "trades": trades,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve open trades: {str(exc)}",
        ) from exc


# --------------------------------------------------
# PAPER TRADING PERFORMANCE
# --------------------------------------------------

@app.get("/api/v1/paper-trades/performance")
def api_get_paper_trade_performance():

    try:
        performance = get_trade_performance()

        return {
            "success": True,
            **performance,
            "execution_mode": "PAPER_TRADING_ONLY",
            "real_order_placed": False,
            "warning": (
                "Performance is based on recorded paper trades. "
                "It does not establish live trading profitability."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not calculate trade performance: {str(exc)}",
        ) from exc


# --------------------------------------------------
# CLOSE PAPER TRADE
# --------------------------------------------------

@app.post("/api/v1/paper-trades/{trade_id}/close")
def api_close_paper_trade(
    trade_id: int,
    request: ClosePaperTradeRequest,
):

    try:
        result = close_paper_trade(
            trade_id=trade_id,
            exit_price=request.exit_price,
            estimated_round_trip_cost_pct=(
                request.estimated_round_trip_cost_pct
            ),
        )

        return {
            "success": True,
            **result,
            "execution_mode": "PAPER_TRADING_ONLY",
            "real_order_placed": False,
            "warning": "Transaction costs are illustrative estimates.",
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not close paper trade: {str(exc)}",
        ) from exc

# --------------------------------------------------
# LIVE SAMCO PREDICTION
# --------------------------------------------------

@app.post("/api/v1/predict/live")
def predict_live(request: LivePredictionRequest):

    try:
        # Use the Indian market's local date.
        current_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        trading_date = current_ist.strftime("%Y-%m-%d")

        # Fetch real market data from Samco.
        provider = SamcoMarketDataProvider()

        minute_candles = provider.fetch_minute_candles(
            symbol=request.symbol,
            trading_date=trading_date,
        )

        if minute_candles.empty:
            raise HTTPException(
                status_code=503,
                detail="Samco returned no market candles for today."
            )

        # Aggregate only completed five-minute candles.
        five_minute_candles = aggregate_completed_5min(
            minute_candles
        )

        if five_minute_candles.empty:
            raise HTTPException(
                status_code=503,
                detail="No completed five-minute candles are available yet."
            )

        # Run the existing experimental prediction model.
        prediction = get_prediction_engine().predict(
            five_minute_candles
        )

        return {
            "success": True,
            "symbol": request.symbol,
            "data_source": "SAMCO",
            "synthetic_data": False,
            "trading_date": trading_date,
            "timeframe": "5minute",
            "input_candles": len(minute_candles),
            "completed_5min_candles": len(five_minute_candles),
            "last_candle_timestamp": (
                five_minute_candles.iloc[-1]["timestamp"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
            "prediction_result": prediction,
            "execution": {
                "mode": "PREDICTION_ONLY",
                "real_order_placed": False,
                "broker_order_execution_enabled": False,
            },
            "warning": (
                "Real market data with an experimental prediction model. "
                "This is not a validated trading signal or a guarantee "
                "of future market movement."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live Samco prediction failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# LIVE SAMCO STRATEGY
# --------------------------------------------------

@app.post("/api/v1/strategy/live")
def evaluate_live_strategy(request: LiveStrategyRequest):

    try:
        # 1. Get current Indian market date.
        current_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        trading_date = current_ist.strftime("%Y-%m-%d")

        # 2. Fetch real Samco market data.
        provider = SamcoMarketDataProvider()

        minute_candles = provider.fetch_minute_candles(
            symbol=request.symbol,
            trading_date=trading_date,
        )

        if minute_candles.empty:
            raise HTTPException(
                status_code=503,
                detail="Samco returned no market candles."
            )

        # 3. Aggregate completed five-minute candles.
        five_minute_candles = aggregate_completed_5min(
            minute_candles
        )

        if five_minute_candles.empty:
            raise HTTPException(
                status_code=503,
                detail="No completed five-minute candles available."
            )

        # 4. Generate ML prediction.
        prediction = get_prediction_engine().predict(
            five_minute_candles
        )

        # 5. Use latest market close as indicative entry price.
        entry_price = float(
            five_minute_candles.iloc[-1]["close"]
        )

        # 6. Evaluate strategy and risk.
        strategy_result = get_strategy_pipeline().evaluate(
            prediction=prediction,
            entry_price=entry_price,
        )

        strategy_data = strategy_result.to_dict()

        # 7. Block paper-trade eligibility for stale predictions.
        prediction_is_fresh = bool(
            prediction.get("is_latest_candle_prediction", False)
        )

        if not prediction_is_fresh:
            strategy_data.update({
                "decision": "NO_TRADE_STALE_PREDICTION",
                "direction": None,
                "risk_accepted": False,
                "entry_price": None,
                "stop_loss": None,
                "quantity": 0,
                "position_value": 0.0,
                "estimated_total_risk": 0.0,
                "risk_reason": (
                    "Prediction is not based on the latest "
                    "completed market candle."
                ),
                "reason": (
                    "Paper-trade candidate blocked because "
                    "the prediction is stale."
                ),
            })

        paper_trade_eligible = (
            prediction_is_fresh
            and strategy_data["decision"] == "PAPER_TRADE_CANDIDATE"
            and strategy_data["risk_accepted"]
        )

        # 8. Return complete strategy response.
        return {
            "success": True,
            "symbol": request.symbol,
            "data_source": "SAMCO",
            "synthetic_data": False,
            "trading_date": trading_date,
            "timeframe": "5minute",
            "input_candles": len(minute_candles),
            "completed_5min_candles": len(five_minute_candles),
            "last_candle_timestamp": (
                five_minute_candles.iloc[-1]["timestamp"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
            "prediction_result": prediction,
            "strategy_result": strategy_data,
            "paper_trade_eligible": paper_trade_eligible,
            "execution": {
                "mode": "PAPER_TRADING_ONLY",
                "real_order_placed": False,
                "broker_connected": False,
            },
            "warning": (
                "Real Samco market data with an experimental model. "
                "Strategy results are for paper trading only. "
                "No real broker orders are placed."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live Samco strategy failed: {str(exc)}",
        ) from exc
