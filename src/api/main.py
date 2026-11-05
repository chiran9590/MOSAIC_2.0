import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.mock_market_data import MockMarketDataProvider
from src.inference.live_candle_aggregator import aggregate_completed_5min
from src.inference.prediction_engine import MosaicPredictionEngine

from src.strategy.strategy_pipeline import StrategyPipeline
from src.strategy.risk_manager import RiskConfig

from src.paper_trading.database import initialize_database
from src.paper_trading.trade_store import (
    create_paper_trade,
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

class StrategyMockRequest(BaseModel):
    entry_price: float | None = Field(default=None, gt=0)


class CreatePaperTradeRequest(BaseModel):
    strategy_result: dict


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
            "data_source": "MOCK",
            "synthetic_data": True,
            "input_candles": len(minute_candles),
            "completed_5min_candles": len(five_minute_candles),
            "prediction_result": prediction,
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
def api_create_paper_trade(request: CreatePaperTradeRequest):

    try:
        result = create_paper_trade(
            strategy_result=request.strategy_result,
            data_source="MOCK",
            synthetic_data=True,
        )

        return {
            "success": True,
            **result,
            "execution_mode": "PAPER_TRADING_ONLY",
            "real_order_placed": False,
            "warning": "This record uses synthetic market data.",
        }

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save paper trade: {str(exc)}",
        ) from exc


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