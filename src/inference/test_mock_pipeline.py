import pandas as pd

from src.data.mock_market_data import MockMarketDataProvider
from src.inference.live_candle_aggregator import aggregate_completed_5min
from src.inference.feature_engine import get_latest_features
from src.inference.prediction_engine import MosaicPredictionEngine


def main():
    print("=" * 65)
    print("MOSAIC 2.0 - MOCK INFERENCE PIPELINE TEST")
    print("=" * 65)

    provider = MockMarketDataProvider(seed=42, initial_price=1400.0)

    print("\n1. Generating synthetic 1-minute candles...")
    minute_candles = provider.fetch_minute_candles(
        symbol="RELIANCE",
        trading_date="2026-11-03"
    )
    print(f"   Generated candles: {len(minute_candles)}")

    print("\n2. Aggregating completed 5-minute candles...")
    five_minute_candles = aggregate_completed_5min(
        minute_candles,
        now="2026-11-03 15:30:00"
    )
    print(f"   Completed 5-minute candles: {len(five_minute_candles)}")

    if len(five_minute_candles) != 75:
        raise RuntimeError(
            f"Expected 75 candles, received {len(five_minute_candles)}"
        )

    print("\n3. Calculating live features...")
    latest_features = get_latest_features(five_minute_candles)
    print(f"   Latest feature shape: {latest_features.shape}")
    print(f"   Latest feature timestamp: {latest_features.index[-1]}")

    print("\n4. Running MOSAIC prediction engine...")
    engine = MosaicPredictionEngine()
    prediction = engine.predict(five_minute_candles)

    print("\n5. Prediction result")
    print("-" * 65)
    print(prediction)
    print("-" * 65)

    print("\nPIPELINE TEST COMPLETED")
    print("IMPORTANT: This is synthetic data, not a live market signal.")


if __name__ == "__main__":
    main()
