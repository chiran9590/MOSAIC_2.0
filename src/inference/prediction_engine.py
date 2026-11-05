
import json
import joblib
import pandas as pd
from pathlib import Path

from src.inference.feature_engine import calculate_live_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models/saved/mosaic_candidate_no_macd_v1.joblib"
ENCODER_PATH = PROJECT_ROOT / "models/saved/mosaic_candidate_no_macd_v1_encoder.joblib"
FEATURES_PATH = PROJECT_ROOT / "models/saved/mosaic_candidate_no_macd_v1_features.json"
METADATA_PATH = PROJECT_ROOT / "models/saved/mosaic_candidate_no_macd_v1_metadata.json"


class MosaicPredictionEngine:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)

        with open(FEATURES_PATH, "r") as file:
            self.features = json.load(file)

        with open(METADATA_PATH, "r") as file:
            self.metadata = json.load(file)

        if len(self.features) != 19:
            raise ValueError(
                f"Expected 19 features, found {len(self.features)}"
            )

        print("MOSAIC prediction engine loaded.")
        print(f"Model: {MODEL_PATH.name}")
        print(f"Features: {len(self.features)}")

    def predict(self, candles: pd.DataFrame):

        # Calculate features using the verified feature engine.
        feature_df = calculate_live_features(candles)

        # Select the latest row containing all model features.
        valid_rows = feature_df.dropna(subset=self.features)

        if valid_rows.empty:
            raise ValueError(
                "Not enough valid candles to calculate model features."
            )

        latest = valid_rows.iloc[-1]

        X = latest[self.features].to_frame().T

        # Generate prediction and probabilities.
        predicted_encoded = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        predicted_label = self.encoder.inverse_transform(
            [predicted_encoded]
        )[0]

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                self.encoder.classes_,
                probabilities
            )
        }

        return {
            "symbol": "RELIANCE",
            "prediction": str(predicted_label),
            "probabilities": probability_map,
            "candle_timestamp": str(latest["timestamp"]),
            "prediction_horizon_minutes": 30,
            "model_version": "mosaic_candidate_no_macd_v1",
            "status": "EXPERIMENTAL"
        }


if __name__ == "__main__":

    candles_path = PROJECT_ROOT / "data/processed/reliance_5min.csv"

    candles = pd.read_csv(candles_path)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])

    engine = MosaicPredictionEngine()

    result = engine.predict(candles)

    print("\n" + "=" * 65)
    print("MOSAIC 2.0 - SAMPLE PREDICTION")
    print("=" * 65)

    print(f"Symbol:       {result['symbol']}")
    print(f"Prediction:   {result['prediction']}")
    print(f"Candle time:  {result['candle_timestamp']}")
    print(f"Horizon:      {result['prediction_horizon_minutes']} minutes")
    print(f"Status:       {result['status']}")

    print("\nCLASS PROBABILITIES")
    for label, probability in result["probabilities"].items():
        print(f"{label:12}: {probability:.4f} ({probability * 100:.2f}%)")

    print("=" * 65)
