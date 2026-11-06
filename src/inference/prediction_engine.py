import json
import os
import joblib
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

from src.inference.feature_engine import calculate_live_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

DEFAULT_MODEL_NAME = "mosaic_candidate_no_macd_v1"

SUPPORTED_MODELS = {
    "mosaic_candidate_no_macd_v1",
    "mosaic_5min_candidate_v1",
}


class MosaicPredictionEngine:

    def __init__(self, model_name: str | None = None):

        selected_model = (
            model_name
            or os.getenv("MOSAIC_MODEL_NAME")
            or DEFAULT_MODEL_NAME
        )

        if selected_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported MOSAIC model: {selected_model}. "
                f"Supported models: {sorted(SUPPORTED_MODELS)}"
            )

        self.model_name = selected_model

        model_dir = PROJECT_ROOT / "models" / "saved"

        self.model_path = model_dir / f"{selected_model}.joblib"
        self.encoder_path = model_dir / f"{selected_model}_encoder.joblib"
        self.features_path = model_dir / f"{selected_model}_features.json"
        self.metadata_path = model_dir / f"{selected_model}_metadata.json"

        required_files = [
            self.model_path,
            self.encoder_path,
            self.features_path,
            self.metadata_path,
        ]

        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required model artifact not found: {path}"
                )

        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)

        with open(self.features_path, "r") as file:
            self.features = json.load(file)

        with open(self.metadata_path, "r") as file:
            self.metadata = json.load(file)

        if not isinstance(self.features, list) or not self.features:
            raise ValueError("Model features must be a non-empty list.")

        if len(self.features) != self.metadata.get("feature_count"):
            raise ValueError(
                "Feature count does not match model metadata."
            )

        if self.metadata.get("model_name") != self.model_name:
            raise ValueError("Model name does not match metadata.")

        horizon_text = self.metadata.get("prediction_horizon", "")
        horizon_parts = str(horizon_text).strip().lower().split()

        if (
            len(horizon_parts) != 2
            or horizon_parts[1] not in {"minute", "minutes"}
        ):
            raise ValueError(
                f"Invalid prediction horizon in metadata: {horizon_text}"
            )

        self.prediction_horizon_minutes = int(horizon_parts[0])

        if self.prediction_horizon_minutes <= 0:
            raise ValueError("Prediction horizon must be positive.")

        self.model_status = str(
            self.metadata.get("status", "UNKNOWN")
        ).upper()

        print("MOSAIC prediction engine loaded.")
        print(f"Model: {self.model_name}")
        print(f"Features: {len(self.features)}")
        print(
            "Prediction horizon: "
            f"{self.prediction_horizon_minutes} minutes"
        )
        print(f"Model status: {self.model_status}")

    def predict(self, candles: pd.DataFrame):

        if candles.empty:
            raise ValueError("No market candles supplied.")

        feature_df = calculate_live_features(candles)

        valid_rows = feature_df.dropna(subset=self.features)

        if valid_rows.empty:
            raise ValueError(
                "Not enough valid candles to calculate model features."
            )

        latest = valid_rows.iloc[-1]

        latest_market_timestamp = candles.iloc[-1]["timestamp"]
        prediction_timestamp = latest["timestamp"]

        matching_indices = candles.index[
            candles["timestamp"] == prediction_timestamp
        ]

        if len(matching_indices) == 0:
            raise ValueError(
                "Could not match prediction timestamp to market candles."
            )

        prediction_position = candles.index.get_indexer(
            [matching_indices[-1]]
        )[0]

        if prediction_position < 0:
            raise ValueError(
                "Prediction timestamp must identify one candle."
            )

        latest_position = len(candles) - 1
        prediction_age = latest_position - prediction_position

        X = latest[self.features].to_frame().T

        predicted_encoded = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        predicted_label = self.encoder.inverse_transform(
            [predicted_encoded]
        )[0]

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                self.encoder.classes_,
                probabilities,
            )
        }

        return {
            "symbol": "RELIANCE",
            "prediction": str(predicted_label),
            "probabilities": probability_map,
            "candle_timestamp": str(prediction_timestamp),
            "latest_market_candle_timestamp": str(
                latest_market_timestamp
            ),
            "prediction_age_candles": int(prediction_age),
            "is_latest_candle_prediction": bool(
                prediction_age == 0
            ),
            "prediction_horizon_minutes": (
                self.prediction_horizon_minutes
            ),
            "model_version": self.model_name,
            "status": self.model_status,
        }
