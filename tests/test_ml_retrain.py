import json
import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# ── PATH SETUP ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../lambdas/ml_retrain"))
import handler as ml


# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_raw_record(**overrides) -> dict:
    """Return a valid raw BTC record."""
    base = {
        "symbol": "BTC-USD",
        "timestamp": "2024-01-15T10:00:00+00:00",
        "open": 42000.0,
        "high": 42500.0,
        "low": 41800.0,
        "close": 42200.0,
        "volume": 1500.0,
    }
    base.update(overrides)
    return base


def make_records(n: int) -> list[dict]:
    """Generate n valid BTC records with slight price variation."""
    return [make_raw_record(close=42000.0 + i * 10) for i in range(n)]


# ── FEATURE EXTRACTION TESTS ──────────────────────────────────────────────────
class TestExtractFeatures:

    def test_returns_correct_shape(self):
        records = make_records(150)
        features = ml.extract_features(records)
        assert features is not None
        assert features.shape == (150, 6)

    def test_returns_none_when_too_few_records(self):
        records = make_records(50)
        features = ml.extract_features(records)
        assert features is None

    def test_skips_malformed_records(self):
        records = make_records(150)
        records[5] = {"symbol": "BTC-USD", "timestamp": "bad"}  # malformed
        features = ml.extract_features(records)
        # Should still return features for valid records
        assert features is not None
        assert features.shape[0] == 149

    def test_feature_values_are_correct(self):
        record = make_raw_record(
            open=42000.0, high=42500.0, low=41800.0,
            close=42200.0, volume=1500.0
        )
        records = [record] * 150
        features = ml.extract_features(records)
        assert features is not None
        # price_range should be high - low
        assert features[0][2] == pytest.approx(42500.0 - 41800.0)


# ── TRAINING TESTS ────────────────────────────────────────────────────────────
class TestTrainModel:

    def test_returns_model_scaler_metrics(self):
        features = np.random.rand(200, 6) * 42000
        result = ml.train_model(features)
        assert result is not None
        model, scaler, metrics = result
        assert model is not None
        assert scaler is not None
        assert isinstance(metrics, dict)

    def test_metrics_contain_required_keys(self):
        features = np.random.rand(200, 6) * 42000
        _, _, metrics = ml.train_model(features)
        required_keys = [
            "trained_at", "training_samples", "anomaly_count",
            "anomaly_rate", "n_estimators", "contamination", "features"
        ]
        for key in required_keys:
            assert key in metrics

    def test_anomaly_rate_is_between_0_and_1(self):
        features = np.random.rand(200, 6) * 42000
        _, _, metrics = ml.train_model(features)
        assert 0.0 <= metrics["anomaly_rate"] <= 1.0

    def test_training_samples_matches_input(self):
        features = np.random.rand(200, 6) * 42000
        _, _, metrics = ml.train_model(features)
        assert metrics["training_samples"] == 200


# ── INTEGRATION TESTS ─────────────────────────────────────────────────────────
class TestLambdaHandler:

    @patch("handler.save_artifact", return_value=True)
    @patch("handler.save_metrics")
    @patch("handler.train_model")
    @patch("handler.extract_features")
    @patch("handler.load_training_data")
    def test_successful_retrain_returns_200(
        self, mock_load, mock_features, mock_train,
        mock_metrics, mock_save
    ):
        mock_load.return_value = make_records(150)
        mock_features.return_value = np.random.rand(150, 6)
        mock_train.return_value = (MagicMock(), MagicMock(), {"anomaly_rate": 0.05, "training_samples": 150})

        result = ml.lambda_handler({}, {})

        assert result["statusCode"] == 200
        assert result["message"] == "Retrain successful"

    @patch("handler.load_training_data", return_value=None)
    def test_insufficient_data_returns_200_with_skip_message(self, mock_load):
        result = ml.lambda_handler({}, {})
        assert result["statusCode"] == 200
        assert "Insufficient" in result["message"]

    @patch("handler.save_artifact", return_value=False)
    @patch("handler.save_metrics")
    @patch("handler.train_model")
    @patch("handler.extract_features")
    @patch("handler.load_training_data")
    def test_failed_save_returns_500(
        self, mock_load, mock_features, mock_train,
        mock_metrics, mock_save
    ):
        mock_load.return_value = make_records(150)
        mock_features.return_value = np.random.rand(150, 6)
        mock_train.return_value = (MagicMock(), MagicMock(), {"anomaly_rate": 0.05, "training_samples": 150})

        result = ml.lambda_handler({}, {})

        assert result["statusCode"] == 500