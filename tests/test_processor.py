import json
import base64
import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from collections import deque

# ── MOCK AWS CLIENTS BEFORE IMPORT ────────────────────────────────────────────
import unittest.mock
sys.modules["boto3"] = unittest.mock.MagicMock()

# ── PATH SETUP ────────────────────────────────────────────────────────────────
processor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lambdas/processor"))

if processor_path not in sys.path:
    sys.path.insert(0, processor_path)

# Force fresh import under unique name to avoid collision with ml_retrain handler
import importlib.util
spec = importlib.util.spec_from_file_location(
    "processor_handler",
    os.path.join(processor_path, "handler.py")
)
processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processor)

# Re-attach real deque so price_window behaves correctly
processor.price_window = deque(maxlen=20)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_kinesis_event(records: list) -> dict:
    """Wrap a list of dicts into a mock Kinesis event."""
    kinesis_records = []
    for r in records:
        encoded = base64.b64encode(json.dumps(r).encode("utf-8")).decode("utf-8")
        kinesis_records.append({
            "kinesis": {
                "data": encoded,
                "sequenceNumber": "1",
                "approximateArrivalTimestamp": 1234567890,
            }
        })
    return {"Records": kinesis_records}


def make_valid_record(**overrides) -> dict:
    """Return a valid BTC record, with optional field overrides."""
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


# ── DECODE TESTS ──────────────────────────────────────────────────────────────
class TestDecodeRecord:

    def test_valid_record_decodes_correctly(self):
        record = make_valid_record()
        encoded = base64.b64encode(json.dumps(record).encode()).decode()
        result = processor.decode_record(encoded)
        assert result == record

    def test_invalid_base64_returns_none(self):
        result = processor.decode_record("not-valid-base64!!!")
        assert result is None

    def test_invalid_json_returns_none(self):
        bad_json = base64.b64encode(b"not json at all").decode()
        result = processor.decode_record(bad_json)
        assert result is None


# ── VALIDATION TESTS ──────────────────────────────────────────────────────────
class TestValidateRecord:

    def test_valid_record_passes(self):
        assert processor.validate_record(make_valid_record()) is True

    def test_missing_field_fails(self):
        record = make_valid_record()
        del record["close"]
        assert processor.validate_record(record) is False

    def test_negative_close_fails(self):
        assert processor.validate_record(make_valid_record(close=-100.0)) is False

    def test_zero_close_fails(self):
        assert processor.validate_record(make_valid_record(close=0.0)) is False

    def test_negative_volume_fails(self):
        assert processor.validate_record(make_valid_record(volume=-1.0)) is False

    def test_high_less_than_low_fails(self):
        assert processor.validate_record(make_valid_record(high=100.0, low=500.0)) is False


# ── ZSCORE TESTS ──────────────────────────────────────────────────────────────
class TestComputeZscore:

    def setup_method(self):
        processor.price_window.clear()

    def test_returns_none_when_window_not_full(self):
        processor.price_window.extend([42000.0] * 10)
        result = processor.compute_zscore(42000.0)
        assert result is None

    def test_returns_zero_for_value_at_mean(self):
        processor.price_window.extend([42000.0] * 20)
        result = processor.compute_zscore(42000.0)
        assert result == 0.0

    def test_detects_high_zscore(self):
        base_prices = [42000.0 + (i % 5) * 100 for i in range(20)]
        processor.price_window.extend(base_prices)
        result = processor.compute_zscore(100000.0)
        assert result is not None
        assert abs(result) > 2.5

    def test_returns_zero_when_stdev_is_zero(self):
        processor.price_window.extend([42000.0] * 20)
        result = processor.compute_zscore(42000.0)
        assert result == 0.0


# ── ENRICH TESTS ──────────────────────────────────────────────────────────────
class TestEnrichRecord:

    def test_price_range_computed_correctly(self):
        record = make_valid_record(high=42500.0, low=41800.0)
        enriched = processor.enrich_record(record, zscore=0.5)
        assert enriched["price_range"] == round(42500.0 - 41800.0, 2)

    def test_anomaly_flag_true_when_zscore_exceeds_threshold(self):
        record = make_valid_record()
        enriched = processor.enrich_record(record, zscore=3.0)
        assert enriched["is_anomaly"] is True

    def test_anomaly_flag_false_when_zscore_below_threshold(self):
        record = make_valid_record()
        enriched = processor.enrich_record(record, zscore=1.0)
        assert enriched["is_anomaly"] is False

    def test_anomaly_flag_false_when_zscore_is_none(self):
        record = make_valid_record()
        enriched = processor.enrich_record(record, zscore=None)
        assert enriched["is_anomaly"] is False

    def test_processed_at_field_added(self):
        record = make_valid_record()
        enriched = processor.enrich_record(record, zscore=0.0)
        assert "processed_at" in enriched


# ── INTEGRATION TESTS ─────────────────────────────────────────────────────────
class TestLambdaHandler:

    def setup_method(self):
        processor.price_window.clear()

    def test_normal_record_processed_no_alert(self):
        base_prices = [42000.0 + (i % 5) * 100 for i in range(20)]
        processor.price_window.extend(base_prices)
        processor.save_to_dynamodb = MagicMock()
        processor.archive_to_s3 = MagicMock()
        processor.send_anomaly_alert = MagicMock()

        record = make_valid_record(close=42100.0)
        event = make_kinesis_event([record])

        result = processor.lambda_handler(event, {})

        assert result["statusCode"] == 200
        assert result["processed"] == 1
        assert result["anomalies"] == 0
        processor.save_to_dynamodb.assert_called_once()
        processor.archive_to_s3.assert_called_once()
        processor.send_anomaly_alert.assert_not_called()

    def test_anomalous_record_triggers_alert(self):
        processor.price_window.extend([42000.0] * 20)
        processor.save_to_dynamodb = MagicMock()
        processor.archive_to_s3 = MagicMock()
        processor.send_anomaly_alert = MagicMock()

        record = make_valid_record(close=200000.0, high=200500.0, low=199000.0)
        event = make_kinesis_event([record])

        result = processor.lambda_handler(event, {})

        assert result["anomalies"] == 1
        processor.send_anomaly_alert.assert_called_once()

    def test_invalid_record_is_skipped(self):
        processor.save_to_dynamodb = MagicMock()
        processor.archive_to_s3 = MagicMock()
        processor.send_anomaly_alert = MagicMock()

        record = make_valid_record(close=-999.0)
        event = make_kinesis_event([record])

        result = processor.lambda_handler(event, {})

        assert result["skipped"] == 1
        assert result["processed"] == 0
        processor.save_to_dynamodb.assert_not_called()

    def test_batch_of_mixed_records(self):
        base_prices = [42000.0 + (i % 5) * 100 for i in range(20)]
        processor.price_window.extend(base_prices)
        processor.save_to_dynamodb = MagicMock()
        processor.archive_to_s3 = MagicMock()
        processor.send_anomaly_alert = MagicMock()

        records = [
            make_valid_record(close=42100.0),
            make_valid_record(close=-500.0),
            make_valid_record(close=42050.0),
        ]
        event = make_kinesis_event(records)

        result = processor.lambda_handler(event, {})

        assert result["processed"] == 2
        assert result["skipped"] == 1