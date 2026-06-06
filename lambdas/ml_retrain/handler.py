import json
import boto3
import os
import pickle
import io
import csv
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET", "btc-raw-archive")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/isolation_forest.pkl")
SCALER_KEY = "models/scaler.pkl"
METRICS_KEY = "models/last_training_metrics.json"
MIN_SAMPLES = 100  # minimum records needed to retrain


# ── AWS CLIENTS ───────────────────────────────────────────────────────────────
s3 = boto3.client("s3")
athena = boto3.client("athena", region_name="eu-west-2")


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_training_data() -> list[dict] | None:
    """
    Load all raw BTC records from S3.
    Walks the raw/ prefix and reads every JSON file.
    """
    try:
        records = []

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix="raw/BTC-USD/")

        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue

                response = s3.get_object(Bucket=S3_BUCKET, Key=key)
                body = response["Body"].read().decode("utf-8")
                record = json.loads(body)
                records.append(record)

        print(f"[DATA] Loaded {len(records)} records from S3")
        return records if len(records) >= MIN_SAMPLES else None

    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        return None


def extract_features(records: list[dict]) -> np.ndarray | None:
    """
    Extract numerical features from raw records for model training.
    Features: close, volume, price_range, open, high, low
    """
    try:
        features = []

        for r in records:
            try:
                row = [
                    float(r["close"]),
                    float(r["volume"]),
                    float(r["high"]) - float(r["low"]),   # price range
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                ]
                features.append(row)
            except (KeyError, ValueError) as e:
                print(f"[WARN] Skipping malformed record: {e}")
                continue

        if len(features) < MIN_SAMPLES:
            print(f"[WARN] Not enough clean features: {len(features)}")
            return None

        return np.array(features)

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_model(features: np.ndarray) -> tuple | None:
    """
    Scale features and train Isolation Forest.
    Returns (model, scaler, metrics) tuple.
    """
    try:
        # Scale features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        # Train Isolation Forest
        # contamination=0.05 means we expect ~5% anomalies
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(scaled)

        # Compute basic training metrics
        predictions = model.predict(scaled)
        anomaly_count = int(np.sum(predictions == -1))
        anomaly_rate = round(anomaly_count / len(predictions), 4)

        metrics = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(features),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "n_estimators": 100,
            "contamination": 0.05,
            "features": ["close", "volume", "price_range", "open", "high", "low"],
        }

        print(f"[TRAIN] Complete — samples={len(features)} anomaly_rate={anomaly_rate}")
        return model, scaler, metrics

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return None


# ── MODEL PERSISTENCE ─────────────────────────────────────────────────────────
def save_artifact(obj, key: str) -> bool:
    """Serialise a Python object and upload to S3."""
    try:
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        print(f"[S3] Saved artifact: {key}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to save artifact {key}: {e}")
        return False


def save_metrics(metrics: dict) -> None:
    """Save training metrics as JSON to S3."""
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=METRICS_KEY,
            Body=json.dumps(metrics, indent=2),
            ContentType="application/json",
        )
        print(f"[S3] Saved training metrics")

    except Exception as e:
        print(f"[ERROR] Failed to save metrics: {e}")


# ── MAIN HANDLER ──────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    """
    Triggered weekly by EventBridge (every Sunday midnight).
    Loads all raw BTC data from S3, retrains Isolation Forest,
    saves model and scaler back to S3.
    """
    print(f"[START] ML retrain triggered at {datetime.now(timezone.utc).isoformat()}")

    # 1. Load raw data from S3
    records = load_training_data()
    if records is None:
        msg = f"Insufficient data for retraining (minimum {MIN_SAMPLES} samples required)"
        print(f"[SKIP] {msg}")
        return {"statusCode": 200, "message": msg}

    # 2. Extract features
    features = extract_features(records)
    if features is None:
        msg = "Feature extraction failed or insufficient clean records"
        print(f"[SKIP] {msg}")
        return {"statusCode": 200, "message": msg}

    # 3. Train model
    result = train_model(features)
    if result is None:
        return {"statusCode": 500, "message": "Training failed"}

    model, scaler, metrics = result

    # 4. Save model, scaler and metrics to S3
    model_saved = save_artifact(model, MODEL_KEY)
    scaler_saved = save_artifact(scaler, SCALER_KEY)
    save_metrics(metrics)

    if not model_saved or not scaler_saved:
        return {"statusCode": 500, "message": "Failed to save model artifacts"}

    print(f"[DONE] Retrain complete — {metrics}")

    return {
        "statusCode": 200,
        "message": "Retrain successful",
        "metrics": metrics,
    }