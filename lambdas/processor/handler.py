import json
import base64
import boto3
import os
from datetime import datetime, timezone
from collections import deque
import statistics


# ── CONFIG ────────────────────────────────────────────────────────────────────
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "btc-market-data")
S3_BUCKET = os.environ.get("S3_BUCKET", "btc-raw-archive")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ZSCORE_THRESHOLD = float(os.environ.get("ZSCORE_THRESHOLD", "2.5"))
ROLLING_WINDOW = 20  # number of data points for rolling stats


# ── AWS CLIENTS ───────────────────────────────────────────────────────────────
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
sns = boto3.client("sns")
table = dynamodb.Table(DYNAMODB_TABLE)


# ── ROLLING WINDOW (in-memory per Lambda instance) ───────────────────────────
price_window = deque(maxlen=ROLLING_WINDOW)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def decode_record(encoded_data: str) -> dict | None:
    """Decode base64 Kinesis record into a dict."""
    try:
        raw = base64.b64decode(encoded_data).decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        print(f"[ERROR] Failed to decode record: {e}")
        return None


def validate_record(record: dict) -> bool:
    """Check all required fields are present and values are sensible."""
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

    for field in required:
        if field not in record:
            print(f"[WARN] Missing field: {field}")
            return False

    if record["close"] <= 0 or record["volume"] < 0:
        print(f"[WARN] Invalid values: close={record['close']} volume={record['volume']}")
        return False

    if record["high"] < record["low"]:
        print(f"[WARN] High < Low: high={record['high']} low={record['low']}")
        return False

    return True


def compute_zscore(price: float) -> float | None:
    """Compute z-score of price against rolling window."""
    if len(price_window) < ROLLING_WINDOW:
        return None  # not enough data yet

    mean = statistics.mean(price_window)
    stdev = statistics.stdev(price_window)

    if stdev == 0:
        return 0.0

    return (price - mean) / stdev


def enrich_record(record: dict, zscore: float | None) -> dict:
    """Add derived fields to the record."""
    record["price_range"] = round(record["high"] - record["low"], 2)
    record["zscore"] = round(zscore, 4) if zscore is not None else None
    record["is_anomaly"] = (
        abs(zscore) > ZSCORE_THRESHOLD if zscore is not None else False
    )
    record["processed_at"] = datetime.now(timezone.utc).isoformat()
    return record


# ── STORAGE ───────────────────────────────────────────────────────────────────
def save_to_dynamodb(record: dict) -> None:
    """Write cleaned enriched record to DynamoDB."""
    try:
        date = record["timestamp"][:10]  # YYYY-MM-DD

        table.put_item(Item={
            "date": date,
            "timestamp": record["timestamp"],
            "symbol": record["symbol"],
            "open": str(record["open"]),
            "high": str(record["high"]),
            "low": str(record["low"]),
            "close": str(record["close"]),
            "volume": str(record["volume"]),
            "price_range": str(record["price_range"]),
            "zscore": str(record["zscore"]) if record["zscore"] is not None else "N/A",
            "is_anomaly": record["is_anomaly"],
            "processed_at": record["processed_at"],
        })
        print(f"[DynamoDB] Saved record: {record['timestamp']}")

    except Exception as e:
        print(f"[ERROR] DynamoDB write failed: {e}")


def archive_to_s3(record: dict) -> None:
    """Archive raw record to S3 as JSON."""
    try:
        timestamp = record["timestamp"].replace(":", "-").replace("+", "")
        key = f"raw/{record['symbol']}/{timestamp[:10]}/{timestamp}.json"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(record),
            ContentType="application/json",
        )
        print(f"[S3] Archived to: {key}")

    except Exception as e:
        print(f"[ERROR] S3 archive failed: {e}")


# ── ALERTING ──────────────────────────────────────────────────────────────────
def send_anomaly_alert(record: dict) -> None:
    """Publish anomaly alert to SNS."""
    try:
        message = (
            f"ANOMALY DETECTED — {record['symbol']}\n\n"
            f"Timestamp : {record['timestamp']}\n"
            f"Close     : ${record['close']}\n"
            f"Z-Score   : {record['zscore']}\n"
            f"Threshold : ±{ZSCORE_THRESHOLD}\n"
            f"Range     : ${record['price_range']}\n\n"
            f"This price movement is statistically significant "
            f"based on the last {ROLLING_WINDOW} data points."
        )

        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[BTC ALERT] Anomaly detected at ${record['close']}",
            Message=message,
        )
        print(f"[SNS] Alert sent for anomaly at {record['timestamp']}")

    except Exception as e:
        print(f"[ERROR] SNS publish failed: {e}")


# ── MAIN HANDLER ──────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    """
    Triggered by Kinesis stream.
    Each event contains a batch of records.
    """
    processed = 0
    anomalies = 0
    skipped = 0

    print(f"[START] Processing batch of {len(event['Records'])} records")

    for kinesis_record in event["Records"]:

        # 1. Decode
        record = decode_record(kinesis_record["kinesis"]["data"])
        if record is None:
            skipped += 1
            continue

        # 2. Validate
        if not validate_record(record):
            skipped += 1
            continue

        # 3. Update rolling window
        price_window.append(record["close"])

        # 4. Compute z-score
        zscore = compute_zscore(record["close"])

        # 5. Enrich
        record = enrich_record(record, zscore)

        # 6. Store
        save_to_dynamodb(record)
        archive_to_s3(record)

        # 7. Alert if anomaly
        if record["is_anomaly"]:
            anomalies += 1
            print(f"[ANOMALY] z={record['zscore']} close=${record['close']}")
            send_anomaly_alert(record)

        processed += 1

    print(f"[DONE] processed={processed} anomalies={anomalies} skipped={skipped}")

    return {
        "statusCode": 200,
        "processed": processed,
        "anomalies": anomalies,
        "skipped": skipped,
    }