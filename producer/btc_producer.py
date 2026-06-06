import json
import time
import boto3
import yfinance as yf
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
QUEUE_URL = "https://sqs.eu-west-2.amazonaws.com/329599618422/btc-market-queue"
REGION = "eu-west-2"
INTERVAL_SECONDS = 60

# ── SQS CLIENT ────────────────────────────────────────────────────────────────
sqs = boto3.client("sqs", region_name=REGION)


def fetch_btc_tick() -> dict | None:
    try:
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period="1d", interval="1m")

        if df.empty:
            print("[WARN] Empty dataframe returned from yfinance")
            return None

        latest = df.iloc[-1]
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "symbol": "BTC-USD",
            "timestamp": timestamp,
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "close": round(float(latest["Close"]), 2),
            "volume": round(float(latest["Volume"]), 2),
        }

    except Exception as e:
        print(f"[ERROR] Failed to fetch BTC tick: {e}")
        return None


def send_to_sqs(record: dict) -> None:
    try:
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(record),
        )
        print(f"[OK] Sent to SQS | MessageId={response['MessageId']} | {record['timestamp']} | close=${record['close']}")

    except Exception as e:
        print(f"[ERROR] Failed to send to SQS: {e}")


def run():
    print(f"[START] BTC producer running — streaming to SQS every {INTERVAL_SECONDS}s")
    print("[INFO] Press Ctrl+C to stop\n")

    while True:
        tick = fetch_btc_tick()
        if tick:
            print(f"[DATA] {tick}")
            send_to_sqs(tick)
        else:
            print("[SKIP] No data this tick")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()