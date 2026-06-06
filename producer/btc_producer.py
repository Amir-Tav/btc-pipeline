import json
import time
import boto3
import yfinance as yf
from datetime import datetime,timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_NAME = "btc-market-stream"
REGION = "eu-west-2"
INTERVAL_SECONDS = 60  # fetch every 60 seconds

# ── KINESIS CLIENT ────────────────────────────────────────────────────────────
kinesis = boto3.client("kinesis", region_name=REGION)


def fetch_btc_tick() -> dict | None:
    """Fetch the latest BTC-USD tick from Yahoo Finance."""
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


def send_to_kinesis(record: dict) -> None:
    """Send a single record to the Kinesis stream."""
    try:
        response = kinesis.put_record(
            StreamName=STREAM_NAME,
            Data=json.dumps(record),
            PartitionKey="btc-partition",
        )
        print(f"[OK] Sent to Kinesis | shard={response['ShardId']} | {record['timestamp']} | close=${record['close']}")

    except Exception as e:
        print(f"[ERROR] Failed to send to Kinesis: {e}")


def run():
    """Main loop — fetch BTC data and stream to Kinesis every 60 seconds."""
    print(f"[START] BTC producer running — streaming to '{STREAM_NAME}' every {INTERVAL_SECONDS}s")
    print("[INFO] Press Ctrl+C to stop\n")

    while True:
        tick = fetch_btc_tick()

        if tick:
            print(f"[DATA] {tick}")
            send_to_kinesis(tick)
        else:
            print("[SKIP] No data this tick")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()