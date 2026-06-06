# Real-Time Bitcoin Market Analytics Pipeline 🚀

A fully serverless, end-to-end data pipeline that ingests live Bitcoin market data, processes and stores it in real-time, detects price anomalies using statistical and ML methods, and automatically retrains its model weekly — all deployed on AWS with a full CI/CD pipeline.

---

## Architecture

```
yfinance (local producer)
        ↓
   SQS Queue
        ↓
Lambda #1 — Processor
  ├── Clean & validate data
  ├── Compute rolling z-score (window=20)
  ├── Detect anomalies (|z| > 2.5)
  ├── Store cleaned data → DynamoDB
  └── Archive raw data → S3
        ↓
  [anomaly detected?]
        ↓
   SNS → Email Alert
        ↓
S3 (raw archive) ← Athena SQL queries
        ↓
Lambda #2 — ML Retrain (every Sunday)
  ├── Load historical data from S3
  ├── Train Isolation Forest
  └── Save model artifacts → S3
        ↓
CodePipeline + CodeBuild (CI/CD on every push)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Amazon SQS |
| Stream processing | AWS Lambda (Python 3.11) |
| Cleaned data store | Amazon DynamoDB |
| Raw data archive | Amazon S3 |
| Historical queries | Amazon Athena + AWS Glue |
| Anomaly detection | Z-score + Isolation Forest (scikit-learn) |
| Alerting | Amazon SNS |
| Scheduling | Amazon EventBridge |
| CI/CD | AWS CodePipeline + CodeBuild |
| Infrastructure as Code | AWS CDK (Python) |
| Data source | yfinance (Yahoo Finance API) |

---

## Project Structure

```
btc-pipeline/
├── infrastructure/        # CDK stack — all AWS resources defined as Python
├── producer/              # Local script — streams BTC ticks to SQS
├── lambdas/
│   ├── processor/         # Lambda #1 — clean, enrich, store, alert
│   └── ml_retrain/        # Lambda #2 — weekly Isolation Forest retrain
├── ml/                    # Local training experiments
├── athena_queries/        # SQL queries for historical analysis
├── tests/                 # 33 pytest tests, all AWS calls mocked
└── buildspec.yml          # CodeBuild CI/CD config
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js (for AWS CDK)
- AWS CLI configured with your credentials
- AWS account with free tier

### Installation

```bash
# Clone the repo
git clone https://github.com/Amir-Tav/btc-pipeline.git
cd btc-pipeline

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install CDK
npm install -g aws-cdk
```

### Configure AWS

```bash
aws configure
# Enter your Access Key ID, Secret Access Key, region (eu-west-2), output format (json)
```

### Deploy Infrastructure

```bash
cd infrastructure
cdk bootstrap aws://YOUR_ACCOUNT_ID/eu-west-2
cdk deploy
```

### Run the Producer

```bash
python producer/btc_producer.py
```

You'll see live BTC ticks streaming every 60 seconds:

```
[START] BTC producer running — streaming to SQS every 60s
[DATA] {'symbol': 'BTC-USD', 'timestamp': '2026-06-06T12:22:08+00:00', 'close': 60896.0, ...}
[OK] Sent to SQS | MessageId=4f104920... | close=$60896.0
```

### Run Tests

```bash
pytest tests/ -v
# 33 passed in 1.65s
```

---

## How It Works

### Anomaly Detection

Every incoming BTC tick is scored against a rolling window of the last 20 close prices using a **z-score**:

```python
z = (price - mean) / stdev
```

If `|z| > 2.5`, the tick is flagged as anomalous and an SNS email alert is fired immediately.

### ML Retraining

Every Sunday at midnight, Lambda #2 automatically:
1. Loads all raw BTC records from S3
2. Extracts 6 features: `close`, `volume`, `price_range`, `open`, `high`, `low`
3. Scales features with `StandardScaler`
4. Retrains an `IsolationForest` (100 estimators, 5% contamination)
5. Saves the updated model and scaler back to S3

### CI/CD

Every push to `main` triggers CodePipeline:
1. **Source** — pulls latest code from GitHub
2. **Build & Test** — CodeBuild runs all 33 pytest tests
3. **Deploy** — if tests pass, CDK deploys any infrastructure changes automatically

---

## Querying Historical Data

Once the Glue crawler has run, you can query all historical BTC data directly from S3 using Athena:

```sql
-- Daily price summary
SELECT
    date,
    MIN(CAST(low AS DOUBLE))    AS day_low,
    MAX(CAST(high AS DOUBLE))   AS day_high,
    AVG(CAST(close AS DOUBLE))  AS avg_close,
    COUNT(*)                    AS tick_count
FROM raw
GROUP BY date
ORDER BY date DESC;

-- All detected anomalies
SELECT timestamp, close, zscore
FROM raw
WHERE is_anomaly = true
ORDER BY ABS(CAST(zscore AS DOUBLE)) DESC;
```

---

## Infrastructure Overview

All AWS resources are defined as Python code using CDK (`infrastructure/pipeline_stack.py`) — no manual console setup required. A single `cdk deploy` creates everything:

- SQS queue with 1-day retention
- Two Lambda functions with least-privilege IAM roles
- DynamoDB table (pay-per-request, no capacity planning)
- S3 bucket with auto-delete on stack teardown
- SNS topic with email subscription
- EventBridge rule for weekly ML retrain
- CodePipeline with GitHub source and CodeBuild stage

---

## Key Design Decisions

- **SQS over Kinesis** — SQS is free tier eligible and sufficient for single-asset tick data at 1 message/minute
- **Z-score first, ML second** — statistical detection works immediately with no training data; Isolation Forest improves over time as data accumulates
- **Lambda for inference** — model artifacts are tiny (~500KB), so inference runs inside Lambda with no SageMaker costs
- **Infrastructure as Code** — entire stack is reproducible with one command, no manual steps

