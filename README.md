# Explainable Credit Risk & Loan Approval Engine

An end-to-end machine learning system for credit risk assessment with full explainability, built for a UK finance data science portfolio.

## Architecture

```
                                  +------------------+
                                  |   Streamlit UI   |
                                  |   (Port 8501)    |
                                  +--------+---------+
                                           |
                                    POST /predict
                                           |
                                  +--------v---------+
                                  |   FastAPI API    |
                                  |   (Port 8000)    |
                                  +--------+---------+
                                           |
                          +----------------+----------------+
                          |                |                |
                   +------v------+  +------v------+  +-----v-------+
                   |  XGBoost    |  |   Scaler    |  |    SHAP     |
                   |   Model     |  | (Standard)  |  | Explainer   |
                   +-------------+  +-------------+  +-------------+

Offline Training Pipeline:
  CSV Data --> Feature Engineering --> SMOTE + XGBoost --> Model Artifacts
```

## Features

- **XGBoost Classifier** with SMOTE for class imbalance handling
- **SHAP Explainability** — waterfall and force plots for every prediction
- **FastAPI REST API** with Pydantic validation and health checks
- **Streamlit Dashboard** — Loan Officer Portal with approve/review decisions
- **Docker** — single container runs both API and UI via supervisord
- **Feature Engineering** from Home Credit Default Risk dataset (bureau data, previous applications)

## Project Structure

```
credit_risk_engine/
├── data/                  # Home Credit CSV files (not tracked in git)
├── models/                # Saved model artifacts (generated)
├── src/
│   ├── 01_ingest.py       # CSV data validation
│   ├── 02_features.py     # Feature engineering pipeline
│   ├── 03_train.py        # XGBoost training with SMOTE
│   └── feature_utils.py   # Shared feature transformation logic
├── api/
│   └── main.py            # FastAPI prediction API
├── frontend/
│   └── app.py             # Streamlit Loan Officer Portal
├── supervisord.conf       # Process manager config
└── start.sh               # Docker entrypoint
```

## Data

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle. Download the CSV files and place them in `credit_risk_engine/data/`.

Required: `application_train.csv`

Optional (enriches features): `bureau.csv`, `previous_application.csv`, `credit_card_balance.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`

## Running Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle and place the CSV files in `credit_risk_engine/data/`.

### 3. Validate Data

```bash
python -m credit_risk_engine.src.01_ingest --data-dir credit_risk_engine/data
```

### 4. Build Feature Matrix

```bash
python -m credit_risk_engine.src.02_features --data-dir credit_risk_engine/data
```

This reads the raw CSVs, engineers features (derived ratios, bureau aggregations, previous application stats), and saves `features.csv`.

### 5. Train the Model

```bash
python -m credit_risk_engine.src.03_train --data-dir credit_risk_engine/data --model-dir credit_risk_engine/models
```

Trains XGBoost with SMOTE oversampling, prints evaluation metrics (ROC AUC, classification report), and saves `model.pkl`, `scaler.pkl`, and `feature_names.json` to `credit_risk_engine/models/`.

### 6. Start the API

```bash
uvicorn credit_risk_engine.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Test with `http://localhost:8000/health`.

### 7. Start the Streamlit Dashboard

Open a second terminal and run:

```bash
streamlit run credit_risk_engine/frontend/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser to use the Loan Officer Portal.

## Running with Docker

```bash
docker build -t credit-risk-engine .
docker run -p 8000:8000 -p 8501:8501 credit-risk-engine
```

This starts both the FastAPI API (port 8000) and Streamlit UI (port 8501) in a single container via supervisord.

> **Note:** The Docker image expects trained model artifacts in `credit_risk_engine/models/`. Run Steps 3–5 locally before building the image, or mount a volume with the model files.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Feature Scaling | StandardScaler (scikit-learn) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Storage | Local CSV files |
| Containerisation | Docker + supervisord |

## License

MIT
