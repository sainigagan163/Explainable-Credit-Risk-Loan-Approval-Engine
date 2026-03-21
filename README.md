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
  CSV Data --> SQLite --> Feature Engineering --> SMOTE + XGBoost --> Model Artifacts
```

## Features

- **XGBoost Classifier** with SMOTE for class imbalance handling
- **SHAP Explainability** — waterfall and force plots for every prediction
- **FastAPI REST API** with Pydantic validation and health checks
- **Streamlit Dashboard** — Loan Officer Portal with approve/review decisions
- **SQLite Pipeline** — simulates enterprise data warehouse workflow
- **Docker** — single container runs both API and UI via supervisord
- **Feature Engineering** from Home Credit Default Risk dataset (bureau data, previous applications)

## Project Structure

```
credit_risk_engine/
├── data/                  # Home Credit CSV files (not tracked in git)
├── database/              # SQLite database (generated)
├── models/                # Saved model artifacts (generated)
├── src/
│   ├── 01_ingest.py       # CSV to SQLite ingestion
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

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Feature Scaling | StandardScaler (scikit-learn) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | SQLite + SQLAlchemy |
| Containerisation | Docker + supervisord |

## License

MIT
