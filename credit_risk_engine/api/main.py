"""
FastAPI prediction API for the Credit Risk Engine.

Loads the trained XGBoost model, StandardScaler, and feature names at startup.
Provides a /predict endpoint that returns the approval decision, probability,
and SHAP-based explanations for each prediction.

Usage:
    uvicorn credit_risk_engine.api.main:app --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("credit_risk_engine")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Global model objects (loaded at startup) ──────────────────────────────────
model = None
scaler = None
feature_names = None
explainer = None


def _load_model() -> None:
    """Load model artifacts from disk into module-level globals."""
    global model, scaler, feature_names, explainer

    model_file = MODEL_DIR / "model.pkl"
    scaler_file = MODEL_DIR / "scaler.pkl"
    names_file = MODEL_DIR / "feature_names.json"

    if not model_file.exists():
        raise RuntimeError(
            f"Model not found at {model_file}. "
            "Run Step 3 training first: python -m credit_risk_engine.src.03_train"
        )

    try:
        model = joblib.load(model_file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model: {exc}\n"
            "Hint: If you see an XGBoost / OpenMP error, install the OpenMP runtime:\n"
            "  macOS  — brew install libomp\n"
            "  Ubuntu — sudo apt-get install libgomp1\n"
        ) from exc

    scaler = joblib.load(scaler_file)
    with open(names_file) as f:
        feature_names = json.load(f)

    explainer = shap.TreeExplainer(model)
    logger.info("Model loaded: %d features", len(feature_names))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up (if needed) on shutdown."""
    _load_model()
    yield


app = FastAPI(
    title="Credit Risk Approval Engine",
    description="Explainable credit risk predictions with SHAP",
    version="1.0.0",
    lifespan=lifespan,
)


class LoanApplication(BaseModel):
    """Input schema — mirrors the feature columns from feature_utils."""
    amt_income_total: float = 200000.0
    amt_credit: float = 500000.0
    amt_annuity: float = 25000.0
    amt_goods_price: float = 450000.0
    days_birth: float = -12000.0
    days_employed: float = -2000.0
    days_registration: float = -5000.0
    days_id_publish: float = -1500.0
    ext_source_1: Optional[float] = None
    ext_source_2: Optional[float] = None
    ext_source_3: Optional[float] = None
    cnt_children: float = 0.0
    cnt_fam_members: float = 2.0
    name_contract_type: float = 0.0
    code_gender: float = 0.0
    flag_own_car: float = 0.0
    flag_own_realty: float = 0.0
    name_income_type: float = 0.0
    name_education_type: float = 0.0
    name_family_status: float = 0.0
    name_housing_type: float = 0.0
    occupation_type: float = 0.0
    organization_type: float = 0.0


class PredictionResponse(BaseModel):
    decision: str
    probability_of_default: float
    risk_score: int
    top_factors: list[dict]


def _build_input_row(application: LoanApplication) -> pd.DataFrame:
    """Convert a LoanApplication into a single-row DataFrame matching training features."""
    data = application.model_dump()

    # Add derived features (same logic as feature_utils.add_derived_features)
    data["credit_income_ratio"] = data["amt_credit"] / (data["amt_income_total"] + 1)
    data["annuity_income_ratio"] = data["amt_annuity"] / (data["amt_income_total"] + 1)
    data["credit_annuity_ratio"] = data["amt_credit"] / (data["amt_annuity"] + 1)
    data["age_years"] = (-data["days_birth"]) / 365.25

    emp = data["days_employed"] if data["days_employed"] != 365243 else np.nan
    data["employment_years"] = (-emp) / 365.25 if emp is not np.nan else np.nan

    data["goods_credit_ratio"] = data["amt_goods_price"] / (data["amt_credit"] + 1)

    ext_vals = [
        v for v in [data.get("ext_source_1"), data.get("ext_source_2"), data.get("ext_source_3")]
        if v is not None
    ]
    data["ext_source_mean"] = np.mean(ext_vals) if ext_vals else 0.0

    # Build row with only the expected feature columns
    row = {}
    for col in feature_names:
        row[col] = data.get(col, 0.0)
        if row[col] is None:
            row[col] = 0.0

    return pd.DataFrame([row], columns=feature_names)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = _build_input_row(application)
    row_scaled = pd.DataFrame(
        scaler.transform(row), columns=feature_names
    )

    # Prediction
    prob = float(model.predict_proba(row_scaled)[0, 1])
    prediction = int(prob >= 0.5)

    # Risk score (0–100, higher = riskier)
    risk_score = int(round(prob * 100))

    # Decision label
    if prob < 0.3:
        decision = "APPROVED"
    elif prob < 0.5:
        decision = "REVIEW"
    else:
        decision = "DECLINED"

    # SHAP explanation
    shap_values = explainer.shap_values(row_scaled)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # class 1 (default)
    else:
        shap_vals = shap_values[0]

    # Top contributing factors
    abs_shap = np.abs(shap_vals)
    top_idx = np.argsort(abs_shap)[::-1][:10]
    top_factors = []
    for i in top_idx:
        top_factors.append({
            "feature": feature_names[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "feature_value": round(float(row.iloc[0, i]), 4),
            "direction": "increases risk" if shap_vals[i] > 0 else "decreases risk",
        })

    return PredictionResponse(
        decision=decision,
        probability_of_default=round(prob, 4),
        risk_score=risk_score,
        top_factors=top_factors,
    )
