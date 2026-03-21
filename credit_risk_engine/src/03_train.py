"""
Step 3 of the pipeline: Train an XGBoost classifier with SMOTE for class
imbalance handling, save model artifacts (model, scaler, feature names),
and print evaluation metrics.

Usage:
    python -m credit_risk_engine.src.03_train
    python -m credit_risk_engine.src.03_train --data-dir credit_risk_engine/data
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent.parent / "models")


def main(data_dir: str = DEFAULT_DATA_DIR, model_dir: str = DEFAULT_MODEL_DIR) -> None:
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix from Step 2 ─────────────────────────────────────
    features_file = data_path / "features.csv"
    if not features_file.exists():
        sys.exit(
            f"Feature matrix not found: {features_file}\n"
            "Run Step 2 first: python -m credit_risk_engine.src.02_features"
        )

    df = pd.read_csv(features_file, index_col="sk_id_curr")
    print(f"  Loaded features: {df.shape}")

    if "target" not in df.columns:
        sys.exit("No 'target' column found in features.csv — cannot train.")

    # ── Separate features and target ─────────────────────────────────────────
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    feature_names = list(X.columns)
    print(f"  Features: {len(feature_names)}")
    print(f"  Target distribution: {dict(y.value_counts())}")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── Scale features ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    # ── SMOTE for class imbalance ────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After SMOTE: {X_train_res.shape[0]:,} samples")
    print(f"    Class 0: {(y_train_res == 0).sum():,}  |  Class 1: {(y_train_res == 1).sum():,}")

    # ── Train XGBoost ────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False,
    )

    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  ROC AUC: {auc:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # ── Feature importance (top 15) ──────────────────────────────────────────
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("  Top 15 features:")
    for i in top_idx:
        print(f"    {feature_names[i]:30s}  {importances[i]:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(model, model_path / "model.pkl")
    joblib.dump(scaler, model_path / "scaler.pkl")
    with open(model_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n  Saved model artifacts to {model_path}/")
    print(f"    model.pkl  |  scaler.pkl  |  feature_names.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost credit risk model")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Path to feature CSV directory"
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help="Path to save model artifacts"
    )
    args = parser.parse_args()
    main(args.data_dir, args.model_dir)
