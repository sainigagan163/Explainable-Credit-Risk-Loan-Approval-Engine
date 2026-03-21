"""
Shared feature transformation utilities used by both the training pipeline
and the prediction API to ensure consistent feature engineering.
"""

import numpy as np
import pandas as pd


# ── Column groups ────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
    "amt_goods_price",
    "days_birth",
    "days_employed",
    "days_registration",
    "days_id_publish",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "cnt_children",
    "cnt_fam_members",
]

CATEGORICAL_COLS = [
    "name_contract_type",
    "code_gender",
    "flag_own_car",
    "flag_own_realty",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "organization_type",
]


# ── Derived features ────────────────────────────────────────────────────────


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hand-crafted features from the application table."""
    df = df.copy()

    # Credit-to-income ratio
    df["credit_income_ratio"] = df["amt_credit"] / (df["amt_income_total"] + 1)

    # Annuity-to-income ratio
    df["annuity_income_ratio"] = df["amt_annuity"] / (df["amt_income_total"] + 1)

    # Credit-to-annuity ratio (loan term proxy)
    df["credit_annuity_ratio"] = df["amt_credit"] / (df["amt_annuity"] + 1)

    # Age in years (days_birth is negative)
    df["age_years"] = (-df["days_birth"]) / 365.25

    # Employment years (days_employed is negative; 365243 = unemployed sentinel)
    emp = df["days_employed"].replace(365243, np.nan)
    df["employment_years"] = (-emp) / 365.25

    # Goods-to-credit ratio (down-payment proxy)
    df["goods_credit_ratio"] = df["amt_goods_price"] / (df["amt_credit"] + 1)

    # External source mean
    ext_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
    df["ext_source_mean"] = df[ext_cols].mean(axis=1)

    return df


# ── Bureau aggregations ──────────────────────────────────────────────────────


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau table to one row per sk_id_curr."""
    agg = bureau.groupby("sk_id_curr").agg(
        bureau_loan_count=("sk_id_bureau", "count"),
        bureau_active_count=(
            "credit_active",
            lambda x: (x == "Active").sum(),
        ),
        bureau_credit_sum=("amt_credit_sum", "sum"),
        bureau_credit_mean=("amt_credit_sum", "mean"),
        bureau_debt_sum=("amt_credit_sum_debt", "sum"),
        bureau_overdue_sum=("amt_credit_sum_overdue", "sum"),
        bureau_days_credit_mean=("days_credit", "mean"),
    )
    return agg.reset_index()


# ── Previous application aggregations ────────────────────────────────────────


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application table to one row per sk_id_curr."""
    agg = prev.groupby("sk_id_curr").agg(
        prev_app_count=("sk_id_prev", "count"),
        prev_approved_count=(
            "name_contract_status",
            lambda x: (x == "Approved").sum(),
        ),
        prev_refused_count=(
            "name_contract_status",
            lambda x: (x == "Refused").sum(),
        ),
        prev_amt_credit_mean=("amt_credit", "mean"),
        prev_amt_annuity_mean=("amt_annuity", "mean"),
    )
    agg["prev_approval_rate"] = agg["prev_approved_count"] / (
        agg["prev_app_count"] + 1
    )
    return agg.reset_index()


# ── Master feature builder ───────────────────────────────────────────────────


def build_feature_matrix(
    app: pd.DataFrame,
    bureau: pd.DataFrame | None = None,
    prev: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw tables.
    Returns a DataFrame with sk_id_curr as index, target column (if present),
    and all engineered features.
    """
    df = app.copy()

    # Lowercase column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Derived features from application table
    df = add_derived_features(df)

    # Merge bureau aggregations
    if bureau is not None:
        bureau.columns = [c.lower().strip() for c in bureau.columns]
        bureau_agg = aggregate_bureau(bureau)
        df = df.merge(bureau_agg, on="sk_id_curr", how="left")

    # Merge previous application aggregations
    if prev is not None:
        prev.columns = [c.lower().strip() for c in prev.columns]
        prev_agg = aggregate_previous(prev)
        df = df.merge(prev_agg, on="sk_id_curr", how="left")

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Select final feature columns
    feature_cols = (
        NUMERIC_COLS
        + CATEGORICAL_COLS
        + [
            "credit_income_ratio",
            "annuity_income_ratio",
            "credit_annuity_ratio",
            "age_years",
            "employment_years",
            "goods_credit_ratio",
            "ext_source_mean",
        ]
    )

    # Add bureau features if available
    bureau_cols = [c for c in df.columns if c.startswith("bureau_")]
    feature_cols += bureau_cols

    # Add previous app features if available
    prev_cols = [c for c in df.columns if c.startswith("prev_")]
    feature_cols += prev_cols

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    result = df[["sk_id_curr"] + feature_cols].copy()

    if "target" in df.columns:
        result["target"] = df["target"]

    result = result.set_index("sk_id_curr")
    return result
