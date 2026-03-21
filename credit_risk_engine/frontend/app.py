"""
Streamlit Loan Officer Portal for the Credit Risk Engine.

Provides a form-based UI for entering applicant details, calls the FastAPI
/predict endpoint, and displays the decision alongside SHAP explanations.

Usage:
    streamlit run credit_risk_engine/frontend/app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Credit Risk Engine", page_icon="🏦", layout="wide")
st.title("Credit Risk & Loan Approval Engine")
st.caption("Explainable AI-powered credit risk assessment")

# ── Sidebar: Applicant Input Form ─────────────────────────────────────────────
st.sidebar.header("Applicant Details")

with st.sidebar.form("applicant_form"):
    st.subheader("Income & Loan")
    amt_income_total = st.number_input("Annual Income", value=200000.0, step=10000.0)
    amt_credit = st.number_input("Loan Amount", value=500000.0, step=10000.0)
    amt_annuity = st.number_input("Annuity Amount", value=25000.0, step=1000.0)
    amt_goods_price = st.number_input("Goods Price", value=450000.0, step=10000.0)

    st.subheader("Personal")
    age_years = st.slider("Age (years)", 18, 80, 35)
    days_birth = -age_years * 365.25
    employment_years = st.slider("Employment (years)", 0, 50, 5)
    days_employed = -employment_years * 365.25
    cnt_children = st.number_input("Number of Children", value=0, step=1)
    cnt_fam_members = st.number_input("Family Members", value=2, step=1)

    st.subheader("Contract & Property")
    name_contract_type = st.selectbox("Contract Type", [0, 1], format_func=lambda x: ["Cash loans", "Revolving loans"][x])
    code_gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    flag_own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    flag_own_realty = st.selectbox("Owns Property", [0, 1], format_func=lambda x: ["No", "Yes"][x])

    st.subheader("External Scores")
    ext_source_1 = st.slider("External Score 1", 0.0, 1.0, 0.5)
    ext_source_2 = st.slider("External Score 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("External Score 3", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("Assess Risk", type="primary")

# ── Main panel ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "amt_income_total": amt_income_total,
        "amt_credit": amt_credit,
        "amt_annuity": amt_annuity,
        "amt_goods_price": amt_goods_price,
        "days_birth": days_birth,
        "days_employed": days_employed,
        "days_registration": -5000.0,
        "days_id_publish": -1500.0,
        "ext_source_1": ext_source_1,
        "ext_source_2": ext_source_2,
        "ext_source_3": ext_source_3,
        "cnt_children": float(cnt_children),
        "cnt_fam_members": float(cnt_fam_members),
        "name_contract_type": float(name_contract_type),
        "code_gender": float(code_gender),
        "flag_own_car": float(flag_own_car),
        "flag_own_realty": float(flag_own_realty),
        "name_income_type": 0.0,
        "name_education_type": 0.0,
        "name_family_status": 0.0,
        "name_housing_type": 0.0,
        "occupation_type": 0.0,
        "organization_type": 0.0,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Ensure the FastAPI server is running on port 8000.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()

    # ── Decision banner ──────────────────────────────────────────────────────
    decision = result["decision"]
    prob = result["probability_of_default"]
    risk_score = result["risk_score"]

    col1, col2, col3 = st.columns(3)
    with col1:
        colour = {"APPROVED": "green", "REVIEW": "orange", "DECLINED": "red"}[decision]
        st.markdown(
            f"### Decision: :{colour}[{decision}]"
        )
    with col2:
        st.metric("Default Probability", f"{prob:.1%}")
    with col3:
        st.metric("Risk Score", f"{risk_score}/100")

    st.divider()

    # ── SHAP explanation ─────────────────────────────────────────────────────
    st.subheader("Explanation — Top Risk Factors")

    factors = result["top_factors"]
    for factor in factors:
        direction_icon = "🔴" if factor["direction"] == "increases risk" else "🟢"
        shap_val = factor["shap_value"]
        bar_width = min(abs(shap_val) * 200, 100)
        bar_colour = "#e74c3c" if shap_val > 0 else "#27ae60"

        col_name, col_bar, col_val = st.columns([2, 4, 1])
        with col_name:
            st.write(f"{direction_icon} **{factor['feature']}**")
        with col_bar:
            st.markdown(
                f'<div style="background:{bar_colour};width:{bar_width}%;height:20px;'
                f'border-radius:4px;"></div>',
                unsafe_allow_html=True,
            )
        with col_val:
            st.write(f"`{factor['feature_value']}`")

    st.divider()
    st.caption(
        "Predictions are powered by an XGBoost model with SHAP (SHapley Additive exPlanations). "
        "Green factors decrease default risk; red factors increase it."
    )
else:
    st.info("Fill in applicant details in the sidebar and click **Assess Risk** to get a prediction.")
