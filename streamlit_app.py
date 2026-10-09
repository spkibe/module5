from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
FIG = APP_DIR / "reports" / "figures"
MODEL_PATH = APP_DIR / "models" / "churn_model_calibrated.joblib"

st.set_page_config(page_title="Netflix Churn Insights", layout="wide")
st.title("Netflix Customer Churn Insights")
st.caption("BAN6800 Module 5 · Stakeholder Dashboard")

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)

@st.cache_data
def load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model_comparison = load_csv("model_comparison.csv")
fairness = load_csv("fairness_metrics.csv")
ablation = load_csv("feature_ablation_results.csv")
watch_time = load_csv("watch_time_treatment_sensitivity.csv")
calibration = load_csv("probability_calibration_results.csv")
thresholds = load_csv("sensitivity_thresholds.csv")
readiness = load_json("data_readiness_summary.json")
metadata = load_json("best_model_metadata.json")

model = load_model()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Prediction Explorer",
    "What-If Analysis",
    "Explainability",
    "Fairness & Transparency",
])

with tab1:
    st.subheader("Model Performance Overview")
    calibrated = calibration[calibration["model"] == "sigmoid_calibrated"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{calibrated['accuracy']:.1%}")
    c2.metric("Recall", f"{calibrated['recall']:.1%}")
    c3.metric("ROC-AUC", f"{calibrated['roc_auc']:.4f}")
    c4.metric("Brier Score", f"{calibrated['brier_score']:.4f}")

    st.write("The calibrated Gradient Boosting model is the dashboard candidate because it improved probability quality while preserving strong classification performance.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(str(FIG / "roc_curve.png"), caption="ROC curve comparison")
    with col2:
        fig = px.bar(
            model_comparison,
            x="model",
            y="roc_auc",
            text="roc_auc",
            title="Model Comparison by ROC-AUC",
            range_y=[0, 1.05],
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Readiness")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Input rows", f"{readiness['input_rows']:,}")
    d2.metric("Duplicates removed", readiness["duplicates_removed"])
    d3.metric("Missing values", sum(readiness["missing_values"].values()))
    d4.metric(">24h watch-time flags", readiness["avg_watch_over_24h"])

with tab2:
    st.subheader("Prediction Explorer")
    if model is None:
        st.warning("Model artifact not found. Add models/churn_model_calibrated.joblib before deployment to enable live predictions.")
    else:
        col1, col2, col3 = st.columns(3)
        watch_hours = col1.number_input("Watch hours", min_value=0.0, value=50.0, step=1.0)
        last_login_days = col1.number_input("Days since last login", min_value=0.0, value=12.0, step=1.0)
        monthly_fee = col1.number_input("Monthly fee", min_value=0.0, value=14.99, step=0.5)
        number_of_profiles = col2.number_input("Number of profiles", min_value=1.0, value=2.0, step=1.0)
        avg_watch_time_per_day = col2.number_input("Average watch time per day", min_value=0.0, value=1.8, step=0.1)
        subscription_type = col2.selectbox("Subscription type", ["Basic", "Standard", "Premium"])
        region = col3.selectbox("Region", ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"])
        device = col3.selectbox("Device", ["TV", "Mobile", "Tablet", "Laptop"])
        payment_method = col3.selectbox("Payment method", ["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"])
        favorite_genre = st.selectbox("Favorite genre", ["Drama", "Action", "Comedy", "Sci-Fi", "Documentary", "Romance", "Thriller", "Horror"])

        row = pd.DataFrame([{
            "watch_hours": watch_hours,
            "last_login_days": last_login_days,
            "monthly_fee": monthly_fee,
            "number_of_profiles": number_of_profiles,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "subscription_type": subscription_type,
            "region": region,
            "device": device,
            "payment_method": payment_method,
            "favorite_genre": favorite_genre,
        }])
        prob = float(model.predict_proba(row)[0, 1])
        pred = int(prob >= 0.50)
        st.metric("Predicted churn probability", f"{prob:.1%}")
        st.write("Prediction label:", "**Churned**" if pred else "**Retained**")
        st.caption("This is decision-support information from an academic model, not an automatic customer decision.")

with tab3:
    st.subheader("What-If Analysis")
    st.write("Move the threshold to see the precision/recall trade-off. This is for exploration only, not a formal recommendation.")
    threshold = st.slider("Decision threshold", 0.30, 0.70, 0.50, 0.05)
    selected = thresholds.iloc[(thresholds["threshold"] - threshold).abs().argsort()[:1]].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{selected['accuracy']:.1%}")
    c2.metric("Precision", f"{selected['precision']:.1%}")
    c3.metric("Recall", f"{selected['recall']:.1%}")
    c4.metric("F1", f"{selected['f1']:.1%}")
    fig = px.line(thresholds, x="threshold", y=["precision", "recall", "f1"], markers=True, title="Threshold Trade-Off")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Robustness Checks")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Feature ablation")
        st.dataframe(ablation[["scenario", "roc_auc", "roc_auc_drop_vs_all_features"]], hide_index=True)
    with col2:
        st.write("Watch-time anomaly sensitivity")
        st.dataframe(watch_time[["scenario", "accuracy", "recall", "roc_auc"]], hide_index=True)

with tab4:
    st.subheader("Explainability")
    st.write("The model mainly relies on understandable engagement and account signals.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIG / "shap_summary.png"), caption="Global SHAP summary")
    with col2:
        st.image(str(FIG / "shap_local.png"), caption="Local SHAP example")
    st.markdown("""
    **Plain language:** customers with more recent usage and stronger viewing activity are generally pushed toward retention, while inactivity and some account/payment patterns can push predictions toward churn.
    """)

with tab5:
    st.subheader("Fairness & Transparency")
    st.write("Age and gender were excluded from training and used only for auditing.")
    gender = fairness[fairness["protected_attribute"] == "gender"].copy()
    age = fairness[fairness["protected_attribute"] == "age_group"].copy()

    gdp = gender["demographic_parity_difference_overall"].iloc[0]
    geo = gender["equalized_odds_difference_overall"].iloc[0]
    adp = age["demographic_parity_difference_overall"].iloc[0]
    aeo = age["equalized_odds_difference_overall"].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gender DP diff", f"{gdp:.3f}")
    m2.metric("Gender EO diff", f"{geo:.3f}")
    m3.metric("Age DP diff", f"{adp:.3f}")
    m4.metric("Age EO diff", f"{aeo:.3f}")

    st.write("Gender group metrics")
    st.dataframe(gender[["gender", "accuracy", "recall", "selection_rate"]], hide_index=True)
    st.write("Age group metrics")
    st.dataframe(age[["age_group", "accuracy", "recall", "selection_rate"]], hide_index=True)

    st.subheader("Transparency Statement")
    st.info(
        "This is an academic churn-status classifier built on a public dataset. It identifies statistical patterns associated with the provided churn label. It does not explain why a customer churned, does not predict a guaranteed future churn date, and should not be used as an automatic decision system."
    )
