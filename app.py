import streamlit as st
import pandas as pd
import numpy as np
import requests 

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.model_training import ModelTrainer
from src.evaluation_system import EvaluationSystem
from src.risk_analysis import RiskAnalyzer
from src.feature_analysis import FeatureAnalyzer
from src.stats_analysis import StatisticalAnalyzer



# Call API 
if st.button("Predict"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    st.write(response.json())
st.set_page_config(layout="wide")

st.title("📦 Supply Chain Optimization & Risk Forecasting Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)")

model_choice = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Linear", "Ridge", "Lasso", "ARIMA", "SARIMA", "Prophet"]
)

test_choice = st.sidebar.selectbox(
    "Select Statistical Test",
    ["t_test", "anova", "chi_square", "granger"]
)

scenario_multiplier = st.sidebar.slider(
    "Risk Scenario Multiplier",
    0.5, 2.0, 1.0
)

# -----------------------------
# LOAD DATA
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# -----------------------------
# PIPELINE
# -----------------------------
df = preprocess_data(df)
df = create_features(df)

# -----------------------------
# TARGET SELECTION
# -----------------------------
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# MODEL TRAINING
# -----------------------------
trainer = ModelTrainer()

if model_choice in ["Linear", "Ridge", "Lasso"]:
    model, _ = trainer.train_regression_models(X_train, X_test, y_train, y_test)

elif model_choice == "ARIMA":
    model, forecast = trainer.train_arima(df[target])
    st.line_chart(forecast)

elif model_choice == "SARIMA":
    model, forecast = trainer.train_sarima(df[target])
    st.line_chart(forecast)

elif model_choice == "Prophet":
    if "date" not in df.columns:
        st.error("Prophet requires a 'date' column")
    else:
        model, forecast = trainer.train_prophet(df)
        st.line_chart(forecast["yhat"])

# -----------------------------
# MODEL EVALUATION
# -----------------------------
if model_choice in ["Linear", "Ridge", "Lasso"]:

    evaluator = EvaluationSystem()
    metrics = evaluator.evaluate_regression(model, X_test, y_test)

    st.subheader("📊 Model Performance")
    st.write(metrics)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
fa = FeatureAnalyzer()

mi_df = fa.mutual_information(X, y)

st.subheader("🔬 Feature Importance (Top 10)")
st.dataframe(mi_df.head(10))

# -----------------------------
# RISK ANALYSIS
# -----------------------------
risk = RiskAnalyzer(df)

df = risk.calculate_risk_score()
df = risk.categorize_risk()
df = risk.calculate_financial_risk()

# Apply scenario multiplier
df["adjusted_risk"] = df["financial_risk"] * scenario_multiplier

st.subheader("⚠️ Risk Distribution")
st.bar_chart(df["risk_level"].value_counts())

st.subheader("💰 Top Risk Nodes")
st.dataframe(df.sort_values(by="adjusted_risk", ascending=False).head(10))

# -----------------------------
# STATISTICAL TESTING
# -----------------------------
stats = StatisticalAnalyzer(df)

st.subheader("📈 Statistical Test Results")

try:
    if test_choice == "t_test":
        result = stats.run_test(
            "t_test",
            col1="supplier_rating",
            col2=target,
            threshold=4
        )

    elif test_choice == "anova":
        result = stats.run_test(
            "anova",
            category_col="product_category",
            value_col=target
        )

    elif test_choice == "chi_square":
        result = stats.run_test(
            "chi_square",
            col1="region",
            col2="risk_level"
        )

    elif test_choice == "granger":
        result = stats.run_test(
            "granger",
            col1=target,
            col2="inventory_level"
        )

    st.write(result)

except Exception as e:
    st.error(f"Test failed: {e}")

# -----------------------------
# DOWNLOAD REPORTS
# -----------------------------
st.subheader("📥 Download Processed Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Results CSV",
    data=csv,
    file_name="processed_results.csv",
    mime="text/csv"
)