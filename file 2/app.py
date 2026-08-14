import streamlit as st
import pandas as pd
import backend

# Set up the webpage
st.set_page_config(page_title="Finsight AI Predictor", layout="wide", page_icon="📈")

st.title("📈 Finsight AI: Financial Crisis Predictor")
st.markdown("**Empirically driven by XGBoost and HP-Filtered 150-Year Macroeconomic Data**")
st.markdown("---")

# Load the AI Brain
model, scaler = backend.load_model_and_scaler()

if model is None:
    st.error("🚨 Model files not found! Please run your '03_Algorithm_Training.ipynb' notebook first to generate the .pkl files in the 'models/' folder.")
    st.stop()

# Display Live Market Data
telemetry = backend.fetch_live_telemetry()
col1, col2 = st.columns(2)
col1.metric("Live Market Volatility (VIX)", telemetry.get('VIX', 'N/A'))
col2.metric("US 10-Year Treasury Yield", f"{telemetry.get('US_10Y', 'N/A')}%")
st.markdown("---")

# Sidebar for User Inputs
st.sidebar.header("🎛️ Macroeconomic Controls")
st.sidebar.markdown("Adjust the sliders to simulate economic conditions.")

yield_curve = st.sidebar.slider("Yield Curve Slope", -5.0, 5.0, 0.5)
credit_gdp = st.sidebar.slider("Credit to GDP Ratio", 0.0, 2.0, 0.8)
credit_diff = st.sidebar.slider("Credit Acceleration (Diff2)", -0.5, 0.5, 0.01)
credit_cycle = st.sidebar.slider("Credit Cycle (HP Filter)", -0.2, 0.2, 0.05)
yield_cycle = st.sidebar.slider("Yield Cycle (HP Filter)", -2.0, 2.0, -0.1)
cpi = st.sidebar.slider("CPI (Inflation %)", -5.0, 30.0, 2.5)
unemp = st.sidebar.slider("Unemployment Rate %", 0.0, 30.0, 5.0)
debtgdp = st.sidebar.slider("Debt to GDP Ratio", 0.0, 3.0, 1.0)

# Package the inputs into a dataframe that matches our training data
input_data = pd.DataFrame([[
    yield_curve, credit_gdp, credit_diff, credit_cycle, yield_cycle, cpi, unemp, debtgdp
]], columns=[
    'yield_curve_slope', 'credit_gdp', 'credit_gdp_diff2', 'credit_gdp_cycle', 
    'yield_curve_cycle', 'cpi', 'unemp', 'debtgdp'
])

# Get the AI Prediction
st.subheader("Systemic Crisis Risk Assessment")
risk_score = backend.predict_risk(model, scaler, input_data)

# Display a dynamic warning bar based on the risk percentage
st.progress(int(risk_score))

if risk_score > 75:
    st.error(f"🔴 **CRITICAL RISK:** {risk_score:.2f}% probability of an impending systemic crisis.")
elif risk_score > 40:
    st.warning(f"🟡 **ELEVATED RISK:** {risk_score:.2f}% probability. The economy is showing bubble characteristics.")
else:
    st.success(f"🟢 **STABLE:** {risk_score:.2f}% probability. Normal economic conditions.")