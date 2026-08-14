import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_live_telemetry():
    """Fetches real-time market indicators from Yahoo Finance."""
    live_data = {}
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        live_data['VIX'] = round(vix['Close'].iloc[-1], 2) if not vix.empty else 18.40
        
        tnx = yf.Ticker("^TNX").history(period="1d")
        live_data['US_10Y'] = round(tnx['Close'].iloc[-1], 3) if not tnx.empty else 4.120
    except Exception:
        live_data = {'VIX': 18.40, 'US_10Y': 4.120}
    return live_data

@st.cache_resource
def load_model_and_scaler():
    """Loads the champion XGBoost model and RobustScaler from the models folder."""
    model_path = "models/crisis_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, scaler

def predict_risk(model, scaler, input_df):
    """Calculates continuous risk probability (0% - 100%)."""
    if model is None:
        return 0.0
        
    # Scale the input data just like we did during training
    X_scaled = scaler.transform(input_df.values) if scaler else input_df.values
        
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        prob = model.predict(X_scaled)[0]
        
    return float(np.clip(prob, 0.0, 1.0) * 100)