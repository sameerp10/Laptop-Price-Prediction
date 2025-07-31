import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os

# Get current and parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define paths
xgb_model_path = os.path.join(parent_dir, "xgboost_log_model.pkl")
lgbm_model_path = os.path.join(parent_dir, "lightgbm_log_model.pkl")
scaler_path = os.path.join(parent_dir, "scaler.pkl")
label_encoders_path = os.path.join(parent_dir, "label_encoders.pkl")
weights_path = os.path.join(parent_dir, "ensemble_weights.json")

# Load files
try:
    with open(xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)
    with open(lgbm_model_path, "rb") as f:
        lgbm_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(label_encoders_path, "rb") as f:
        label_encoders = pickle.load(f)
    with open(weights_path, "r") as f:
        weights = json.load(f)
except FileNotFoundError as e:
    st.error(f"❌ Error loading model/preprocessing files: {e}")
    st.stop()

# UI Inputs
company = st.selectbox("Company", options=label_encoders["Company"].classes_)
typename = st.selectbox("Type", options=label_encoders["TypeName"].classes_)
ram = st.selectbox("RAM (GB)", options=[4, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (in kg)", value=2.0)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])
screen_size = st.number_input("Screen Size (in inches)", value=15.6)
resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800"])
cpu = st.selectbox("CPU Type", options=label_encoders["Cpu_type"].classes_)
hdd = st.selectbox("HDD (in GB)", options=[0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (in GB)", options=[0, 128, 256, 512, 1024])
gpu = st.selectbox("GPU Type", options=label_encoders["Gpu_type"].classes_)
os_type = st.selectbox("Operating System", options=label_encoders["OS"].classes_)

# Feature Engineering
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

# Prepare input
query = pd.DataFrame({
    "Company": [company],
    "TypeName": [typename],
    "Ram": [ram],
    "Weight": [weight],
    "Touchscreen": [1 if touchscreen == "Yes" else 0],
    "IPS": [1 if ips == "Yes" else 0],
    "ppi": [ppi],
    "Cpu_type": [cpu],
    "HDD": [hdd],
    "SSD": [ssd],
    "Gpu_type": [gpu],
    "OS": [os_type]
})

# Label Encoding
for col in ["Company", "TypeName", "Cpu_type", "Gpu_type", "OS"]:
    le = label_encoders[col]
    query[col] = le.transform(query[col])

# Scale Numerical Features
X_scaled = query.copy()
X_scaled[["Ram", "Weight", "ppi", "HDD", "SSD"]] = scaler.transform(
    query[["Ram", "Weight", "ppi", "HDD", "SSD"]])

# Reorder columns to match training
expected_order = ['Company', 'TypeName', 'Ram', 'Weight', 'Cpu_type',
                  'Touchscreen', 'IPS', 'ppi', 'HDD', 'SSD', 'Gpu_type', 'OS']
X_scaled = X_scaled[expected_order]

# Model selection
model_choice = st.radio("Choose Model", ["Gradient Boosting", "LightGBM", "Ensemble (Avg)"])

if st.button("Predict Price"):
    try:
        if model_choice == "Gradient Boosting":
            log_price = xgb_model.predict(X_scaled)[0]

        elif model_choice == "LightGBM":
            log_price = lgbm_model.predict(X_scaled)[0]

        else:  # Ensemble
            log_price_gb = xgb_model.predict(X_scaled)[0]
            log_price_lgbm = lgbm_model.predict(X_scaled)[0]
            log_price = (log_price_gb + log_price_lgbm) / 2  # Average the log prices

        predicted_price = np.exp(log_price)
        st.success(f"💻 Estimated Laptop Price: ₹{int(predicted_price)}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
