import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Hugging Face model details
# -----------------------------
MODEL_REPO_ID = "Sasimscct/Predictive-Maintenance"
MODEL_FILENAME = "best_predictive_maintenance_model.joblib"

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance App",
    layout="centered"
)

st.title("Predictive Maintenance for Engine Health")
st.markdown("Enter the engine sensor values below to predict whether the engine is healthy or requires maintenance.")

# -----------------------------
# Load model from Hugging Face
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter Sensor Readings")

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=700.0, step=1.0)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, value=3.5, step=0.1)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=4.0, step=0.1)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.5, step=0.1)
lub_oil_temp = st.number_input("Lubricating Oil Temperature", min_value=0.0, value=75.0, step=0.1)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, value=80.0, step=0.1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Engine Condition"):
    input_df = pd.DataFrame([{
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp
    }])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Engine Requires Maintenance")
        st.write(f"Confidence: {prediction_proba[1]:.2%}")
    else:
        st.success("Engine is Healthy")
        st.write(f"Confidence: {prediction_proba[0]:.2%}")
