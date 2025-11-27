import streamlit as st
import joblib
import numpy as np

# Load model and scaler (remove scaler if not using it)
model = joblib.load("C:/Users/jeeva/OneDrive/Desktop/Infosys_project/xgboost_air_quality_model.pkl")
# scaler = joblib.load("C:/Users/jeeva/OneDrive/Desktop/Infosys_project/scaler.pkl")

st.title("🌫️ Air Quality Prediction App")
st.write("Enter the environmental values below to predict the Air Quality category.")

# User inputs
pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=10.0)
pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=20.0)
no2 = st.number_input("NO2", min_value=0.0, max_value=500.0, value=15.0)
so2 = st.number_input("SO2", min_value=0.0, max_value=200.0, value=5.0)
co = st.number_input("CO", min_value=0.0, max_value=10.0, value=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
population = st.number_input("Population Density", min_value=1.0, max_value=10000.0, value=500.0)

# Predict button
if st.button("Predict Air Quality"):
    input_data = np.array([[temperature, humidity, pm25, pm10, no2, so2, co, population]])
    
    # ❗ Use this ONLY if you have a scaler.pkl
    # scaled_input = scaler.transform(input_data)
    # prediction = model.predict(scaled_input)[0]

    # If NO SCALER → use raw input
    prediction = model.predict(input_data)[0]

    label_map = {
        0: "Good",
        1: "Moderate",
        2: "Unhealthy",
        3: "Hazardous"
    }

    st.success(f"Predicted Air Quality: **{label_map[prediction]}**")
