import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="EnviroScan – Pollution Source Analysis",
    layout="wide"
)

st.title("🌍 EnviroScan: Pollution Source Identification System")

# --------------------------------------------------
# BASE PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data_for_training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "pollution_rf_realistic.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")

MAP_FILES = {
    "Overview Map": "main_dashboard_map.html",
    "Pollution Intensity": "pollution_heatmap.html",
    "Critical Zones": "high_risk_zones.html",
}

# --------------------------------------------------
# LOAD MODEL COMPONENTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, target_encoder = load_artifacts()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()
st.success("✅ System initialized successfully")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Environmental Inputs")

co = st.sidebar.number_input("CO AQI", 0.0, 500.0, 50.0)
no2 = st.sidebar.number_input("NO₂ AQI", 0.0, 500.0, 40.0)
ozone = st.sidebar.number_input("Ozone AQI", 0.0, 500.0, 30.0)
pm25 = st.sidebar.number_input("PM2.5 AQI", 0.0, 500.0, 60.0)
aqi = st.sidebar.number_input("Overall AQI", 0.0, 500.0, 80.0)

st.sidebar.subheader("🌦 Weather Parameters")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)

traffic_index = st.sidebar.slider("Traffic Density Index", 0.0, 10.0, 5.0)

weather_options = ["Clear", "Cloudy", "Rain", "Fog", "Haze"]
weather = st.sidebar.selectbox("Weather Condition", weather_options)
weather_enc = weather_options.index(weather)

# --------------------------------------------------
# FEATURE ENGINEERING (MODIFIED)
# --------------------------------------------------
particle_load_factor = pm25 / (aqi + 1)
wind_dispersion_index = (co + no2) / (wind + 0.5)
thermal_stress_index = temp * (humidity / 100)

# --------------------------------------------------
# INPUT DATAFRAME
# --------------------------------------------------
input_df = pd.DataFrame([[  
    co,
    no2,
    ozone,
    pm25,
    aqi,
    temp,
    humidity,
    wind,
    weather_enc,
    traffic_index,
    particle_load_factor,
    thermal_stress_index
]], columns=[
    "co_aqi_value",
    "no2_aqi_value",
    "ozone_aqi_value",
    "pm2.5_aqi_value",
    "aqi_value",
    "temperature_c",
    "humidity_%",
    "wind_speed_m/s",
    "weather_description_enc",
    "traffic_pollution_index",
    "particle_load_factor",
    "thermal_stress_index"
])

input_df = input_df[scaler.feature_names_in_]

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
X_scaled = scaler.transform(input_df)
pred_class = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

confidence = float(np.max(probabilities))
predicted_label = target_encoder.inverse_transform([pred_class])[0]

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
st.subheader("📌 Prediction Summary")

c1, c2 = st.columns(2)

with c1:
    st.metric("Predicted Source", predicted_label)

with c2:
    st.metric("Prediction Confidence", f"{confidence*100:.1f}%")

# --------------------------------------------------
# ALERT LOGIC (MODIFIED)
# --------------------------------------------------
st.subheader("🚦 Pollution Risk Assessment")

if confidence > 0.8:
    st.error(f"🚨 High-risk {predicted_label} pollution detected")
elif confidence > 0.6:
    st.warning(f"⚠ Moderate {predicted_label} pollution detected")
else:
    st.success("✅ Pollution levels appear manageable")

# --------------------------------------------------
# SOURCE DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Historical Source Distribution")

counts = df["pollution_source"].value_counts()

fig, ax = plt.subplots()
ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
ax.axis("equal")
st.pyplot(fig)

# --------------------------------------------------
# MAP VISUALIZATION
# --------------------------------------------------
st.subheader("🗺 Spatial Pollution Analysis")

tabs = st.tabs(list(MAP_FILES.keys()))

for tab, (title, file) in zip(tabs, MAP_FILES.items()):
    with tab:
        path = os.path.join(BASE_DIR, "maps", file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=550)
        else:
            st.warning(f"{file} not available")

# --------------------------------------------------
# REPORT DOWNLOAD
# --------------------------------------------------
st.subheader("📥 Export Analysis Report")

report = pd.DataFrame({
    "CO AQI": [co],
    "NO2 AQI": [no2],
    "PM2.5 AQI": [pm25],
    "AQI": [aqi],
    "Temperature": [temp],
    "Humidity": [humidity],
    "Wind Speed": [wind],
    "Weather": [weather],
    "Predicted Source": [predicted_label],
    "Confidence": [confidence]
})

st.download_button(
    "Download CSV Report",
    report.to_csv(index=False).encode("utf-8"),
    file_name="enviroscan_report.csv",
    mime="text/csv"
)
