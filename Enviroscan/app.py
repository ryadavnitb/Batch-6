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
    page_title="EnviroScan: Pollution Source Monitoring",
    layout="wide"
)

st.title("🌍 EnviroScan: Pollution Source Identification Dashboard")

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
    "Pollution Heatmap": "pollution_heatmap.html",
    "High Risk Zones": "high_risk_zones.html",
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
st.success("✅ Model and data loaded successfully")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Environmental Inputs")

co = st.sidebar.number_input("CO AQI Value", 0.0, 500.0, 50.0)
no2 = st.sidebar.number_input("NO₂ AQI Value", 0.0, 500.0, 40.0)
ozone = st.sidebar.number_input("Ozone AQI Value", 0.0, 500.0, 30.0)
pm25 = st.sidebar.number_input("PM2.5 AQI Value", 0.0, 500.0, 60.0)
aqi = st.sidebar.number_input("Overall AQI", 0.0, 500.0, 80.0)

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)
wind_dir = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 180.0)

st.sidebar.subheader("📍 Spatial Context (Defaults)")
st.sidebar.caption("Used to maintain feature consistency")

# --------------------------------------------------
# INPUT DATAFRAME (EXACT TRAINING SCHEMA)
# --------------------------------------------------
input_df = pd.DataFrame([{
    "latitude": 0.0,
    "longitude": 0.0,
    "co aqi value": co,
    "ozone aqi value": ozone,
    "no2 aqi value": no2,
    "pm2.5 aqi value": pm25,
    "aqi value": aqi,
    "temperature (°c)": temp,
    "humidity (%)": humidity,
    "wind speed (m/s)": wind_speed,
    "wind direction (Ã¢Â°)": wind_dir,   # IMPORTANT: encoding match
    "road_count": 1,
    "industrial_count": 1,
    "farmland_count": 0,
    "dump_site_count": 0,
    "recycling_count": 0,
    "green_area_count": 1,
    "near_road_2km": 1,
    "near_industry_2km": 0,
    "near_dump_2km": 0,
    "dist_city_km": 5.0,
    "dist_road_km": 0.5,
    "dist_industry_km": 3.0,
    "dist_dump_km": 4.0
}])

# Align with training feature order
input_df = input_df[scaler.feature_names_in_]

# --------------------------------------------------
# SCALE & PREDICT
# --------------------------------------------------
X_scaled = scaler.transform(input_df)
pred_class = model.predict(X_scaled)[0]
probs = model.predict_proba(X_scaled)[0]

confidence = float(np.max(probs))
predicted_label = target_encoder.inverse_transform([pred_class])[0]

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
st.subheader("📌 Prediction Result")

c1, c2 = st.columns(2)

with c1:
    st.metric("Predicted Pollution Source", predicted_label)

with c2:
    st.metric("Prediction Confidence", f"{confidence*100:.2f}%")

# --------------------------------------------------
# ALERT SYSTEM
# --------------------------------------------------
st.subheader("🚦 Pollution Alert Status")

if confidence > 0.8:
    st.error(f"🚨 High risk {predicted_label} pollution detected")
elif confidence > 0.6:
    st.warning(f"⚠ Moderate {predicted_label} pollution detected")
else:
    st.success("✅ Pollution levels are within acceptable limits")

# --------------------------------------------------
# PIE CHART – SOURCE DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Pollution Source Distribution")

if "pollution_source" in df.columns:
    counts = df["pollution_source"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("Pollution source labels not available in dataset")

# --------------------------------------------------
# MAP EMBED
# --------------------------------------------------
st.subheader("🗺 Interactive Pollution Maps")

tabs = st.tabs(list(MAP_FILES.keys()))

for tab, (title, file) in zip(tabs, MAP_FILES.items()):
    with tab:
        path = os.path.join(BASE_DIR, "maps", file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=550)
        else:
            st.warning(f"{file} not found")

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------
st.subheader("📥 Download Pollution Report")

report_df = pd.DataFrame({
    "CO AQI": [co],
    "NO2 AQI": [no2],
    "Ozone AQI": [ozone],
    "PM2.5 AQI": [pm25],
    "AQI": [aqi],
    "Temperature (°C)": [temp],
    "Humidity (%)": [humidity],
    "Wind Speed (m/s)": [wind_speed],
    "Wind Direction (°)": [wind_dir],
    "Predicted Source": [predicted_label],
    "Confidence": [confidence]
})

st.download_button(
    "Download Report (CSV)",
    report_df.to_csv(index=False).encode("utf-8"),
    file_name="enviroscan_report.csv",
    mime="text/csv"
)
