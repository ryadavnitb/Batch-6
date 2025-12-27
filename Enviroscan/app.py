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
    page_title="EnviroScan: Pollution Monitoring",
    layout="wide"
)

st.title("🌍 EnviroScan: Real-Time Pollution Source Monitoring")

# --------------------------------------------------
# BASE PATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data_for_training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "pollution_rf_realistic.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
MAP_FILES = {
    "Main Dashboard Map": "main_dashboard_map.html",
    "Pollution Heatmap": "pollution_heatmap.html",
    "High Risk Zones": "high_risk_zones.html",
}


# --------------------------------------------------
# LOAD MODEL, SCALER, ENCODER
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, target_encoder = load_model()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()

st.success("✅ Data, Model & Scaler Loaded Successfully")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

co = st.sidebar.number_input("CO AQI Value", 0.0, 500.0, 50.0)
no2 = st.sidebar.number_input("NO₂ AQI Value", 0.0, 500.0, 40.0)
ozone = st.sidebar.number_input("Ozone AQI Value", 0.0, 500.0, 30.0)
pm25 = st.sidebar.number_input("PM2.5 AQI Value", 0.0, 500.0, 60.0)
aqi = st.sidebar.number_input("Overall AQI", 0.0, 500.0, 80.0)

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)

traffic_index = st.sidebar.slider("Traffic Pollution Index", 0.0, 10.0, 5.0)


# --------------------------------------------------
# WEATHER INPUT (REQUIRED)
# --------------------------------------------------
st.sidebar.subheader("🌦 Weather")

weather_options = ["Clear", "Cloudy", "Rain", "Fog", "Haze"]
weather = st.sidebar.selectbox("Weather Condition", weather_options)

# MUST match training encoding
weather_map = {w: i for i, w in enumerate(weather_options)}
weather_enc = weather_map[weather]

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
particulate_ratio = pm25 / (aqi + 1)
heat_humidity = temp * humidity

# --------------------------------------------------
# CREATE INPUT DATAFRAME (EXACT TRAINING FEATURES)
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
    particulate_ratio,
    heat_humidity
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
    "particulate_ratio",
    "heat_humidity_index"
])
input_df = input_df[scaler.feature_names_in_]
# --------------------------------------------------
# SCALE & PREDICT
# --------------------------------------------------
X_scaled = scaler.transform(input_df)
prediction = model.predict(X_scaled)[0]
confidence = np.max(model.predict_proba(X_scaled))

predicted_label = target_encoder.inverse_transform([prediction])[0]


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
st.subheader("📌 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Pollution Source", predicted_label)

with col2:
    st.metric("Confidence", f"{confidence*100:.2f}%")


# --------------------------------------------------
# ALERT SYSTEM
# --------------------------------------------------
st.subheader("🚦 Pollution Alert Status")

if confidence >= 0.75:
    if predicted_label == "Industrial":
        st.error("🚨 Severe Industrial Pollution Detected!")
    elif predicted_label == "Vehicular":
        st.warning("🚗 Severe Vehicular Pollution Detected!")
    else:
        st.info("🌿 Natural pollution but elevated levels")

elif confidence >= 0.5:
    if predicted_label == "Industrial":
        st.warning("⚠ Moderate Industrial Pollution Detected")
    elif predicted_label == "Vehicular":
        st.warning("⚠ Moderate Vehicular Pollution Detected")
    else:
        st.success("🌿 Mostly Natural Pollution")

else:
    st.success("✅ Pollution Levels Are Within Safe Limits")



# --------------------------------------------------
# PIE CHART – SOURCE DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Pollution Source Distribution")

source_counts = df["pollution_source"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    source_counts.values,
    labels=source_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")

st.pyplot(fig)

# --------------------------------------------------
# MAP EMBED (MODULE 5 INTEGRATION)
# --------------------------------------------------
st.subheader("🗺 Interactive Pollution Maps")

tabs = st.tabs(list(MAP_FILES.keys()))

for tab, (map_name, file_name) in zip(tabs, MAP_FILES.items()):
    with tab:
        map_path = os.path.join(BASE_DIR, "maps", file_name)

        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=550)
        else:
            st.error(f"Map file not found: {file_name}")

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
    "Wind Speed (m/s)": [wind],
    "Weather": [weather],
    "Predicted Source": [predicted_label],
    "Confidence": [confidence]
})

csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Pollution Report",
    csv,
    "pollution_report.csv",
    "text/csv"
)
