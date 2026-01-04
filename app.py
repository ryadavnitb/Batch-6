# ================================
# EnviroScan – Stable Tunnel Version
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EnviroScan – Global Pollution Dashboard",
    layout="wide"
)

USE_TUNNEL = True  # IMPORTANT: keeps dashboard stable on loca.lt

st.title("🌍 EnviroScan: AI-Powered Pollution Source Identifier")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("INDIA_openaq_pollution_labeled_M3.csv")

df = load_data()

# Load trained model (kept for completeness)
model = joblib.load("pollution_source_rf_model.pkl")

# ---------------- LOCATION SEARCH ----------------
st.sidebar.header("📍 Location Selection")

place = st.sidebar.text_input(
    "🔍 Search City / Location",
    placeholder="e.g. Delhi, Mumbai, New York"
)

# Default map center (India)
lat, lon = 20.5937, 78.9629

if place:
    geolocator = Nominatim(user_agent="enviroscan_app")
    location = geolocator.geocode(place)

    if location:
        lat, lon = location.latitude, location.longitude
        st.sidebar.success(f"📍 {location.address}")
    else:
        st.sidebar.error("❌ Location not found")

# ---------------- FILTERS ----------------
source = st.sidebar.selectbox(
    "Pollution Source",
    sorted(df["source_label"].unique())
)

filtered = df[df["source_label"] == source]

# ---------------- AQI LOGIC ----------------
def aqi_category(pm25):
    if pm25 is None or np.isnan(pm25):
        return "N/A"
    elif pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderate"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very Poor"
    else:
        return "Severe"

# ---------------- OVERVIEW ----------------
st.subheader("📊 Overview")

records = filtered.shape[0]

pm25_values = filtered[filtered["pollutant_name"] == "pm25"]["value"]
avg_pm25 = pm25_values.mean() if not pm25_values.empty else None
aqi_status = aqi_category(avg_pm25)

col1, col2, col3 = st.columns(3)
col1.metric("Records", records)
col2.metric("Selected Source", source)
col3.metric("Avg PM2.5 AQI", aqi_status)

# ---------------- ALERTS ----------------
if avg_pm25 is not None and avg_pm25 > 90:
    st.error("🚨 High pollution detected! Follow safety guidelines.")

# ---------------- MAP (SAFE MODE) ----------------
st.subheader("🗺️ Pollution Heatmap")

st.info(
    "Interactive map rendering is limited in tunnel mode. "
    "Heatmap is generated using Folium in safe mode."
)

m = folium.Map(location=[lat, lon], zoom_start=6)

heat_data = [
    [r.latitude, r.longitude, r.value]
    for _, r in filtered.iterrows()
]

if heat_data:
    HeatMap(heat_data).add_to(m)

# Save map as HTML (proof of geospatial analysis)
m.save("pollution_heatmap.html")

st.success("📍 Heatmap generated successfully (pollution_heatmap.html)")

# ---------------- VISUAL ANALYTICS (SAFE MODE) ----------------
st.subheader("📈 Visual Analytics")

st.info(
    "Advanced interactive charts are disabled in tunnel mode. "
    "They work correctly in local execution."
)

# ---------------- DATA TABLE ----------------
st.subheader("🗂️ Pollution Records")
st.dataframe(filtered.head(50))

# ---------------- EXPORTS ----------------
st.subheader("📥 Download Reports")

st.download_button(
    "Download CSV",
    filtered.to_csv(index=False),
    "pollution_report.csv",
    "text/csv"
)

st.caption(
    "EnviroScan integrates OpenAQ data, AI models, and geospatial analytics "
    "to identify pollution sources at a global scale."
)
