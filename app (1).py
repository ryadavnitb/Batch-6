import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from geopy.geocoders import Nominatim
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EnviroScan – Global Pollution Dashboard",
    layout="wide"
)

st.title("🌍 EnviroScan: AI-Powered Pollution Source Identifier")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("INDIA_openaq_pollution_labeled_M3.csv")

df = load_data()

# Load trained model
model = joblib.load("pollution_source_rf_model.pkl")

# ---------------- LOCATION SEARCH ----------------
st.sidebar.header("📍 Location Selection")

city = st.sidebar.text_input("Enter City Name (optional)")
lat = st.sidebar.number_input("Latitude", value=float(df.latitude.mean()))
lon = st.sidebar.number_input("Longitude", value=float(df.longitude.mean()))

if city:
    geolocator = Nominatim(user_agent="enviroscan")
    location = geolocator.geocode(city)
    if location:
        lat, lon = location.latitude, location.longitude

# ---------------- FILTERS ----------------
source = st.sidebar.selectbox(
    "Pollution Source",
    sorted(df.source_label.unique())
)

filtered = df[df.source_label == source]

# ---------------- AQI LOGIC ----------------
def aqi_category(pm25):
    if pm25 <= 30: return "Good", "green"
    elif pm25 <= 60: return "Satisfactory", "yellow"
    elif pm25 <= 90: return "Moderate", "orange"
    elif pm25 <= 120: return "Poor", "red"
    elif pm25 <= 250: return "Very Poor", "purple"
    else: return "Severe", "maroon"

# ---------------- METRICS ----------------
st.subheader("📊 Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Records", filtered.shape[0])
col2.metric("Selected Source", source)

avg_pm25 = filtered[filtered.pollutant_name == "pm25"].value.mean()
if not np.isnan(avg_pm25):
    cat, color = aqi_category(avg_pm25)
    col3.metric("Avg PM2.5 AQI", cat)

# ---------------- ALERTS ----------------
if avg_pm25 and avg_pm25 > 90:
    st.error("🚨 High pollution detected! Follow safety guidelines.")

# ---------------- MAP ----------------
st.subheader("🗺️ Pollution Heatmap")

m = folium.Map(location=[lat, lon], zoom_start=5)

heat_data = [
    [r.latitude, r.longitude, r.value]
    for _, r in filtered.iterrows()
]

HeatMap(heat_data).add_to(m)
st_folium(m, width=1000, height=500)

# ---------------- CHARTS ----------------
st.subheader("📈 Visual Analytics")

col1, col2 = st.columns(2)

with col1:
    pie = px.pie(
        df,
        names="source_label",
        title="Pollution Source Distribution"
    )
    st.plotly_chart(pie, use_container_width=True)

with col2:
    bar = px.bar(
        df,
        x="pollutant_name",
        y="value",
        title="Pollutant Concentrations",
        color="pollutant_name"
    )
    st.plotly_chart(bar, use_container_width=True)

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
