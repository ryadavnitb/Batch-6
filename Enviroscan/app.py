import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import folium
from streamlit_folium import st_folium
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
# PATHS (VERY IMPORTANT)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data_for_training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "xgboost_pollution.pkl")

MAP_PATH = os.path.join(BASE_DIR, "maps", "main_dashboard_map.html")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# LOAD DATA (FIX UNICODE ERROR HERE)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()

st.success("✅ Data & Model Loaded Successfully")

# --------------------------------------------------
# CLEAN COLUMN NAMES (ROBUST & SAFE)
# --------------------------------------------------
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace("°", "", regex=False)
    .str.replace("ã¢â", "", regex=False)
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "pct", regex=False)
    .str.replace("/", "_", regex=False)
)
# st.write("✅ Columns after cleaning:", df.columns.tolist())

FEATURES = [
    "temperature_c",
    "dist_industry_km",
    "dist_road_km",
    "dist_dump_km"
]

TARGET = "pollution_source"

df = df.dropna(subset=FEATURES + [TARGET])


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

temp = st.sidebar.slider(
    "Temperature (°C)",
    float(df["temperature_c"].min()),
    float(df["temperature_c"].max()),
    float(df["temperature_c"].mean())
)


        # humidity = st.sidebar.slider(
        #     "Humidity (%)",
        #     float(df.humidity_pct.min()),
        #     float(df.humidity_pct.max()),
        #     float(df.humidity_pct.mean())
        # )

        # wind = st.sidebar.slider(
        #     "Wind Speed (m/s)",
        #     float(df.wind_speed_m_s.min()),
        #     float(df.wind_speed_m_s.max()),
        #     float(df.wind_speed_m_s.mean())
        # )

road = st.sidebar.slider(
    "Distance to Road (km)",
    0.0, float(df.dist_road_km.max()), 2.0
)

industry = st.sidebar.slider(
    "Distance to Industry (km)",
    0.0, float(df.dist_industry_km.max()), 3.0
)

dump = st.sidebar.slider(
    "Distance to Dump Site (km)",
    0.0, float(df.dist_dump_km.max()), 4.0
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
input_data = np.array([[temp, road, industry, dump]])
prediction = model.predict(input_data)[0]
confidence = model.predict_proba(input_data).max()

label_map = {
    0: "Industrial",
    1: "Natural",
    2: "Vehicular"
}

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
st.subheader("📌 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Source", label_map[prediction])

with col2:
    st.metric("Confidence", f"{confidence*100:.2f}%")

# --------------------------------------------------
# ALERT SYSTEM
# --------------------------------------------------
if prediction == 0 and confidence > 0.6:
    st.error("🚨 High Industrial Pollution Risk Detected!")
elif prediction == 2 and confidence > 0.6:
    st.warning("⚠ Vehicular Pollution Likely")
else:
    st.success("✅ Pollution Levels Appear Natural")

# --------------------------------------------------
# PIE CHART – SOURCE DISTRIBUTION
# --------------------------------------------------
source_counts = df[TARGET].value_counts()

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
st.subheader("🗺 Pollution Heatmap & Sources")

if os.path.exists(MAP_PATH):
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=550)
else:
    st.warning("Map file not found.")

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------
st.subheader("📥 Download Report")

report_df = pd.DataFrame({
    "Temperature (°C)": [temp],
    "Distance to Industry (km)": [industry],
    "Distance to Road (km)": [road],
    "Distance to Dump (km)": [dump],
    "Predicted Source": [label_map[prediction]],
    "Confidence": [confidence]
})

csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Pollution Report",
    csv,
    "pollution_report.csv",
    "text/csv"
)
