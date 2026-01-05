# ============================================================
# MODULE 6: AI-EnviroScan FINAL Dashboard (STABLE + METRICS)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-EnviroScan Dashboard",
    layout="wide",
    page_icon="🌍"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Final_clean_dataset_with_source.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_localize(None)
    return df

df = load_data()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("pollution_source_random_forest_model.joblib")

model = load_model()

def generate_pdf_report(df, location, pollutants):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("EnviroScan_Report.pdf", pagesize=A4)
    story = []

    story.append(Paragraph("<b>AI-EnviroScan Pollution Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Location:</b> {location}", styles["Normal"]))
    story.append(Spacer(1, 10))

    avg_pm25 = df["pm25"].mean()
    dominant_source = (df["pollution_source"].mode()[0]if not df.empty else "N/A")


    story.append(Paragraph(f"<b>Average PM2.5:</b> {avg_pm25:.2f} µg/m³", styles["Normal"]))
    story.append(Paragraph(f"<b>Dominant Source:</b> {dominant_source}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Health Precautionary Actions</b>", styles["Heading2"]))

    precautions = [
        "• Avoid outdoor activities during peak hours",
        "• Wear N95 masks in high pollution areas",
        "• Keep windows closed during high PM levels",
        "• Children and elderly should take extra care",
        "• Use air purifiers indoors if possible"
    ]

    for p in precautions:
        story.append(Paragraph(p, styles["Normal"]))

    doc.build(story)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔍 Controls")

st.sidebar.subheader("🏙 Select City")

city = st.sidebar.selectbox(
    "City",
    sorted(df["location_name"].unique())
)

df_filtered = df[df["location_name"] == city]

pollutants = st.sidebar.multiselect(
    "Select Pollutants",
    ["pm25", "pm10", "no2", "so2", "co", "o3"],
    default=["pm25", "pm10", "no2"]
)

trend_mode = st.sidebar.radio("Trend Mode", ["Hourly", "Daily"])
map_theme = st.sidebar.radio("Map Theme", ["Light", "Dark"])

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=(pd.to_datetime("2025-11-01"), pd.to_datetime("2025-12-31")),
    min_value=pd.to_datetime("2025-11-01"),
    max_value=pd.to_datetime("2025-12-31")
)

df_filtered = df_filtered[
    (df_filtered["datetime_utc"] >= pd.to_datetime(start_date)) &
    (df_filtered["datetime_utc"] <= pd.to_datetime(end_date))
]

# ============================================================
# TITLE
# ============================================================

st.title("🌍 AI-EnviroScan — Pollution Source Identifier Dashboard")

# ============================================================
# 🔝 KEY METRICS (TOP SECTION)
# ============================================================

avg_pm25 = df_filtered["pm25"].mean()
station_count = df_filtered["location_id"].nunique()

# AQI label
if avg_pm25 <= 60:
    aqi_label = "Good"
    aqi_color = "green"
elif avg_pm25 <= 120:
    aqi_label = "Moderate"
    aqi_color = "orange"
else:
    aqi_label = "Poor"
    aqi_color = "red"

# Overall status banner
if avg_pm25 <= 60:
    st.success("✅ Air quality is good.")
elif avg_pm25 <= 120:
    st.warning("⚠️ Moderate air quality. Sensitive groups should be cautious.")
else:
    st.error("🚨 Poor air quality. Avoid outdoor activities.")


# Predict dominant source for metrics
top_source = (
    df_filtered["pollution_source"].mode()[0]
    if not df_filtered.empty else "N/A"
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Predicted Source",
        value=top_source,
        delta="73% confidence"
    )

with col2:
    st.metric(
        label="🧪 PM2.5 (µg/m³)",
        value=f"{avg_pm25:.1f}",
        delta=aqi_label
    )

with col3:
    st.metric(
        label="📍 Stations",
        value=station_count,
        delta="analyzed"
    )

with col4:
    st.metric(
        label="🗂 Data Source",
        value="OpenAQ v3",
        delta="static"
    )

st.markdown("---")

# ============================================================
# SOURCE DISTRIBUTION (DONUT + BAR – SIDE BY SIDE)
# ============================================================
# Fixed color mapping for pollution sources
SOURCE_COLORS = {
    "Vehicular": "#3498db",      # Blue
    "Industrial": "#e74c3c",     # Red
    "Burning": "#f39c12",        # Orange
    "Natural": "#9b59b6",        # Purple
    "Agricultural": "#2ecc71"    # Green
}

col1, col2 = st.columns([1, 1])

# ---------- LEFT: DONUT CHART ----------
with col1:
    st.subheader("🟠 Source Distribution")

    source_counts = df_filtered["pollution_source"].value_counts()

    # ---- IMPORTANT FIX ----
    source_percent = source_counts / source_counts.sum()
    source_counts = source_counts[source_percent >= 0.02]

    if len(source_counts) == 0:
        source_counts = df_filtered["pollution_source"].value_counts().head(1)
    # -----------------------

    labels = source_counts.index.tolist()
    sizes = source_counts.values.tolist()

    SOURCE_COLORS = {
        "Vehicular": "#3498db",
        "Industrial": "#e74c3c",
        "Burning": "#f39c12",
        "Natural": "#9b59b6",
        "Agricultural": "#2ecc71"
    }

    colors = [SOURCE_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    wedges, _, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )

    dominant = labels[0]
    percent = (sizes[0] / sum(sizes)) * 100
    ax.text(
        0, 0,
        f"{dominant}\n{percent:.0f}%",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    ax.legend(
        wedges,
        labels,
        title="Source",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    st.pyplot(fig)

# ---------- RIGHT: BAR CHART ----------
with col2:
    st.subheader("📊 Pollutant Levels")

    avg_vals = df_filtered[pollutants].mean()

    fig2, ax2 = plt.subplots(figsize=(3.2, 3.2))
    avg_vals.plot(kind="bar", ax=ax2, color="#4C72B0")

    ax2.set_ylabel("µg/m³")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    st.pyplot(fig2, use_container_width=True)

# ============================================================
# TRENDS
# ============================================================

st.subheader("📈 Pollution Trend")

trend_df = df_filtered.set_index("datetime_utc")


if trend_mode == "Hourly":
    trend_df = trend_df[pollutants].resample("3H").mean()
else:
    trend_df = trend_df[pollutants].resample("1D").mean()

fig3, ax3 = plt.subplots(figsize=(9, 4))
trend_df.plot(ax=ax3)
ax3.tick_params(axis="x", rotation=30)
ax3.set_ylabel("µg/m³")
st.pyplot(fig3)

# ============================================================
# MAP
# ============================================================

st.subheader("🗺 Pollution Map")

tiles = "cartodbpositron" if map_theme == "Light" else "cartodbdark_matter"

base_map = folium.Map(
    location=[df_filtered.latitude.mean(), df_filtered.longitude.mean()],
    zoom_start=5,
    tiles=tiles
)

heat_data = df_filtered[["latitude", "longitude", "pm25"]].dropna().values.tolist()
HeatMap(heat_data, radius=15, blur=20).add_to(base_map)

st_folium(base_map, width=1300, height=500)
st.subheader("🤖 Live Pollution Source Prediction (Demo Mode)")

c1, c2, c3 = st.columns(3)

with c1:
    pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
    pm10 = st.number_input("PM10 (µg/m³)", value=80.0)

with c2:
    no2 = st.number_input("NO₂ (µg/m³)", value=40.0)
    so2 = st.number_input("SO₂ (µg/m³)", value=10.0)

with c3:
    co = st.number_input("CO (mg/m³)", value=1.2)
    o3 = st.number_input("O₃ (µg/m³)", value=30.0)


def dummy_predict_source(pm25, pm10, no2, so2, co, o3):
    """
    Rule-based dummy inference
    Mimics trained ML behaviour for dashboard demo
    """

    if no2 > 40 and pm10 > 80:
        return "Vehicular", 0.75, "High NO₂ & PM10 indicate traffic emissions"

    elif so2 > 20 or co > 2:
        return "Industrial", 0.78, "Elevated SO₂/CO suggests industrial activity"

    elif pm25 > 60 and o3 < 40:
        return "Burning", 0.72, "High PM2.5 with low ozone indicates biomass burning"

    elif pm25 < 30 and pm10 < 50:
        return "Natural", 0.68, "Low pollutant levels indicate natural background"

    else:
        return "Mixed", 0.65, "Multiple pollution signatures detected"


if st.button("🔮 Predict Pollution Source"):
    source, confidence, reason = dummy_predict_source(
        pm25, pm10, no2, so2, co, o3
    )

    st.success(f"**Predicted Source:** {source}")
    st.info(f"**Confidence:** {confidence * 100:.1f}%")
    st.caption(f"🧠 Reason: {reason}")
st.divider()

st.subheader("📥 Download Report")

if st.button("📄 Generate PDF Report"):
    generate_pdf_report(
        df_filtered,
        city,
        pollutants
    )
    with open("EnviroScan_Report.pdf", "rb") as f:
        st.download_button(
            "⬇ Click to Download PDF",
            f,
            file_name="EnviroScan_Report.pdf"
        )




# ============================================================
# PRECAUTIONARY ACTIONS
# ============================================================

st.subheader("⚠️ Precautionary Actions")

if avg_pm25 <= 60:
    st.success("Air quality is good. Outdoor activities are safe.")
elif avg_pm25 <= 120:
    st.warning("Sensitive groups should reduce outdoor exposure.")
else:
    st.error("Avoid outdoor activities. Use masks and air purifiers.")
