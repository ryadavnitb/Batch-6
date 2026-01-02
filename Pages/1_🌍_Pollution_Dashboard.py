import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Pollution Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Pollution Analytics Dashboard")
st.caption("Geospatial pollution trends & hotspot analysis")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("labeled_dataset.csv")

try:
    df = load_data()
except Exception as e:
    st.error("❌ Pipeline output not found. Please run pipeline from Home page.")
    st.stop()

# -------------------------------------------------
# COLUMN FIXES
# -------------------------------------------------
if "city" not in df.columns and "city_x" in df.columns:
    df.rename(columns={"city_x": "city"}, inplace=True)

if "timestamp" not in df.columns:
    for col in ["datetime", "date", "time"]:
        if col in df.columns:
            df.rename(columns={col: "timestamp"}, inplace=True)
            break

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])  # remove invalid timestamps

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filters")

# City filter
city = st.sidebar.selectbox(
    "Select City",
    ["All"] + sorted(df["city"].dropna().unique().tolist())
)

# Coordinates filter
min_lat, max_lat = float(df["latitude"].min()), float(df["latitude"].max())
min_lon, max_lon = float(df["longitude"].min()), float(df["longitude"].max())

lat_range = st.sidebar.slider(
    "Latitude Range", min_value=min_lat, max_value=max_lat, value=(min_lat, max_lat), step=0.01
)
lon_range = st.sidebar.slider(
    "Longitude Range", min_value=min_lon, max_value=max_lon, value=(min_lon, max_lon), step=0.01
)

# Time range filter
min_time = df["timestamp"].min().date()
max_time = df["timestamp"].max().date()
time_range = st.sidebar.date_input(
    "Select Date Range",
    value=[min_time, max_time],
    min_value=min_time,
    max_value=max_time
)

# Pollutant filter
ALLOWED_PARAMS = ["PM25 AQ","PM10 AQ","CO AQ","NO2 AQ","SO2 AQ","O3 AQ","AQI","temperature","humidity","wind_speed","wind_deg"]
available_params = [c for c in ALLOWED_PARAMS if c in df.columns]
pollutant = st.sidebar.selectbox("Select Parameter", available_params)

# -------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------
df_filtered = df.copy()

if city != "All":
    df_filtered = df_filtered[df_filtered["city"] == city]

df_filtered = df_filtered[
    (df_filtered["latitude"] >= lat_range[0]) & (df_filtered["latitude"] <= lat_range[1]) &
    (df_filtered["longitude"] >= lon_range[0]) & (df_filtered["longitude"] <= lon_range[1]) &
    (df_filtered["timestamp"].dt.date >= time_range[0]) &
    (df_filtered["timestamp"].dt.date <= time_range[1])
]

# -------------------------------------------------
# METRICS
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("📍 Cities", df_filtered["city"].nunique())
c2.metric("📊 Records", len(df_filtered))
avg_pollution = round(df_filtered[pollutant].mean(), 2) if not df_filtered.empty else 0
c3.metric("⚠️ Avg Pollution", avg_pollution)

# -----------------------------
# REAL-TIME ALERTS
# -----------------------------
SAFE_THRESHOLDS = {
    "PM25 AQ": 60,
    "PM10 AQ": 100,
    "CO AQ": 2,
    "NO2 AQ": 40,
    "SO2 AQ": 20,
    "O3 AQ": 50,
    "AQI": 100
}

if pollutant in SAFE_THRESHOLDS and not df_filtered.empty:
    threshold = SAFE_THRESHOLDS[pollutant]
    
    # Overall average alert
    avg_pollution = df_filtered[pollutant].mean()
    if avg_pollution > threshold:
        st.error(f"⚠️ Alert: Average {pollutant} exceeds safe threshold ({threshold})!")
    else:
        st.success(f"✅ {pollutant} within safe levels ({threshold})")
    
    # Identify individual records exceeding threshold
    critical_points = df_filtered[df_filtered[pollutant] > threshold]
    if not critical_points.empty:
        st.warning(f"🚨 {len(critical_points)} records exceed the safe {pollutant} threshold!")
        # Show a mini table of hotspots (city, timestamp, value)
        st.dataframe(
            critical_points[["city","timestamp", pollutant]].sort_values(pollutant, ascending=False),
            use_container_width=True
        )

# -------------------------------------------------
# TIME SERIES PLOT
# -------------------------------------------------
st.subheader("📈 Pollution Trend Over Time")
if not df_filtered.empty:
    fig = px.line(
        df_filtered.sort_values("timestamp"),
        x="timestamp",
        y=pollutant,
        color="city" if city=="All" else None,
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for selected filters.")

# -------------------------------------------------
# HEATMAP WITH DYNAMIC LAYERS
# -------------------------------------------------
st.subheader("🔥 Pollution Heatmap")
if {"latitude","longitude"}.issubset(df_filtered.columns) and not df_filtered.empty:
    m = folium.Map(location=[df_filtered["latitude"].mean(), df_filtered["longitude"].mean()], zoom_start=5)
    heat_data = list(zip(df_filtered["latitude"], df_filtered["longitude"], df_filtered[pollutant]))
    HeatMap(heat_data, radius=12).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=500)
else:
    st.warning("⚠️ Latitude / Longitude missing or no data available. Heatmap skipped.")

# -------------------------------------------------
# SOURCE DISTRIBUTION
# -------------------------------------------------
st.subheader("🧪 Predicted Pollution Sources")
source_col = None
if "predicted_source" in df_filtered.columns:
    source_col = "predicted_source"
elif "Pollution_Source" in df_filtered.columns:
    source_col = "Pollution_Source"

if source_col and not df_filtered.empty:
    pie = px.pie(df_filtered, names=source_col, title="Predicted Source Distribution")
    st.plotly_chart(pie, use_container_width=True)
else:
    st.warning("⚠️ Pollution source column not found or no data available.")

# -------------------------------------------------
# RAW DATA VIEW
# -------------------------------------------------
with st.expander("📋 View Raw Data"):
    st.dataframe(df_filtered, use_container_width=True)

# -------------------------------------------------
# DOWNLOAD FILTERED RESULTS
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⬇️ Download Data")
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convert_df_to_csv(df_filtered)
file_name = f"pollution_data_{city}_{pollutant}.csv".replace(" ", "_")
st.sidebar.download_button(label="📥 Download Filtered Results", data=csv_data, file_name=file_name, mime="text/csv")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<center style='color:gray;'>EnviroScan AI • Pollution Dashboard</center>", unsafe_allow_html=True)
