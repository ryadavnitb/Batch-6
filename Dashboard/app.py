import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import joblib
import os

# ================= CONFIG =================
DATA_FILE = "final_labeled_dataset.csv"

MODELS = {
    "Decision Tree": "pollution_source_dt_model.pkl",
    "Random Forest": "pollution_source_rf_model.pkl",
    "XGBoost": "pollution_source_xgb_model.pkl",
}

SOURCE_COLORS = {
    "Vehicular": "red",
    "Industrial": "green",
    "Agricultural": "orange",
    "Burning": "purple",
    "Natural": "blue",
}

SOURCE_OPTIONS = list(SOURCE_COLORS.keys())

# ================= PAGE =================
st.set_page_config(page_title="Real-Time Pollution Dashboard", layout="wide")
st.title("🌍 Real-Time Environmental Monitoring Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["latitude", "longitude", "timestamp"], inplace=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # enforce allowed pollution sources only
    df = df[df["Source_Label"].isin(SOURCE_OPTIONS)]

    return df

df = load_data(DATA_FILE)

if df.empty:
    st.error("❌ Dataset is empty or contains invalid pollution labels.")
    st.stop()

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()

# ================= SIDEBAR =================
st.sidebar.header("🔎 Filters")

# ---- City Filter ----
if "city" in df.columns:
    city = st.sidebar.selectbox("City", ["All"] + sorted(df["city"].dropna().unique()))
    if city != "All":
        df = df[df["city"] == city]

if df.empty:
    st.warning("No data after city filter")
    st.stop()

# ---- Pollution Source Filter (FIXED OPTIONS) ----
selected_sources = st.sidebar.multiselect(
    "Pollution Source",
    SOURCE_OPTIONS,
    default=SOURCE_OPTIONS
)

df = df[df["Source_Label"].isin(selected_sources)]

if df.empty:
    st.warning("No data after pollution source filter")
    st.stop()

# ---- Latitude / Longitude (NO SLIDER ERROR) ----
lat_min, lat_max = float(df.latitude.min()), float(df.latitude.max())
lon_min, lon_max = float(df.longitude.min()), float(df.longitude.max())

lat_range = (lat_min, lat_max)
lon_range = (lon_min, lon_max)

if lat_min != lat_max:
    lat_range = st.sidebar.slider("Latitude Range", lat_min, lat_max, lat_range)
else:
    st.sidebar.info(f"Latitude fixed: {lat_min}")

if lon_min != lon_max:
    lon_range = st.sidebar.slider("Longitude Range", lon_min, lon_max, lon_range)
else:
    st.sidebar.info(f"Longitude fixed: {lon_min}")

df = df[
    df.latitude.between(*lat_range) &
    df.longitude.between(*lon_range)
]

if df.empty:
    st.warning("No data after coordinate filter")
    st.stop()

# ---- Date Filter ----
start_date = st.sidebar.date_input("Start Date", df.timestamp.min().date())
end_date = st.sidebar.date_input("End Date", df.timestamp.max().date())

df = df[
    (df.timestamp.dt.date >= start_date) &
    (df.timestamp.dt.date <= end_date)
]

if df.empty:
    st.warning("No data after date filter")
    st.stop()

# ---- Pollutant ----
pollutants = [c for c in df.columns if c == "AQI" or c.endswith("AQ")]
selected_pollutant = st.sidebar.selectbox("Pollutant", pollutants)

st.success(f"✅ {len(df)} records loaded")

# ================= ALERTS =================
st.subheader("⚠️ Pollution Alerts")

if "AQI" in df.columns:
    risk = df[df.AQI > 100]
    if not risk.empty:
        st.warning(f"{len(risk)} locations exceed AQI 100")
    else:
        st.success("All locations are within safe AQI limits")

# ================= MODEL PREDICTIONS =================
st.subheader("🧪 Model Predictions")

for name, model in models.items():
    required = list(model.feature_names_in_)
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(f"{name} skipped (missing features)")
        continue

    X = df[required]
    df[f"Predicted_{name}"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        df[f"Confidence_{name}"] = model.predict_proba(X).max(axis=1)

pred_cols = [c for c in df.columns if c.startswith("Predicted_")]
if pred_cols:
    st.dataframe(df[["location_id", "Source_Label"] + pred_cols].head(20))

# ================= MAP =================
st.subheader("📍 Pollution Source Map")

m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=6)
cluster = MarkerCluster().add_to(m)

for _, r in df.iterrows():
    folium.CircleMarker(
        [r.latitude, r.longitude],
        radius=6,
        color=SOURCE_COLORS[r.Source_Label],
        fill=True,
        fill_opacity=0.7,
        popup=f"""
        <b>Source:</b> {r.Source_Label}<br>
        <b>{selected_pollutant}:</b> {r[selected_pollutant]}
        """
    ).add_to(cluster)

st_folium(m, height=650, width=1200)

# ================= HEATMAP =================
st.subheader(f"🔥 {selected_pollutant} Heatmap")

heat_data = df[["latitude", "longitude", selected_pollutant]].dropna().values.tolist()
heat_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=6)
HeatMap(heat_data, radius=15).add_to(heat_map)
st_folium(heat_map, height=650, width=1200)

# ================= PIE CHART (ERROR-FREE) =================
st.subheader("📊 Pollution Source Distribution")

pie_df = df["Source_Label"].value_counts().reset_index()
pie_df.columns = ["Source_Label", "count"]

fig_pie = px.pie(
    pie_df,
    names="Source_Label",
    values="count",
    color="Source_Label",
    color_discrete_map=SOURCE_COLORS
)

st.plotly_chart(fig_pie, use_container_width=True)

# ================= AQI TREND (FIXED LINE CHART) =================
st.subheader(f"📈 {selected_pollutant} Trend Over Time")

trend_df = df[["timestamp", selected_pollutant]].dropna()

# Safety check
if len(trend_df) < 2:
    st.warning("⚠️ Not enough data points to draw trend line.")
else:
    trend_df = (
        trend_df
        .set_index("timestamp")
        .resample("D")[selected_pollutant]
        .mean()
        .reset_index()
    )

    # Remove zero-only data
    if trend_df[selected_pollutant].sum() == 0:
        st.warning(f"⚠️ {selected_pollutant} values are all zero.")
    else:
        fig_trend = px.line(
            trend_df,
            x="timestamp",
            y=selected_pollutant,
            title=f"{selected_pollutant} Daily Trend"
        )

        fig_trend.update_traces(
            mode="lines",
            line=dict(width=3)
        )

        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_pollutant,
            template="plotly_white"
        )

        st.plotly_chart(fig_trend, use_container_width=True)


# ================= DOWNLOAD =================
st.subheader("📥 Download Filtered Data")

st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    "final_pollution_dashboard.csv",
    "text/csv",
)
