import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI-EnviroScan Dashboard", layout="wide")

# =========================
# LOAD MODEL + ENCODER
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("models/source_classifier_balanced_rf.pkl", mmap_mode="r")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

model, label_encoder = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/india_states/osm_enriched_dataset.csv")

df = load_data()

# =========================
# FEATURES
# =========================
feature_cols = [
    'value','hour','month','weekday',
    'recent_spike','pm_trend_mean_3h','pm_trend_mean_6h',
    'near_road','near_industry','near_farmland',
    'near_dumpyard','near_landfill','latitude','longitude'
]

# =========================
# HEADER
# =========================
st.title("🌍 AI-EnviroScan — Pollution Source Intelligence Dashboard")
st.markdown("---")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🧠 Source Prediction Input")

value = st.sidebar.number_input("Pollutant Value", value=50.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 10)
month = st.sidebar.slider("Month", 1, 12, 6)
weekday = st.sidebar.slider("Weekday (0 = Mon)", 0, 6, 2)

recent_spike = st.sidebar.selectbox("Recent Spike Detected?", [0,1])
trend3 = st.sidebar.number_input("PM Trend (3h avg)", value=0.0)
trend6 = st.sidebar.number_input("PM Trend (6h avg)", value=0.0)

near_road = st.sidebar.selectbox("Near Road?", [0,1])
near_industry = st.sidebar.selectbox("Near Industry?", [0,1])
near_farmland = st.sidebar.selectbox("Near Farmland?", [0,1])
near_dumpyard = st.sidebar.selectbox("Near Dumpyard?", [0,1])
near_landfill = st.sidebar.selectbox("Near Landfill?", [0,1])

lat = st.sidebar.number_input("Latitude", value=float(df['latitude'].mean()))
lon = st.sidebar.number_input("Longitude", value=float(df['longitude'].mean()))

user_features = pd.DataFrame([[value,hour,month,weekday,
    recent_spike,trend3,trend6,
    near_road,near_industry,near_farmland,
    near_dumpyard,near_landfill,
    lat,lon]], columns=feature_cols).astype(float)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("🔮 Predict Pollution Source", type="primary"):
    with st.spinner("Predicting..."):
        probs = model.predict_proba(user_features)[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.success(f"Predicted Source: **{pred_label}**")

        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": np.round(probs, 3)
        }).sort_values("Probability", ascending=False)

        st.write("📊 **Prediction Confidence**")
        st.dataframe(prob_df, use_container_width=True)

st.markdown("---")

# =========================
# 🗺️ STYLED POLLUTION HEATMAP
# =========================
st.subheader("🔥 Pollution Heatmap & Sources")

@st.cache_data
def styled_heatmap():

    m = folium.Map(location=[20.59, 78.96], zoom_start=5, tiles=None)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Map").add_to(m)

    heat_df = df[['latitude','longitude','value']].dropna()

    HeatMap(
        heat_df.values.tolist(),
        name="Pollution Heatmap",
        radius=18,
        blur=22,
        max_zoom=8,
        gradient={0.2:"green",0.4:"yellow",0.6:"orange",0.8:"red",1.0:"purple"}
    ).add_to(m)

    # Title Card
    m.get_root().html.add_child(folium.Element("""
   
    """))

    # Legend
    m.get_root().html.add_child(folium.Element("""
    <div style='position: fixed; bottom: 40px; right: 40px; z-index:9999;
         background:white; padding:12px 15px; border-radius:8px;
         width:180px; box-shadow:0 2px 10px rgba(0,0,0,0.25); font-size:13px'>
         <b>AQI Severity</b><br><br>
         <span style='color:green'>●</span> Low<br>
         <span style='color:yellow'>●</span> Moderate<br>
         <span style='color:orange'>●</span> High<br>
         <span style='color:red'>●</span> Very High<br>
         <span style='color:purple'>●</span> Extreme
    </div>
    """))

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

map_html = styled_heatmap()
st.components.v1.html(map_html, height=600, scrolling=False)

st.info("💡 Tip: You can switch models by changing the model path in load_model().")
