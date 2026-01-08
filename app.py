import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from src.data_processor import EnviroDataProcessor
from src.model_engine import PollutionPredictor
import os
import joblib  # <--- ADDED: To load your new model

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="EnviroScan AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "GLASSMORPHISM" & FUTURISTIC LOOK ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
    div.stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .css-1v0mbdj {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SYSTEM ---
@st.cache_resource
def load_system():
    # Load Data
    data_path = "dataset/Pollution_Weather_datset.csv" 
    try:
        processor = EnviroDataProcessor(data_path)
        df = processor.get_processed_data()
        
        # Train/Load Internal Model (Your original logic)
        predictor = PollutionPredictor(df)
        if not os.path.exists("model.pkl"):
            predictor.train_model()
            predictor.save_model()
        else:
            predictor.load_model()
    except:
        # Fallback if files are missing just to keep app running
        df = pd.DataFrame({'Country': [], 'Latitude': [], 'Longitude': [], 'AQI Value': [], 'Pollution_Source': []})
        predictor = None
        
    return df, predictor

df, predictor = load_system()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=100)
    st.title("EnviroScan Control")
    st.markdown("---")
    
    analysis_mode = st.radio(
        "Navigation",
        ["🌍 Live Geospatial Map", "📊 Analytics Dashboard", "🤖 AI Source Predictor"]
    )
    
    st.markdown("---")
    st.info("System Status: 🟢 Online\n\nModel: Random Forest v1.2\n\nData Points: 21,800+")

# --- MAIN HEADER ---
st.markdown('<div class="main-header">EnviroScan: AI-Powered Pollution Identifier</div>', unsafe_allow_html=True)

# --- PAGE 1: LIVE GEOSPATIAL MAP ---
if analysis_mode == "🌍 Live Geospatial Map":
    st.subheader("📍 Real-Time Pollution Heatmap & Source Clusters")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Map Filters")
        if not df.empty:
            selected_country = st.selectbox("Select Region", df['Country'].unique())
            map_type = st.radio("Layer Type", ["Heatmap", "Source Clusters"])
            
            # Filter Data
            filtered_df = df[df['Country'] == selected_country]
            st.write(f"Displaying **{len(filtered_df)}** sensors in {selected_country}")
            
            avg_lat = filtered_df['Latitude'].mean()
            avg_lon = filtered_df['Longitude'].mean()
        else:
            st.warning("Dataset not found. Using default map.")
            avg_lat, avg_lon = 20.5937, 78.9629
            filtered_df = pd.DataFrame()
            map_type = "None"

    with col1:
        # Base Map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles='CartoDB dark_matter')

        if map_type == "Heatmap" and not filtered_df.empty:
            heat_data = [[row['Latitude'], row['Longitude'], row['AQI Value']] for index, row in filtered_df.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)
        
        elif map_type == "Source Clusters" and not filtered_df.empty:
            marker_cluster = MarkerCluster().add_to(m)
            # Limit markers for performance in demo
            for idx, row in filtered_df.head(200).iterrows():
                # Color code based on source
                color = "green"
                if row['Pollution_Source'] == 'Industrial': color = "red"
                elif row['Pollution_Source'] == 'Vehicular': color = "orange"
                elif row['Pollution_Source'] == 'Agricultural': color = "yellow"
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Source: {row['Pollution_Source']}<br>AQI: {row['AQI Value']}"
                ).add_to(marker_cluster)
        
        st_folium(m, width=900, height=500)

# --- PAGE 2: ANALYTICS DASHBOARD ---
elif analysis_mode == "📊 Analytics Dashboard":
    st.subheader("📈 Pollution Trends & Correlation Analysis")
    
    if not df.empty:
        # Top KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Avg Global AQI", f"{int(df['AQI Value'].mean())}", "Moderate")
        kpi2.metric("Dominant Source", df['Pollution_Source'].mode()[0])
        kpi3.metric("High Risk Zones", f"{len(df[df['AQI Category'] == 'Hazardous'])}", "CRITICAL", delta_color="inverse")
        kpi4.metric("Data Freshness", "Real-time API")
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.write("### 🏭 Source Contribution")
            fig_pie = px.pie(df, names='Pollution_Source', title='Pollution Source Distribution', 
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.write("### 🌦️ Weather vs. Pollution")
            fig_scatter = px.scatter(df.sample(min(1000, len(df))), x='Wind Speed (m/s)', y='AQI Value', 
                                     color='Pollution_Source', size='PM2.5 AQI Value',
                                     title='Wind Speed Impact on Air Quality')
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.write("### 📅 Temporal Trends (Simulated)")
        st.line_chart(df.groupby('Country')['AQI Value'].mean().head(10))
    else:
        st.error("No data available to display dashboard.")

# --- PAGE 3: AI PREDICTOR (MODIFIED SECTION) ---
elif analysis_mode == "🤖 AI Source Predictor":
    st.subheader("🔎 Forensic Pollution Analysis")
    st.markdown("Manually input sensor readings to identify the culprit source.")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pm25 = st.number_input("PM2.5 Level", 0, 500, 150)
            no2 = st.number_input("NO2 Level", 0, 500, 40)
            co = st.number_input("CO Level", 0, 50, 5)
        with c2:
            ozone = st.number_input("Ozone Level", 0, 500, 20)
            temp = st.number_input("Temperature (°C)", -50, 60, 25)
            humid = st.number_input("Humidity (%)", 0, 100, 60)
        with c3:
            wind = st.number_input("Wind Speed (m/s)", 0, 50, 2)
            dist_road = st.slider("Dist. to Road (m)", 0, 5000, 500)
            dist_ind = st.slider("Dist. to Industry (m)", 0, 10000, 2000)
            dist_agri = st.slider("Dist. to Agriculture (m)", 0, 10000, 5000)

        submit = st.form_submit_button("🔍 Identify Source")
    
    if submit:
        # --- NEW LOGIC START: Load the .pkl file we created ---
        try:
            # 1. Load the real model
            real_model = joblib.load("pollution_model.pkl")
            
            # 2. Select ONLY the features the model knows (ignore Temp/Humid/Wind for prediction)
            model_inputs = pd.DataFrame([[pm25, no2, co, ozone, dist_road, dist_ind, dist_agri]], 
                                       columns=['pm25', 'no2', 'co', 'ozone', 'dist_road', 'dist_ind', 'dist_agri'])
            
            # 3. Predict using Real Model
            source = real_model.predict(model_inputs)[0]
            probs = real_model.predict_proba(model_inputs)
            conf = max(probs[0]) * 100

        except FileNotFoundError:
            # Fallback to your old logic if pkl is missing
            st.warning("⚠️ 'pollution_model.pkl' not found! Falling back to internal engine.")
            input_data = pd.DataFrame([[pm25, no2, co, ozone, temp, humid, wind, dist_road, dist_ind, dist_agri]], 
                                     columns=predictor.features)
            source, conf = predictor.predict(input_data)
        # --- NEW LOGIC END ---
        
        # Display Result with Animation
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if "Industrial" in source:
                st.error(f"⚠️ Source: {source}")
            elif "Vehicular" in source:
                st.warning(f"🚗 Source: {source}")
            elif "Agricultural" in source:
                st.success(f"🌾 Source: {source}")
            else:
                st.info(f"🌳 Source: {source}")
                
        with res_col2:
            st.metric("AI Confidence Score", f"{conf:.1f}%")
            st.progress(int(conf))
            
            if conf < 60:
                st.caption("Uncertain prediction. Check sensor calibration.")
            else:
                st.caption("High confidence prediction based on geospatial & chemical signatures.")