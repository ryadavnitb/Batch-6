import streamlit as st
import pandas as pd
import requests
import joblib
import folium
import numpy as np
import datetime
from folium.plugins import HeatMap, Draw, Geocoder, Fullscreen, LocateControl
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import time

# ======================================================
# 1. CONFIGURATION & STYLING (FIXED DROPDOWN VISIBILITY)
# ======================================================
st.set_page_config(
    page_title="AI-EnviroScan Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Pro" UI (High Contrast)
st.markdown("""
<style>
    /* 1. Global Settings */
    .stApp { background-color: #f8f9fa; color: #333333 !important; }

    /* 2. Sidebar - Dark Theme */
    section[data-testid="stSidebar"] { background-color: #2c3e50 !important; }
    section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    
    /* 3. Inputs & Dropdowns - FORCE BLACK TEXT */
    /* This specifically targets the dropdown box text */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] > input, 
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #333333 !important;
        -webkit-text-fill-color: #333333 !important;
    }
    
    /* This targets the dropdown OPTIONS list (when you click it) */
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* 4. Headers - Dark Navy */
    h1, h2, h3, h4 { color: #1e3c72 !important; font-family: 'Helvetica Neue', sans-serif; }

    /* 5. Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    
    /* 6. Action Plan Card */
    .action-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-weight: 600;
        border-left: 5px solid;
    }

    /* 7. Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { color: #333333; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #e74c3c; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. CORE LOGIC
# ======================================================
API_KEY = "YOUR_API_KEY" 

@st.cache_resource
def load_models():
    try:
        model = joblib.load("randomforest_pollution_model.pkl")
        encoder = joblib.load("pollution_label_encoder.pkl")
        features = list(model.feature_names_in_)
        return model, encoder, features
    except:
        return None, None, []

model, encoder, MODEL_FEATURES = load_models()

# Session State
if "results" not in st.session_state: st.session_state.results = []
if "points" not in st.session_state: st.session_state.points = []
if "last_reading" not in st.session_state: st.session_state.last_reading = None
if "center_map" not in st.session_state: st.session_state.center_map = [20.5937, 78.9629]
if "zoom_level" not in st.session_state: st.session_state.zoom_level = 5

def get_health_advice(risk):
    if risk == "Critical": return ("⛔ CRITICAL ACTION", "Do not go outside. Keep windows closed. Run air purifiers on high.", "#e74c3c", "#fadbd8")
    elif risk == "High": return ("⚠️ WARNING", "Avoid outdoor exercise. Wear an N95 mask if travel is necessary.", "#e67e22", "#fdebd0")
    elif risk == "Moderate": return ("😷 CAUTION", "Sensitive groups (asthma/elderly) should limit outdoor exertion.", "#f1c40f", "#fcf3cf")
    else: return ("✅ SAFE", "Air quality is good. Enjoy outdoor activities.", "#27ae60", "#d4efdf")

def generate_trend_data(current_pm25, current_source):
    # Simulated trend logic for demo
    now = datetime.datetime.now()
    times = [(now - datetime.timedelta(hours=x)).strftime("%H:%M") for x in range(24, 0, -1)]
    values = [max(5, current_pm25 + np.random.randint(-25, 25)) for _ in range(24)]
    sources = ["Natural" if v < 35 else current_source for v in values]
    values[-1], sources[-1] = current_pm25, current_source
    return times, values, sources

def fetch_data_and_predict(lat, lon):
    try:
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric").json()
        p = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}").json()
        comp = p["list"][0]["components"]
        
        row = {
            "Temperature": w["main"]["temp"], "Humidity": w["main"]["humidity"],
            "PM2.5": comp["pm2_5"], "PM10": comp["pm10"], "NO2": comp["no2"], 
            "SO2": comp["so2"], "CO": comp["co"], "O3": comp["o3"],
            "Proximity_to_Industrial_Areas": 0, "roads": 0, "dump_sites": 0, "agricultural_fields": 0,
            "wind_speed": w["wind"]["speed"], "wind_direction": w["wind"]["deg"],
        }
        
        df_in = pd.DataFrame([row])
        for c in MODEL_FEATURES: 
            if c not in df_in.columns: df_in[c] = 0
                
        pred = model.predict(df_in[MODEL_FEATURES])[0]
        source = encoder.inverse_transform([pred])[0]
        confidence = np.max(model.predict_proba(df_in[MODEL_FEATURES])) * 100
        risk = "Critical" if comp["pm2_5"] > 150 else "High" if comp["pm2_5"] > 80 else "Moderate" if comp["pm2_5"] > 30 else "Safe"
        
        return row, source, risk, confidence
    except:
        return None, None, None, 0.0

def analyze_and_store(lat, lng):
    with st.spinner("🛰️ Analyzing location data..."):
        row_data, source, risk, confidence = fetch_data_and_predict(lat, lng)
        
        if row_data:
            trend_x, trend_y, trend_s = generate_trend_data(row_data["PM2.5"], source)
            
            new_entry = {
                "Latitude": lat, "Longitude": lng,
                "PM2.5": row_data["PM2.5"], "PM10": row_data["PM10"],
                "NO2": row_data["NO2"], "SO2": row_data["SO2"], "O3": row_data["O3"],
                "Pollution_Source": source, "Risk": risk, "Confidence": confidence,
                "Trend_X": trend_x, "Trend_Y": trend_y, "Trend_Source": trend_s,
                "Timestamp": pd.Timestamp.now()
            }
            st.session_state.results.append(new_entry)
            st.session_state.points.append([lat, lng, row_data["PM2.5"]])
            st.session_state.last_reading = new_entry
            st.session_state.center_map = [lat, lng]
            st.session_state.zoom_level = 12
            
            st.toast(f"Detected {source}", icon="✅")
            time.sleep(1)
            st.rerun()

# ======================================================
# 3. UI: SIDEBAR
# ======================================================
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.markdown("### 📍 Location Service")
    if st.button("🚀 Find My Location (Auto)", type="primary"):
        try:
            ip_info = requests.get('https://ipinfo.io/json').json()
            if 'loc' in ip_info:
                lat, lng = ip_info['loc'].split(',')
                analyze_and_store(float(lat), float(lng))
        except: st.error("Location Service Error.")

    st.markdown("---")
    st.markdown("### 🗺️ Map Settings")
    map_style = st.selectbox("Map Style", ["CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"])
    
    if st.button("Clear Session Data"):
        st.session_state.points = []
        st.session_state.results = []
        st.session_state.last_reading = None
        st.rerun()

# ======================================================
# 4. MAIN DASHBOARD
# ======================================================
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("# 🌍 AI-EnviroScan Pro")
    st.markdown("#### Real-time Pollution Source Identification & Analytics")

with col_head2:
    if st.session_state.last_reading:
        risk_val = st.session_state.last_reading.get('Risk', 'Unknown')
        color = "#e74c3c" if risk_val in ['High', 'Critical'] else "#27ae60"
        st.markdown(f'<div style="background-color:{color}; color:white; padding:10px; border-radius:5px; text-align:center;"><strong>Status:</strong> {risk_val.upper()}</div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ Geospatial Analysis", "📈 Trend Analytics"])

# === TAB 1: MAP & INTELLIGENCE HUB ===
with tab1:
    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        m = folium.Map(location=st.session_state.center_map, zoom_start=st.session_state.zoom_level, tiles=map_style)
        Geocoder().add_to(m)
        Fullscreen().add_to(m)
        LocateControl(auto_start=False).add_to(m)
        Draw(draw_options={"marker": False, "circle": False}).add_to(m)
        
        if st.session_state.points:
            HeatMap(st.session_state.points, radius=25, blur=15).add_to(m)
            for res in st.session_state.results:
                folium.CircleMarker(
                    location=[res['Latitude'], res['Longitude']], radius=6,
                    color='#e74c3c' if res.get('Risk') == 'High' else '#27ae60', fill=True,
                    popup=f"{res['Pollution_Source']}"
                ).add_to(m)
        map_data = st_folium(m, height=600, width="100%")

        if map_data and map_data.get("last_clicked"):
            lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            last = st.session_state.results[-1] if st.session_state.results else None
            if not last or (last['Latitude'] != lat):
                analyze_and_store(lat, lng)

    with col_stats:
        st.subheader("📡 Intelligence Hub")
        if st.session_state.last_reading:
            last = st.session_state.last_reading
            
            # 1. Metric Cards
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="metric-card"><div style="font-size:12px; color:gray;">SOURCE</div><div style="font-size:18px; font-weight:bold; color:#2980b9;">{last['Pollution_Source']}</div></div>""", unsafe_allow_html=True)
            with c2:
                c_code = "#e74c3c" if last['Risk'] in ['High', 'Critical'] else "#27ae60"
                st.markdown(f"""<div class="metric-card"><div style="font-size:12px; color:gray;">RISK</div><div style="font-size:18px; font-weight:bold; color:{c_code};">{last['Risk']}</div></div>""", unsafe_allow_html=True)

            # 2. Action Protocol
            title, msg, border_col, bg_col = get_health_advice(last['Risk'])
            st.markdown(f"""
            <div class="action-card" style="border-left-color: {border_col}; background-color: {bg_col}; color: #333;">
                <div style="font-size:14px; margin-bottom:5px;">{title}</div>
                <div style="font-size:16px;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

           # --- ADVANCED POLLUTANT COMPOSITION CHART ---
            st.markdown("### 🧪 Chemical Fingerprint")
            
            # 1. Define Categories & Values
            categories = ['PM2.5', 'PM10', 'NO2', 'SO2', 'Ozone']
            # Fetch real values or default to 0
            values = [
                last.get('PM2.5', 0), 
                last.get('PM10', 0), 
                last.get('NO2', 0), 
                last.get('SO2', 0), 
                last.get('O3', 0)
            ]
            
            # 2. Define Approximate Safe Limits (WHO/CPCB standards)
            safe_limits = [60, 100, 80, 80, 100]

            fig = go.Figure()

            # Layer 1: Actual Data (Filled Area)
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Level',
                line_color='#3498db',
                fillcolor='rgba(52, 152, 219, 0.3)'
            ))

            # Layer 2: Safe Limit Reference (Dashed Line)
            fig.add_trace(go.Scatterpolar(
                r=safe_limits,
                theta=categories,
                name='Safe Limit',
                line=dict(color='#27ae60', dash='dot'),
                hoverinfo='skip'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(max(values), 120)], gridcolor="#e0e0e0"),
                    angularaxis=dict(tickfont=dict(size=10, color="#333"))
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                height=250,
                margin=dict(l=30, r=30, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#333')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("ℹ️ **Blue Area:** Current Levels. **Green Dotted Line:** Safe Limits.")

        # else:
        #     st.info("👈 Click map to analyze.")

            # 4. Source Distribution (Pie Chart)
            if len(st.session_state.results) > 0:
                st.markdown("### 📈 Source Distribution")
                df_all = pd.DataFrame(st.session_state.results)
                fig_pie = px.pie(df_all, names='Pollution_Source', hole=0.4)
                fig_pie.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font=dict(color='#333'))
                st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.info("👈 Click map to analyze.")

# === TAB 2: TREND ANALYTICS ===
with tab2:
    if st.session_state.last_reading:
        last = st.session_state.last_reading
        st.subheader(f"📈 24-Hour Source Analysis")
        
        df_trend = pd.DataFrame({"Time": last['Trend_X'], "PM2.5": last['Trend_Y'], "Source": last['Trend_Source']})
        color_map = {"Industrial": "red", "Vehicular": "orange", "Biomass": "brown", "Natural": "green", "Safe": "green"}
        
        fig_trend = px.line(df_trend, x="Time", y="PM2.5", markers=True)
        fig_trend.update_traces(line_color='gray', line_width=1)
        
        # Custom Tooltip
        fig_trend.add_trace(go.Scatter(
            x=df_trend['Time'], 
            y=df_trend['PM2.5'], 
            mode='markers',
            marker=dict(size=10, color=[color_map.get(s, "blue") for s in df_trend['Source']]),
            text=df_trend['Source'], 
            hovertemplate="<b>Time:</b> %{x}<br><b>PM2.5:</b> %{y}<br><b>Source:</b> %{text}<extra></extra>", 
            name="Source Context"
        ))
        
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color='#333'), hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Please analyze a location first.")

# --- EXPORTS ---
with st.expander("📂 Download Detailed Report", expanded=False):
    if st.session_state.results:
        df_results = pd.DataFrame(st.session_state.results)
        df_export = df_results.drop(columns=['Trend_X', 'Trend_Y', 'Trend_Source'], errors='ignore')
        st.dataframe(df_export, use_container_width=True)
        st.download_button("⬇️ Download CSV Report", df_export.to_csv(index=False).encode('utf-8'), "enviroscan_report.csv", "text/csv")
    else:
        st.info("No data to export. Analyze a location first.")

