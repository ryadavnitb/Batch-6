"""
🌍 EnviroScan: AI-Powered Pollution Source Identifier Dashboard
================================================================

A comprehensive Streamlit dashboard for real-time air quality monitoring,
pollution source prediction, and environmental alerts.

Features:
- Real-time air quality data from OpenAQ API v3
- ML-powered pollution source prediction (92.26% accuracy)
- Interactive maps with heatmap overlays
- Real-time alerts for critical pollution levels
- Email alerts for critical conditions
- Downloadable reports (CSV/PDF/JSON)

Author: Praveen S
Date: December 2025
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import warnings
import io
import base64

warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EnviroScan - AI Pollution Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING (Dark Theme + Responsive)
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    
    /* Responsive font sizing */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        p, span { font-size: 0.9rem; }
    }
    
    /* Metric cards - responsive */
    .stMetric {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    @media (max-width: 768px) {
        .stMetric {
            padding: 10px;
        }
        .stMetric label {
            font-size: 0.8rem !important;
        }
    }
    
    /* Alert boxes - responsive with proper icons */
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse 2s infinite;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 1rem;
    }
    
    .alert-warning {
        background-color: #ff8800;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 1rem;
    }
    
    .alert-good {
        background-color: #00c853;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 1rem;
    }
    
    @media (max-width: 768px) {
        .alert-critical, .alert-warning, .alert-good {
            padding: 12px 15px;
            font-size: 0.9rem;
        }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Custom card styling - responsive */
    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 10px 0;
    }
    
    @media (max-width: 768px) {
        .info-card {
            padding: 15px;
        }
    }
    
    /* Button styling - responsive */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        color: #0e1117;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        transform: scale(1.05);
    }
    
    @media (max-width: 768px) {
        .stButton > button {
            padding: 8px 20px;
            font-size: 0.9rem;
        }
    }
    
    /* Download buttons - responsive grid */
    .stDownloadButton {
        width: 100%;
    }
    
    .stDownloadButton > button {
        width: 100%;
        font-size: 0.9rem;
        padding: 8px 12px;
    }
    
    @media (max-width: 768px) {
        .stDownloadButton > button {
            font-size: 0.8rem;
            padding: 6px 10px;
        }
    }
    
    /* Emoji/Icon sizing fix */
    .emoji-icon {
        font-size: 1.5rem;
        vertical-align: middle;
    }
    
    @media (max-width: 768px) {
        .emoji-icon {
            font-size: 1.2rem;
        }
    }
    
    /* Source tags */
    .source-tag {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
        margin: 5px;
    }
    
    .source-industrial { background-color: #e74c3c; color: white; }
    .source-vehicular { background-color: #3498db; color: white; }
    .source-agricultural { background-color: #27ae60; color: white; }
    .source-natural { background-color: #9b59b6; color: white; }
    .source-burning { background-color: #f39c12; color: white; }
    
    /* Plotly chart container - responsive */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* Map container - responsive */
    iframe {
        max-width: 100%;
    }
    
    /* Table responsiveness */
    .stDataFrame {
        overflow-x: auto;
    }
    
    /* Column layout fix for mobile */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Load API Keys
OPENAQ_API_KEY = os.getenv('OPENAQ_API_KEY', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')

# Request headers
HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'EnviroScan-Dashboard/2.0'
}
if OPENAQ_API_KEY:
    HEADERS['X-API-Key'] = OPENAQ_API_KEY

# Suggested Locations
SUGGESTED_LOCATIONS = {
    "Delhi, India": {"lat": 28.6139, "lon": 77.2090, "description": "Capital - High pollution"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "description": "Financial capital - Industrial"},
    "Chennai, India": {"lat": 13.0827, "lon": 80.2707, "description": "Southern metro - Port city"},
    "Bangalore, India": {"lat": 12.9716, "lon": 77.5946, "description": "IT hub - Vehicular"},
    "Kolkata, India": {"lat": 22.5726, "lon": 88.3639, "description": "Eastern metro - Dense urban"},
    "Hyderabad, India": {"lat": 17.3850, "lon": 78.4867, "description": "Tech city - Growing"},
    "Pune, India": {"lat": 18.5204, "lon": 73.8567, "description": "Industrial hub"},
    "Ahmedabad, India": {"lat": 23.0225, "lon": 72.5714, "description": "Industrial center"},
}

# Pollution thresholds (India NAAQS standards)
THRESHOLDS = {
    'pm25': {'good': 30, 'satisfactory': 60, 'moderate': 90, 'poor': 120, 'very_poor': 250},
    'pm10': {'good': 50, 'satisfactory': 100, 'moderate': 250, 'poor': 350, 'very_poor': 430},
    'no2': {'good': 40, 'satisfactory': 80, 'moderate': 180, 'poor': 280, 'very_poor': 400},
    'co': {'good': 1000, 'satisfactory': 2000, 'moderate': 10000, 'poor': 17000, 'very_poor': 34000},
    'so2': {'good': 40, 'satisfactory': 80, 'moderate': 380, 'poor': 800, 'very_poor': 1600},
    'o3': {'good': 50, 'satisfactory': 100, 'moderate': 168, 'poor': 208, 'very_poor': 748},
}

# Source colors
SOURCE_COLORS = {
    'Industrial': '#e74c3c',
    'Vehicular': '#3498db',
    'Agricultural': '#27ae60',
    'Natural': '#9b59b6',
    'Burning': '#f39c12'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_format(value, fmt=".2f", default="N/A"):
    """Safely format a value, returning default if not a number"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):
                return default
            return f"{value:{fmt}}"
        except:
            return default
    return default


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def get_aqi_category(pm25):
    """Get AQI category based on PM2.5 value"""
    if pm25 is None:
        return 'Unknown', '#808080', 0
    if pm25 <= 30:
        return 'Good', '#00e400', 1
    elif pm25 <= 60:
        return 'Satisfactory', '#92d050', 2
    elif pm25 <= 90:
        return 'Moderate', '#ffff00', 3
    elif pm25 <= 120:
        return 'Poor', '#ff7e00', 4
    elif pm25 <= 250:
        return 'Very Poor', '#ff0000', 5
    else:
        return 'Severe', '#8b0000', 6


def get_pollution_status(param, value):
    """Get pollution status for a parameter"""
    if param not in THRESHOLDS or value is None:
        return 'Unknown', '#808080'
    
    thresholds = THRESHOLDS[param]
    if value <= thresholds['good']:
        return 'Good', '#00e400'
    elif value <= thresholds['satisfactory']:
        return 'Satisfactory', '#92d050'
    elif value <= thresholds['moderate']:
        return 'Moderate', '#ffff00'
    elif value <= thresholds['poor']:
        return 'Poor', '#ff7e00'
    elif value <= thresholds['very_poor']:
        return 'Very Poor', '#ff0000'
    else:
        return 'Severe', '#8b0000'


# ============================================================================
# API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def geocode_location(query):
    """Geocode a location name to coordinates"""
    try:
        url = f"{NOMINATIM_BASE_URL}/search"
        params = {'q': query, 'format': 'json', 'limit': 1}
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        
        if response.status_code == 200:
            results = response.json()
            if results:
                return {
                    'lat': float(results[0]['lat']),
                    'lon': float(results[0]['lon']),
                    'name': results[0].get('display_name', query)
                }
        return None
    except:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_openaq_data(lat, lon, radius_km=25):
    """Fetch air quality data from OpenAQ API v3"""
    try:
        # Get locations
        url = f"{OPENAQ_BASE_URL}/locations"
        params = {
            'coordinates': f"{lat},{lon}",
            'radius': min(radius_km * 1000, 25000),  # Max 25km
            'limit': 100,
        }
        
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return {}, [], {'success': False, 'error': f"API Error: {response.status_code}"}
        
        locations = response.json().get('results', [])
        if not locations:
            return {}, [], {'success': False, 'error': 'No stations found'}
        
        # Fetch sensor data
        all_measurements = {}
        station_data = []
        param_map = {'pm25': 'pm25', 'pm2.5': 'pm25', 'pm10': 'pm10', 'no2': 'no2', 
                     'co': 'co', 'so2': 'so2', 'o3': 'o3'}
        
        for loc in locations[:15]:  # Check first 15 stations
            loc_id = loc.get('id')
            loc_name = loc.get('name', 'Unknown')
            coords = loc.get('coordinates', {})
            
            if not loc_id:
                continue
            
            station_info = {
                'id': loc_id,
                'name': loc_name,
                'lat': coords.get('latitude'),
                'lon': coords.get('longitude'),
                'distance_km': round(haversine_distance(lat, lon, 
                    coords.get('latitude', lat), coords.get('longitude', lon)) / 1000, 2),
                'data': {}
            }
            
            # Get sensors
            sensors_url = f"{OPENAQ_BASE_URL}/locations/{loc_id}/sensors"
            try:
                sensors_response = requests.get(sensors_url, headers=HEADERS, timeout=10)
                if sensors_response.status_code == 200:
                    sensors = sensors_response.json().get('results', [])
                    
                    for sensor in sensors:
                        param_info = sensor.get('parameter', {})
                        param_name = param_info.get('name', '').lower() if isinstance(param_info, dict) else ''
                        
                        for key, value in param_map.items():
                            if key in param_name:
                                latest = sensor.get('latest', {})
                                if isinstance(latest, dict) and latest.get('value') is not None:
                                    val = float(latest.get('value'))
                                    if value not in all_measurements:
                                        all_measurements[value] = []
                                    all_measurements[value].append(val)
                                    station_info['data'][value] = val
                                break
            except:
                continue
            
            if station_info['data']:
                station_data.append(station_info)
        
        # Aggregate measurements
        pollution_data = {k: np.mean(v) for k, v in all_measurements.items() if v}
        
        return pollution_data, station_data, {
            'success': True, 
            'stations': len(station_data),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {}, [], {'success': False, 'error': str(e)}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_osm_features(lat, lon, radius_m=5000):
    """Fetch geospatial features from OpenStreetMap"""
    features = {
        'roads_distance_m': 1000,
        'industrial_distance_m': 5000,
        'agricultural_distance_m': 5000,
        'dump_sites_distance_m': 10000
    }
    
    queries = {
        'roads': f'[out:json][timeout:15];(way["highway"~"motorway|trunk|primary"](around:{radius_m},{lat},{lon}););out center;',
        'industrial': f'[out:json][timeout:15];(way["landuse"="industrial"](around:{radius_m},{lat},{lon}););out center;',
    }
    
    for feature_type, query in queries.items():
        try:
            response = requests.post(OVERPASS_API_URL, data={'data': query}, timeout=20)
            if response.status_code == 200:
                elements = response.json().get('elements', [])
                if elements:
                    min_dist = float('inf')
                    for elem in elements:
                        elem_lat = elem.get('lat') or elem.get('center', {}).get('lat')
                        elem_lon = elem.get('lon') or elem.get('center', {}).get('lon')
                        if elem_lat and elem_lon:
                            dist = haversine_distance(lat, lon, elem_lat, elem_lon)
                            min_dist = min(min_dist, dist)
                    if min_dist != float('inf'):
                        if feature_type == 'roads':
                            features['roads_distance_m'] = round(min_dist, 2)
                        elif feature_type == 'industrial':
                            features['industrial_distance_m'] = round(min_dist, 2)
        except:
            continue
    
    return features


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    model_dir = 'models'
    try:
        model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        
        with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, label_encoder, feature_cols, model_info
    except:
        return None, None, None, None, None


def make_prediction(pollution_data, osm_features, model, scaler, label_encoder, feature_cols):
    """Make pollution source prediction"""
    if model is None:
        return None, None, None, None
    
    # Default values
    defaults = {
        'pm25': 35, 'pm10': 60, 'no2': 25, 'co': 500, 'so2': 10, 'o3': 40,
        'temperature': 28, 'humidity': 60, 'wind_speed': 3,
        'roads_distance_m': 1000, 'industrial_distance_m': 5000,
        'agricultural_distance_m': 5000, 'dump_sites_distance_m': 10000
    }
    
    features = {}
    for col in feature_cols:
        if col in pollution_data:
            features[col] = pollution_data[col]
        elif col in osm_features:
            features[col] = osm_features[col]
        else:
            features[col] = defaults.get(col, 0)
    
    feature_vector = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
    
    try:
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = model.predict(feature_vector_scaled)[0]
        source_name = label_encoder.inverse_transform([prediction])[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_vector_scaled)[0]
            confidence = float(max(probabilities) * 100)
            all_probs = {
                label_encoder.classes_[i]: float(prob * 100)
                for i, prob in enumerate(probabilities)
            }
        else:
            confidence = 75.0
            all_probs = {source_name: 75.0}
        
        return source_name, confidence, all_probs, features
    except:
        return None, None, None, None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_source_pie_chart(all_probs, predicted_source, confidence, location_name):
    """Create pollution source distribution pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=list(all_probs.keys()),
        values=list(all_probs.values()),
        hole=0.4,
        marker_colors=[SOURCE_COLORS.get(src, '#95a5a6') for src in all_probs.keys()],
        textinfo='percent',
        textposition='inside',
        insidetextorientation='horizontal',
        pull=[0.05 if v == max(all_probs.values()) else 0 for v in all_probs.values()]
    )])
    
    fig.update_layout(
        title=dict(text=f"Pollution Source Distribution", font=dict(size=16, color='white')),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(size=11, color='white'),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        height=400,
        margin=dict(l=20, r=120, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        annotations=[dict(text=f'{predicted_source}<br>{confidence:.0f}%' if isinstance(confidence, (int, float)) else predicted_source, 
                         x=0.5, y=0.5, font_size=12, showarrow=False, font_color='white')]
    )
    return fig


def create_pollutant_bar_chart(pollution_data, location_name):
    """Create pollutant levels bar chart"""
    pollutants = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    labels = ['PM2.5', 'PM10', 'NO₂', 'CO', 'SO₂', 'O₃']
    values = [(pollution_data.get(p, 0) or 0) for p in pollutants]
    
    colors = []
    for p, v in zip(pollutants, values):
        status, color = get_pollution_status(p, v)
        colors.append(color)
    
    # Calculate max value for proper y-axis range
    max_val = max(values) if values else 100
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[safe_format(v, '.1f', '0') for v in values],
        textposition='outside',
        textfont=dict(size=11, color='white'),
        cliponaxis=False
    )])
    
    fig.update_layout(
        title=dict(text="Pollutant Levels (µg/m³)", font=dict(size=16, color='white')),
        xaxis_title="Pollutant",
        yaxis_title="Concentration",
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333', tickfont=dict(size=12)),
        yaxis=dict(gridcolor='#333', range=[0, max_val * 1.2]),
        bargap=0.3
    )
    return fig


def create_aqi_gauge(pm25_value, location_name):
    """Create AQI gauge chart"""
    aqi_cat, aqi_color, _ = get_aqi_category(pm25_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pm25_value,
        title={'text': f"Air Quality Index", 'font': {'color': 'white'}},
        delta={'reference': 60, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        number={'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar': {'color': aqi_color},
            'bgcolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00e400'},
                {'range': [30, 60], 'color': '#92d050'},
                {'range': [60, 90], 'color': '#ffff00'},
                {'range': [90, 120], 'color': '#ff7e00'},
                {'range': [120, 250], 'color': '#ff0000'},
                {'range': [250, 300], 'color': '#8b0000'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig


def create_trend_chart(historical_data):
    """Create pollution trend chart"""
    if not historical_data:
        return None
    
    df = pd.DataFrame(historical_data)
    
    fig = go.Figure()
    
    for col in ['pm25', 'pm10', 'no2']:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[col],
                mode='lines+markers',
                name=col.upper(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Pollution Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Concentration (µg/m³)",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def create_map(lat, lon, station_data, pollution_data, location_name):
    """Create interactive Folium map with heatmap overlay"""
    # Create base map
    m = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )
    
    # Add center marker
    aqi_cat, aqi_color, _ = get_aqi_category(pollution_data.get('pm25', 0))
    
    # Safely format pollution values
    pm25_val = pollution_data.get('pm25')
    pm10_val = pollution_data.get('pm10')
    pm25_str = f"{pm25_val:.1f}" if isinstance(pm25_val, (int, float)) else "N/A"
    pm10_str = f"{pm10_val:.1f}" if isinstance(pm10_val, (int, float)) else "N/A"
    
    popup_html = f"""
    <div style="font-family: Arial; width: 200px;">
        <h4 style="color: #00d4ff;">{location_name}</h4>
        <p><b>AQI Status:</b> <span style="color: {aqi_color};">{aqi_cat}</span></p>
        <p><b>PM2.5:</b> {pm25_str} µg/m³</p>
        <p><b>PM10:</b> {pm10_str} µg/m³</p>
    </div>
    """
    
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    # Add station markers
    if station_data:
        marker_cluster = MarkerCluster().add_to(m)
        heat_data = []
        
        for station in station_data:
            if station.get('lat') and station.get('lon'):
                pm25 = station['data'].get('pm25', 0) or 0
                status, color = get_pollution_status('pm25', pm25)
                
                # Add to heatmap data
                heat_data.append([station['lat'], station['lon'], pm25/100 if pm25 else 0.1])
                
                # Station popup
                pm25_display = f"{pm25:.1f}" if isinstance(pm25, (int, float)) else "N/A"
                station_popup = f"""
                <div style="font-family: Arial;">
                    <b>{station['name']}</b><br>
                    Distance: {station['distance_km']} km<br>
                    PM2.5: {pm25_display} µg/m³<br>
                    Status: {status}
                </div>
                """
                
                folium.CircleMarker(
                    [station['lat'], station['lon']],
                    radius=8,
                    popup=station_popup,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
        
        # Add heatmap layer
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=25,
                blur=15,
                gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1: 'darkred'}
            ).add_to(m)
    
    return m


# ============================================================================
# REPORT GENERATION
# ============================================================================
def generate_report_csv(pollution_data, prediction_results, location_info, station_data):
    """Generate CSV report"""
    report_data = {
        'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Location': [location_info['name']],
        'Latitude': [location_info['lat']],
        'Longitude': [location_info['lon']],
        'PM2.5 (µg/m³)': [pollution_data.get('pm25', 'N/A')],
        'PM10 (µg/m³)': [pollution_data.get('pm10', 'N/A')],
        'NO2 (µg/m³)': [pollution_data.get('no2', 'N/A')],
        'CO (µg/m³)': [pollution_data.get('co', 'N/A')],
        'SO2 (µg/m³)': [pollution_data.get('so2', 'N/A')],
        'O3 (µg/m³)': [pollution_data.get('o3', 'N/A')],
        'Predicted Source': [prediction_results.get('source', 'N/A')],
        'Confidence (%)': [prediction_results.get('confidence', 'N/A')],
        'AQI Category': [prediction_results.get('aqi_category', 'N/A')],
        'Stations Analyzed': [len(station_data)]
    }
    
    df = pd.DataFrame(report_data)
    return df.to_csv(index=False)


def generate_detailed_report(pollution_data, prediction_results, location_info, station_data, osm_features):
    """Generate detailed text report"""
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENVIROSCAN - AIR QUALITY REPORT                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 LOCATION INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Location:    {location_info['name']}
    Coordinates: ({location_info['lat']:.4f}, {location_info['lon']:.4f})
    Stations:    {len(station_data)} monitoring stations analyzed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 POLLUTION DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PM2.5:   {safe_format(pollution_data.get('pm25'), '.2f')} µg/m³  [{get_pollution_status('pm25', pollution_data.get('pm25'))[0]}]
    PM10:    {safe_format(pollution_data.get('pm10'), '.2f')} µg/m³  [{get_pollution_status('pm10', pollution_data.get('pm10'))[0]}]
    NO2:     {safe_format(pollution_data.get('no2'), '.2f')} µg/m³  [{get_pollution_status('no2', pollution_data.get('no2'))[0]}]
    CO:      {safe_format(pollution_data.get('co'), '.2f')} µg/m³  [{get_pollution_status('co', pollution_data.get('co'))[0]}]
    SO2:     {safe_format(pollution_data.get('so2'), '.2f')} µg/m³  [{get_pollution_status('so2', pollution_data.get('so2'))[0]}]
    O3:      {safe_format(pollution_data.get('o3'), '.2f')} µg/m³  [{get_pollution_status('o3', pollution_data.get('o3'))[0]}]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 AI PREDICTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Predicted Source: {prediction_results.get('source', 'N/A')}
    Confidence:       {safe_format(prediction_results.get('confidence'), '.1f', 'N/A')}%
    AQI Category:     {prediction_results.get('aqi_category', 'N/A')}
    AQI Color:        {prediction_results.get('aqi_color', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🗺️ GEOSPATIAL FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Roads Distance:       {safe_format(osm_features.get('roads_distance_m'), '.0f')} meters
    Industrial Distance:  {safe_format(osm_features.get('industrial_distance_m'), '.0f')} meters
    Agricultural Distance:{safe_format(osm_features.get('agricultural_distance_m'), '.0f')} meters
    Dump Sites Distance:  {safe_format(osm_features.get('dump_sites_distance_m'), '.0f')} meters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    pm25 = pollution_data.get('pm25', 0) or 0
    if pm25 > 120:
        report += """
    ⚠️ CRITICAL: Air quality is very poor!
    - Avoid outdoor activities
    - Use air purifiers indoors
    - Wear N95 masks if going outside
    - Keep windows closed
"""
    elif pm25 > 60:
        report += """
    ⚠️ MODERATE: Air quality is concerning
    - Limit prolonged outdoor exertion
    - Sensitive groups should reduce outdoor activity
    - Monitor air quality updates
"""
    else:
        report += """
    ✅ GOOD: Air quality is acceptable
    - Normal outdoor activities can continue
    - Continue monitoring for changes
"""
    
    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        Report generated by EnviroScan AI
                   Developed by Praveen | © 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report


def generate_pdf_report(pollution_data, prediction_results, location_info, station_data, osm_features):
    """Generate a professional, modern PDF report using reportlab"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics import renderPDF
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch,
            leftMargin=0.6*inch,
            rightMargin=0.6*inch
        )
        elements = []
        styles = getSampleStyleSheet()
        page_width = A4[0] - inch  # Available width
        
        # Color scheme
        primary_color = colors.HexColor('#00d4ff')
        secondary_color = colors.HexColor('#1a1a2e')
        accent_color = colors.HexColor('#16213e')
        success_color = colors.HexColor('#4CAF50')
        warning_color = colors.HexColor('#FF9800')
        danger_color = colors.HexColor('#FF5252')
        dark_red = colors.HexColor('#8B0000')
        
        # Get PM2.5 value for AQI
        pm25_val = pollution_data.get('pm25', 0) if pollution_data else 0
        pm25_val = pm25_val if isinstance(pm25_val, (int, float)) else 0
        
        # Determine AQI status and color
        if pm25_val > 120:
            aqi_status = "Very Poor"
            aqi_color = dark_red
            health_level = "Severe"
        elif pm25_val > 90:
            aqi_status = "Poor"
            aqi_color = danger_color
            health_level = "Unhealthy"
        elif pm25_val > 60:
            aqi_status = "Moderate"
            aqi_color = warning_color
            health_level = "Moderate"
        elif pm25_val > 30:
            aqi_status = "Satisfactory"
            aqi_color = colors.HexColor('#FFEB3B')
            health_level = "Acceptable"
        else:
            aqi_status = "Good"
            aqi_color = success_color
            health_level = "Healthy"
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=5,
            alignment=TA_CENTER,
            textColor=primary_color,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.gray,
            spaceAfter=20
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=secondary_color,
            fontName='Helvetica-Bold',
            borderPadding=5
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            textColor=colors.HexColor('#333333'),
            leading=14
        )
        
        # ==================== HEADER SECTION ====================
        # Create header with gradient-like effect
        header_drawing = Drawing(page_width, 80)
        # Background rectangles for gradient effect
        header_drawing.add(Rect(0, 0, page_width, 80, fillColor=secondary_color, strokeColor=None))
        header_drawing.add(Rect(0, 0, page_width, 3, fillColor=primary_color, strokeColor=None))
        # Title text
        header_drawing.add(String(page_width/2, 50, "EnviroScan", fontName='Helvetica-Bold', fontSize=24, fillColor=primary_color, textAnchor='middle'))
        header_drawing.add(String(page_width/2, 30, "AI-Powered Air Quality Report", fontName='Helvetica', fontSize=12, fillColor=colors.white, textAnchor='middle'))
        header_drawing.add(String(page_width/2, 12, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", fontName='Helvetica', fontSize=9, fillColor=colors.gray, textAnchor='middle'))
        elements.append(header_drawing)
        elements.append(Spacer(1, 15))
        
        # ==================== LOCATION CARD ====================
        loc_name = location_info.get('name', 'Unknown Location') if location_info else 'Unknown'
        loc_lat = location_info.get('lat', 0) if location_info else 0
        loc_lon = location_info.get('lon', 0) if location_info else 0
        
        # Location info box
        location_drawing = Drawing(page_width, 50)
        location_drawing.add(Rect(0, 0, page_width, 50, fillColor=colors.HexColor('#e3f2fd'), strokeColor=colors.HexColor('#2196F3'), strokeWidth=1, rx=8, ry=8))
        location_drawing.add(String(15, 32, f"Location: {loc_name[:50]}", fontName='Helvetica-Bold', fontSize=12, fillColor=colors.HexColor('#1565C0')))
        location_drawing.add(String(15, 14, f"Coordinates: {loc_lat:.4f}, {loc_lon:.4f}  |  Stations Analyzed: {len(station_data) if station_data else 0}", fontName='Helvetica', fontSize=10, fillColor=colors.HexColor('#666666')))
        elements.append(location_drawing)
        elements.append(Spacer(1, 20))
        
        # ==================== AQI STATUS BANNER ====================
        aqi_banner = Drawing(page_width, 70)
        aqi_banner.add(Rect(0, 0, page_width, 70, fillColor=aqi_color, strokeColor=None, rx=10, ry=10))
        text_color = colors.white if pm25_val > 30 else colors.HexColor('#333333')
        aqi_banner.add(String(page_width/2, 48, "Air Quality Index", fontName='Helvetica', fontSize=11, fillColor=text_color, textAnchor='middle'))
        aqi_banner.add(String(page_width/2, 25, aqi_status.upper(), fontName='Helvetica-Bold', fontSize=22, fillColor=text_color, textAnchor='middle'))
        aqi_banner.add(String(page_width/2, 8, f"Health Level: {health_level}", fontName='Helvetica', fontSize=10, fillColor=text_color, textAnchor='middle'))
        elements.append(aqi_banner)
        elements.append(Spacer(1, 20))
        
        # ==================== POLLUTION DATA TABLE ====================
        elements.append(Paragraph("Pollution Levels", section_header_style))
        
        def fmt_val(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"
        
        def get_status_color(param, value):
            if not isinstance(value, (int, float)):
                return colors.gray
            if param == 'pm25':
                if value > 120: return dark_red
                elif value > 90: return danger_color
                elif value > 60: return warning_color
                elif value > 30: return colors.HexColor('#FFEB3B')
                else: return success_color
            elif param == 'pm10':
                if value > 250: return dark_red
                elif value > 150: return danger_color
                elif value > 100: return warning_color
                else: return success_color
            return colors.gray
        
        # Create pollution data with status indicators (using plain text for PDF compatibility)
        pollution_params = [
            ('pm25', 'PM2.5', pollution_data.get('pm25') if pollution_data else None),
            ('pm10', 'PM10', pollution_data.get('pm10') if pollution_data else None),
            ('no2', 'NO2', pollution_data.get('no2') if pollution_data else None),
            ('co', 'CO', pollution_data.get('co') if pollution_data else None),
            ('so2', 'SO2', pollution_data.get('so2') if pollution_data else None),
            ('o3', 'O3', pollution_data.get('o3') if pollution_data else None),
        ]
        
        # Two-column layout for pollution data
        pollution_table_data = [
            ['Pollutant', 'Value (µg/m³)', 'Pollutant', 'Value (µg/m³)'],
        ]
        
        for i in range(0, len(pollution_params), 2):
            row = []
            for j in range(2):
                if i + j < len(pollution_params):
                    param, label, value = pollution_params[i + j]
                    row.extend([label, fmt_val(value)])
                else:
                    row.extend(['', ''])
            pollution_table_data.append(row)
        
        pollution_table = Table(pollution_table_data, colWidths=[1.3*inch, 1.5*inch, 1.3*inch, 1.5*inch])
        pollution_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('BOX', (0, 0), (-1, -1), 1, secondary_color),
            # Alternating row colors
            ('BACKGROUND', (0, 2), (-1, 2), colors.white),
        ]))
        elements.append(pollution_table)
        elements.append(Spacer(1, 15))
        
        # ==================== POLLUTANT BAR CHART ====================
        elements.append(Paragraph("Pollutant Levels Chart", section_header_style))
        
        # Prepare data for bar chart
        chart_labels = []
        chart_values = []
        chart_colors_list = []
        
        for param, label, value in pollution_params:
            if isinstance(value, (int, float)) and value > 0:
                chart_labels.append(label)
                chart_values.append(value)
                color = get_status_color(param, value)
                chart_colors_list.append(color)
        
        if chart_values:
            # Create bar chart
            bar_chart_drawing = Drawing(page_width, 180)
            bar_chart_drawing.add(Rect(0, 0, page_width, 180, fillColor=colors.HexColor('#f8f9fa'), strokeColor=colors.HexColor('#dee2e6'), strokeWidth=1, rx=8, ry=8))
            
            bc = VerticalBarChart()
            bc.x = 60
            bc.y = 35
            bc.height = 110
            bc.width = page_width - 100
            bc.data = [chart_values]
            bc.strokeColor = None
            bc.barWidth = 35
            bc.groupSpacing = 15
            bc.barSpacing = 0
            
            # Color bars based on pollution level
            for i, color in enumerate(chart_colors_list):
                bc.bars[(0, i)].fillColor = color
            
            bc.valueAxis.valueMin = 0
            bc.valueAxis.valueMax = max(chart_values) * 1.2 if chart_values else 100
            bc.valueAxis.valueStep = max(chart_values) / 5 if chart_values else 20
            bc.valueAxis.labels.fontName = 'Helvetica'
            bc.valueAxis.labels.fontSize = 8
            bc.valueAxis.strokeColor = colors.HexColor('#666666')
            bc.valueAxis.gridStrokeColor = colors.HexColor('#e0e0e0')
            bc.valueAxis.visibleGrid = 1
            
            bc.categoryAxis.labels.boxAnchor = 'ne'
            bc.categoryAxis.labels.dx = 8
            bc.categoryAxis.labels.dy = -2
            bc.categoryAxis.labels.angle = 0
            bc.categoryAxis.labels.fontName = 'Helvetica-Bold'
            bc.categoryAxis.labels.fontSize = 9
            bc.categoryAxis.categoryNames = chart_labels
            bc.categoryAxis.strokeColor = colors.HexColor('#666666')
            
            bar_chart_drawing.add(bc)
            
            # Add title
            bar_chart_drawing.add(String(page_width/2, 160, "Pollutant Concentrations (ug/m3)", fontName='Helvetica-Bold', fontSize=11, fillColor=colors.HexColor('#333333'), textAnchor='middle'))
            
            elements.append(bar_chart_drawing)
            elements.append(Spacer(1, 20))
        
        # Page break before AI section for better layout
        elements.append(PageBreak())
        
        # ==================== AI PREDICTION SECTION ====================
        elements.append(Paragraph("AI Prediction Results", section_header_style))
        
        pred_source = prediction_results.get('source', 'N/A') if prediction_results else 'N/A'
        pred_confidence = prediction_results.get('confidence', 0) if prediction_results else 0
        pred_confidence = pred_confidence if isinstance(pred_confidence, (int, float)) else 0
        aqi_cat = prediction_results.get('aqi_category', 'N/A') if prediction_results else 'N/A'
        
        # Create prediction cards
        pred_drawing = Drawing(page_width, 60)
        card_width = (page_width - 20) / 2
        
        # Source card (purple gradient)
        pred_drawing.add(Rect(0, 0, card_width, 60, fillColor=colors.HexColor('#667eea'), strokeColor=None, rx=8, ry=8))
        pred_drawing.add(String(card_width/2, 42, "Predicted Source", fontName='Helvetica', fontSize=10, fillColor=colors.HexColor('#ffffff99'), textAnchor='middle'))
        pred_drawing.add(String(card_width/2, 18, str(pred_source), fontName='Helvetica-Bold', fontSize=16, fillColor=colors.white, textAnchor='middle'))
        
        # Confidence card (pink gradient)
        pred_drawing.add(Rect(card_width + 20, 0, card_width, 60, fillColor=colors.HexColor('#f093fb'), strokeColor=None, rx=8, ry=8))
        pred_drawing.add(String(card_width + 20 + card_width/2, 42, "Confidence Level", fontName='Helvetica', fontSize=10, fillColor=colors.HexColor('#ffffff99'), textAnchor='middle'))
        pred_drawing.add(String(card_width + 20 + card_width/2, 18, f"{pred_confidence:.1f}%", fontName='Helvetica-Bold', fontSize=16, fillColor=colors.white, textAnchor='middle'))
        
        elements.append(pred_drawing)
        elements.append(Spacer(1, 15))
        
        # Model info
        model_info_table = Table([
            ['Model', 'XGBoost Classifier'],
            ['Accuracy', '92.26%'],
            ['AQI Category', str(aqi_cat)]
        ], colWidths=[2*inch, 3.5*inch])
        model_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ]))
        elements.append(model_info_table)
        elements.append(Spacer(1, 20))
        
        # ==================== SOURCE DISTRIBUTION PIE CHART ====================
        elements.append(Paragraph("Pollution Source Distribution", section_header_style))
        
        # Source categories and their estimated contributions based on prediction
        source_categories = ['Vehicular', 'Industrial', 'Agricultural', 'Burning', 'Natural']
        
        # Generate distribution based on predicted source
        if pred_source and pred_source != 'N/A':
            # Calculate distribution with predicted source as dominant
            base_value = 10
            source_values = []
            source_colors_pie = []
            
            pie_colors = {
                'Vehicular': colors.HexColor('#FF6384'),
                'Industrial': colors.HexColor('#36A2EB'),
                'Agricultural': colors.HexColor('#4BC0C0'),
                'Burning': colors.HexColor('#FF9F40'),
                'Natural': colors.HexColor('#9966FF')
            }
            
            for cat in source_categories:
                if cat.lower() in str(pred_source).lower():
                    source_values.append(pred_confidence)
                else:
                    remaining = (100 - pred_confidence) / (len(source_categories) - 1)
                    source_values.append(remaining)
                source_colors_pie.append(pie_colors.get(cat, colors.gray))
        else:
            # Default equal distribution
            source_values = [20, 20, 20, 20, 20]
            source_colors_pie = [colors.HexColor('#FF6384'), colors.HexColor('#36A2EB'), 
                                colors.HexColor('#4BC0C0'), colors.HexColor('#FF9F40'), 
                                colors.HexColor('#9966FF')]
        
        # Create pie chart with legend (avoids label overlap)
        pie_drawing = Drawing(page_width, 160)
        pie_drawing.add(Rect(0, 0, page_width, 160, fillColor=colors.HexColor('#f8f9fa'), strokeColor=colors.HexColor('#dee2e6'), strokeWidth=1, rx=8, ry=8))
        
        pie = Pie()
        pie.x = 50
        pie.y = 25
        pie.width = 100
        pie.height = 100
        pie.data = source_values
        pie.labels = None  # Disable labels to avoid overlap
        pie.slices.strokeWidth = 2
        pie.slices.strokeColor = colors.white
        
        for i, color in enumerate(source_colors_pie):
            pie.slices[i].fillColor = color
        
        pie_drawing.add(pie)
        
        # Add title
        pie_drawing.add(String(page_width/2, 145, "Estimated Source Contribution (%)", fontName='Helvetica-Bold', fontSize=11, fillColor=colors.HexColor('#333333'), textAnchor='middle'))
        
        # Add legend on the right side
        legend_x = 200
        legend_y = 115
        for i, (cat, val, col) in enumerate(zip(source_categories, source_values, source_colors_pie)):
            # Color box
            pie_drawing.add(Rect(legend_x, legend_y - (i * 20), 12, 12, fillColor=col, strokeColor=None))
            # Label text
            pie_drawing.add(String(legend_x + 18, legend_y - (i * 20) + 2, f"{cat}: {val:.1f}%", fontName='Helvetica', fontSize=9, fillColor=colors.HexColor('#333333')))
        
        elements.append(pie_drawing)
        elements.append(Spacer(1, 20))
        
        # ==================== GEOSPATIAL FEATURES ====================
        elements.append(Paragraph("Geospatial Analysis", section_header_style))
        
        geo_data = [
            ['Feature', 'Distance', 'Impact'],
            ['Roads & Highways', f"{safe_format(osm_features.get('roads_distance_m') if osm_features else None, '.0f', 'N/A')} m", 'Traffic emissions'],
            ['Industrial Areas', f"{safe_format(osm_features.get('industrial_distance_m') if osm_features else None, '.0f', 'N/A')} m", 'Industrial pollution'],
            ['Agricultural Land', f"{safe_format(osm_features.get('agricultural_distance_m') if osm_features else None, '.0f', 'N/A')} m", 'Agricultural burning'],
            ['Waste Sites', f"{safe_format(osm_features.get('dump_sites_distance_m') if osm_features else None, '.0f', 'N/A')} m", 'Waste burning'],
        ]
        
        geo_table = Table(geo_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        geo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), accent_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('BOX', (0, 0), (-1, -1), 1, accent_color),
        ]))
        elements.append(geo_table)
        elements.append(Spacer(1, 20))
        
        # ==================== LOCATION MAP ====================
        elements.append(Paragraph("Location Map", section_header_style))
        
        try:
            # Generate static map using OpenStreetMap
            from reportlab.platypus import Image
            import urllib.request
            import tempfile
            
            # Create a static map URL (using OpenStreetMap's static map service)
            map_zoom = 12
            map_width = 400
            map_height = 200
            
            # Use OpenStreetMap static map
            map_url = f"https://staticmap.openstreetmap.de/staticmap.php?center={loc_lat},{loc_lon}&zoom={map_zoom}&size={map_width}x{map_height}&markers={loc_lat},{loc_lon},red-pushpin"
            
            # Create a temporary file for the map
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                urllib.request.urlretrieve(map_url, tmp_file.name)
                map_image = Image(tmp_file.name, width=page_width*0.9, height=150)
                
                # Create map container with border
                map_container = Drawing(page_width, 170)
                map_container.add(Rect(0, 0, page_width, 170, fillColor=colors.HexColor('#f8f9fa'), strokeColor=colors.HexColor('#dee2e6'), strokeWidth=1, rx=8, ry=8))
                elements.append(map_container)
                elements.append(Spacer(1, -165))
                elements.append(map_image)
                
        except Exception as map_error:
            # If map fails, show a placeholder with coordinates
            map_placeholder = Drawing(page_width, 80)
            map_placeholder.add(Rect(0, 0, page_width, 80, fillColor=colors.HexColor('#e3f2fd'), strokeColor=colors.HexColor('#2196F3'), strokeWidth=1, rx=8, ry=8))
            map_placeholder.add(String(page_width/2, 50, "Map Location", fontName='Helvetica-Bold', fontSize=12, fillColor=colors.HexColor('#1565C0'), textAnchor='middle'))
            map_placeholder.add(String(page_width/2, 30, f"Latitude: {loc_lat:.4f}  |  Longitude: {loc_lon:.4f}", fontName='Helvetica', fontSize=10, fillColor=colors.HexColor('#666666'), textAnchor='middle'))
            map_placeholder.add(String(page_width/2, 12, "View interactive map on EnviroScan Dashboard", fontName='Helvetica', fontSize=9, fillColor=colors.gray, textAnchor='middle'))
            elements.append(map_placeholder)
        
        elements.append(Spacer(1, 20))
        
        # ==================== HEALTH RECOMMENDATIONS ====================
        elements.append(Paragraph("Health Recommendations", section_header_style))
        
        if pm25_val > 120:
            rec_color = dark_red
            rec_icon = "CRITICAL"
            recommendations = [
                "• Avoid all outdoor physical activities",
                "• Keep all windows and doors closed",
                "• Use air purifiers with HEPA filters",
                "• Wear N95 masks if going outside is unavoidable",
                "• Seek medical attention if experiencing respiratory issues"
            ]
        elif pm25_val > 90:
            rec_color = danger_color
            rec_icon = "WARNING"
            recommendations = [
                "• Limit prolonged outdoor activities",
                "• Sensitive groups should stay indoors",
                "• Use air purifiers indoors",
                "• Consider wearing masks outdoors",
                "• Monitor air quality updates regularly"
            ]
        elif pm25_val > 60:
            rec_color = warning_color
            rec_icon = "CAUTION"
            recommendations = [
                "• Reduce prolonged outdoor exertion",
                "• Sensitive individuals should limit outdoor activities",
                "• Keep windows closed during peak hours",
                "• Consider using air purifiers",
                "• Stay hydrated and take breaks"
            ]
        else:
            rec_color = success_color
            rec_icon = "GOOD"
            recommendations = [
                "• Air quality is acceptable for outdoor activities",
                "• Enjoy outdoor exercise and activities",
                "• Good time for ventilating indoor spaces",
                "• Continue monitoring air quality",
                "• Stay healthy and active"
            ]
        
        # Recommendation banner
        rec_banner = Drawing(page_width, 30)
        rec_banner.add(Rect(0, 0, page_width, 30, fillColor=rec_color, strokeColor=None, rx=5, ry=5))
        text_col = colors.white if pm25_val > 30 else colors.HexColor('#333333')
        rec_banner.add(String(page_width/2, 10, f"STATUS: {rec_icon}", fontName='Helvetica-Bold', fontSize=14, fillColor=text_col, textAnchor='middle'))
        elements.append(rec_banner)
        elements.append(Spacer(1, 10))
        
        # Recommendations list
        for rec in recommendations:
            elements.append(Paragraph(rec, body_style))
        
        elements.append(Spacer(1, 25))
        
        # ==================== FOOTER ====================
        footer_line = Drawing(page_width, 2)
        footer_line.add(Line(0, 1, page_width, 1, strokeColor=colors.HexColor('#dee2e6'), strokeWidth=1))
        elements.append(footer_line)
        elements.append(Spacer(1, 10))
        
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9, textColor=colors.gray)
        elements.append(Paragraph("Report generated by EnviroScan AI | Model Accuracy: 92.26%", footer_style))
        elements.append(Paragraph("Developed by Praveen | © 2026 All Rights Reserved", footer_style))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError as e:
        # Fallback if reportlab is not available
        return None
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


# ============================================================================
# EMAIL ALERT FUNCTIONS
# ============================================================================
def send_email_alert(to_email, subject, message, location_data=None, pollution_data=None, prediction_data=None, smtp_config=None):
    """
    Send email alert for critical pollution levels
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        message: Alert message body (or 'test_alert' for test notification)
        location_data: Dict with location info (name, lat, lon)
        pollution_data: Dict with pollution readings
        prediction_data: Dict with AI prediction results
        smtp_config: Optional SMTP configuration dict
    
    Returns:
        tuple: (success: bool, message: str)
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Get SMTP config from environment or provided config
    smtp_server = smtp_config.get('server') if smtp_config else os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = smtp_config.get('port') if smtp_config else int(os.getenv('SMTP_PORT', 587))
    smtp_user = smtp_config.get('user') if smtp_config else os.getenv('SMTP_USER', '')
    smtp_password = smtp_config.get('password') if smtp_config else os.getenv('SMTP_PASSWORD', '')
    
    if not all([smtp_user, smtp_password]):
        return False, "Email credentials not configured in environment"
    
    if not to_email:
        return False, "Please enter a valid email address"
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"EnviroScan <{smtp_user}>"
        msg['To'] = to_email
        
        # Determine if this is a test alert or real alert
        is_test = message == 'test_alert'
        current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        # Extract location info
        loc_name = location_data.get('name', 'Unknown Location') if location_data else 'Unknown Location'
        loc_lat = location_data.get('lat', 0) if location_data else 0
        loc_lon = location_data.get('lon', 0) if location_data else 0
        
        # Extract pollution data
        pm25 = pollution_data.get('pm25') if pollution_data else None
        pm10 = pollution_data.get('pm10') if pollution_data else None
        no2 = pollution_data.get('no2') if pollution_data else None
        co = pollution_data.get('co') if pollution_data else None
        so2 = pollution_data.get('so2') if pollution_data else None
        o3 = pollution_data.get('o3') if pollution_data else None
        
        # Extract prediction data
        pred_source = prediction_data.get('source', 'N/A') if prediction_data else 'N/A'
        pred_confidence = prediction_data.get('confidence', 0) if prediction_data else 0
        aqi_category = prediction_data.get('aqi_category', 'N/A') if prediction_data else 'N/A'
        
        # Format pollution values
        def fmt_val(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"
        
        # Determine AQI color based on PM2.5
        pm25_val = pm25 if isinstance(pm25, (int, float)) else 0
        if pm25_val > 120:
            aqi_color = "#8B0000"  # Dark red - Very Poor
            aqi_status = "Very Poor"
            health_msg = "Health alert: Everyone may experience serious health effects. Avoid all outdoor activities."
        elif pm25_val > 90:
            aqi_color = "#FF5252"  # Red - Poor
            aqi_status = "Poor"
            health_msg = "Health warning: Sensitive groups should avoid outdoor activities. Others should limit prolonged outdoor exertion."
        elif pm25_val > 60:
            aqi_color = "#FF9800"  # Orange - Moderate
            aqi_status = "Moderate"
            health_msg = "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion."
        elif pm25_val > 30:
            aqi_color = "#FFEB3B"  # Yellow - Satisfactory
            aqi_status = "Satisfactory"
            health_msg = "Air quality is satisfactory. Enjoy outdoor activities."
        else:
            aqi_color = "#4CAF50"  # Green - Good
            aqi_status = "Good"
            health_msg = "Air quality is excellent. Perfect for outdoor activities!"
        
        if is_test:
            alert_title = "Test Air Quality Report"
            alert_color = "#2196F3"  # Blue for test
            alert_icon = "📊"
            banner_text = "This is a test report with real-time data from your selected location."
        else:
            alert_title = "Air Quality Alert"
            alert_color = "#FF5252"  # Red for real alerts
            alert_icon = "🚨"
            banner_text = message
        
        # Plain text version
        plain_text = f"""EnviroScan Air Quality Report
{'='*50}

📍 Location: {loc_name}
📐 Coordinates: ({loc_lat:.4f}, {loc_lon:.4f})
🕐 Time: {current_time}

📊 POLLUTION DATA
{'-'*30}
PM2.5: {fmt_val(pm25)} µg/m³
PM10: {fmt_val(pm10)} µg/m³
NO₂: {fmt_val(no2)} µg/m³
CO: {fmt_val(co)} µg/m³
SO₂: {fmt_val(so2)} µg/m³
O₃: {fmt_val(o3)} µg/m³

🎯 AI PREDICTION
{'-'*30}
Predicted Source: {pred_source}
Confidence: {pred_confidence:.1f}%
AQI Status: {aqi_status}

⚠️ HEALTH ADVISORY
{'-'*30}
{health_msg}

📋 RECOMMENDATIONS
{'-'*30}
• Monitor air quality levels regularly
• Limit outdoor activities during high pollution
• Use air purifiers indoors if available
• Wear N95 masks when outdoors in poor air quality

{'='*50}
© 2026 by Praveen | EnviroScan AI
"""
        
        # Professional HTML Email Template with data
        html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f4;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 0;">
                <table role="presentation" style="width: 600px; border-collapse: collapse; background-color: #ffffff; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
                            <h1 style="color: #00d4ff; margin: 0; font-size: 28px;">🌍 EnviroScan</h1>
                            <p style="color: #888; margin: 10px 0 0 0; font-size: 14px;">AI-Powered Environmental Monitoring</p>
                        </td>
                    </tr>
                    
                    <!-- Alert Banner -->
                    <tr>
                        <td style="padding: 0 30px;">
                            <div style="background-color: {alert_color}; color: white; padding: 20px; border-radius: 12px; margin-top: 25px; text-align: center;">
                                <span style="font-size: 32px;">{alert_icon}</span>
                                <h2 style="margin: 10px 0 5px 0; font-size: 22px;">{alert_title}</h2>
                                <p style="margin: 0; font-size: 14px; opacity: 0.9;">{current_time}</p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Location Info -->
                    <tr>
                        <td style="padding: 25px 30px 15px 30px;">
                            <div style="background-color: #e3f2fd; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #2196F3;">
                                <p style="margin: 0; color: #1565C0; font-size: 14px;">📍 <strong>{loc_name}</strong></p>
                                <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">Coordinates: {loc_lat:.4f}, {loc_lon:.4f}</p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Pollution Data Table -->
                    <tr>
                        <td style="padding: 10px 30px;">
                            <h3 style="color: #1a1a2e; margin: 0 0 15px 0; font-size: 16px;">📊 Pollution Levels</h3>
                            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                                <tr style="background-color: #f8f9fa;">
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>PM2.5</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(pm25)} µg/m³</td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>PM10</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(pm10)} µg/m³</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>NO₂</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(no2)} µg/m³</td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>CO</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(co)} µg/m³</td>
                                </tr>
                                <tr style="background-color: #f8f9fa;">
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>SO₂</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(so2)} µg/m³</td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0;"><strong>O₃</strong></td>
                                    <td style="padding: 12px; border: 1px solid #e0e0e0; text-align: right;">{fmt_val(o3)} µg/m³</td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- AI Prediction -->
                    <tr>
                        <td style="padding: 15px 30px;">
                            <h3 style="color: #1a1a2e; margin: 0 0 15px 0; font-size: 16px;">🎯 AI Prediction</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px 0 0 10px; text-align: center; width: 50%;">
                                        <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 12px;">Predicted Source</p>
                                        <p style="margin: 5px 0 0 0; color: white; font-size: 18px; font-weight: bold;">{pred_source}</p>
                                    </td>
                                    <td style="padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 0 10px 10px 0; text-align: center; width: 50%;">
                                        <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 12px;">Confidence</p>
                                        <p style="margin: 5px 0 0 0; color: white; font-size: 18px; font-weight: bold;">{pred_confidence:.1f}%</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- AQI Status -->
                    <tr>
                        <td style="padding: 15px 30px;">
                            <div style="background-color: {aqi_color}; color: {'white' if pm25_val > 60 else '#333'}; padding: 20px; border-radius: 12px; text-align: center;">
                                <p style="margin: 0; font-size: 14px; opacity: 0.9;">Air Quality Index</p>
                                <h2 style="margin: 10px 0; font-size: 28px;">{aqi_status}</h2>
                                <p style="margin: 0; font-size: 13px;">{health_msg}</p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Recommendations -->
                    <tr>
                        <td style="padding: 15px 30px 25px 30px;">
                            <h3 style="color: #1a1a2e; margin: 0 0 15px 0; font-size: 16px;">📋 Recommendations</h3>
                            <ul style="margin: 0; padding-left: 20px; color: #555; line-height: 1.8;">
                                <li>Monitor air quality levels regularly</li>
                                <li>Limit outdoor activities during high pollution</li>
                                <li>Use air purifiers indoors if available</li>
                                <li>Wear N95 masks when outdoors in poor air quality</li>
                                <li>Keep windows closed during peak pollution hours</li>
                            </ul>
                        </td>
                    </tr>
                    
                    <!-- Divider -->
                    <tr>
                        <td style="padding: 0 30px;">
                            <hr style="border: none; border-top: 1px solid #eee; margin: 0;">
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 25px 30px; text-align: center;">
                            <p style="margin: 0 0 10px 0; color: #888; font-size: 12px;">Powered by EnviroScan AI | Model Accuracy: 92.26%</p>
                            <p style="margin: 0; color: #aaa; font-size: 11px;">© 2026 by Praveen | All Rights Reserved</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
        """
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Failed: {str(e)}"


def process_critical_alerts(alerts, location_name, email=None, enable_email=False):
    """
    Process critical alerts and send email notifications
    
    Returns:
        list: Notification results
    """
    results = []
    critical_alerts = [a for a in alerts if a['level'] == 'critical']
    
    if not critical_alerts:
        return results
    
    # Build alert message
    alert_messages = [a['message'] for a in critical_alerts]
    full_message = f"Critical Air Quality Alert for {location_name}:\n" + "\n".join(alert_messages)
    
    # Send email if enabled
    if enable_email and email:
        success, msg = send_email_alert(
            email,
            f"🚨 CRITICAL: Air Quality Alert - {location_name}",
            full_message
        )
        results.append({'type': 'email', 'success': success, 'message': msg})
    
    return results


# ============================================================================
# ALERT FUNCTIONS
# ============================================================================
def check_alerts(pollution_data):
    """Check pollution levels and generate alerts"""
    alerts = []
    
    for param, value in pollution_data.items():
        if param in THRESHOLDS and value is not None and isinstance(value, (int, float)):
            if value > THRESHOLDS[param]['very_poor']:
                alerts.append({
                    'level': 'critical',
                    'param': param.upper(),
                    'value': value,
                    'message': f"CRITICAL: {param.upper()} at {value:.1f} µg/m³ - SEVERE pollution!"
                })
            elif value > THRESHOLDS[param]['poor']:
                alerts.append({
                    'level': 'warning',
                    'param': param.upper(),
                    'value': value,
                    'message': f"WARNING: {param.upper()} at {value:.1f} µg/m³ - Poor air quality"
                })
    
    return alerts


def display_alerts(alerts):
    """Display alerts in the dashboard - consolidated by level"""
    if not alerts:
        st.markdown("""
        <div class="alert-good">
            ✅ <b>Air Quality Status: GOOD</b><br>
            All pollutant levels are within safe limits.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Separate alerts by level
    critical_alerts = [a for a in alerts if a['level'] == 'critical']
    warning_alerts = [a for a in alerts if a['level'] == 'warning']
    
    # Display critical alerts (consolidated)
    if critical_alerts:
        critical_messages = "<br>".join([f"• {a['message']}" for a in critical_alerts])
        st.markdown(f"""
        <div class="alert-critical">
            🚨 <b>CRITICAL AIR QUALITY ALERT</b><br><br>
            {critical_messages}<br><br>
            <i>Immediate action recommended. Avoid outdoor exposure.</i>
        </div>
        """, unsafe_allow_html=True)
    
    # Display warning alerts (consolidated)
    if warning_alerts:
        warning_messages = "<br>".join([f"• {a['message']}" for a in warning_alerts])
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ <b>AIR QUALITY WARNING</b><br><br>
            {warning_messages}<br><br>
            <i>Consider limiting outdoor activities.</i>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🌍 EnviroScan</h1>
        <h3 style="color: #00d4ff;">AI-Powered Pollution Source Identifier</h3>
        <p style="color: #888;">Real-time air quality monitoring & pollution source prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoder, feature_cols, model_info = load_model()
    
    if model is None:
        st.error("⚠️ ML Model not found! Please ensure model files are in the 'models/' directory.")
        st.stop()
    
    # Sidebar - Input Controls
    with st.sidebar:
        st.markdown("## 📍 Location Selection")
        
        input_method = st.radio(
            "Select input method:",
            ["🏙️ Choose City", "📍 Enter Coordinates", "🔍 Search Location"],
            index=0
        )
        
        lat, lon, location_name = None, None, None
        
        if input_method == "🏙️ Choose City":
            selected_city = st.selectbox(
                "Select a city:",
                list(SUGGESTED_LOCATIONS.keys()),
                index=0
            )
            loc_info = SUGGESTED_LOCATIONS[selected_city]
            lat, lon = loc_info['lat'], loc_info['lon']
            location_name = selected_city
            st.info(f"📍 {loc_info['description']}")
            
        elif input_method == "📍 Enter Coordinates":
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0)
            with col2:
                lon = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0)
            location_name = f"Custom ({lat:.4f}, {lon:.4f})"
            
        else:  # Search Location
            search_query = st.text_input("Enter location name:", "Hyderabad, India")
            if st.button("🔍 Search"):
                with st.spinner("Searching..."):
                    result = geocode_location(search_query)
                    if result:
                        st.session_state['geocode_result'] = result
                        st.success(f"Found: {result['name'][:50]}...")
                    else:
                        st.error("Location not found!")
            
            if 'geocode_result' in st.session_state:
                result = st.session_state['geocode_result']
                lat, lon = result['lat'], result['lon']
                location_name = search_query
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        search_radius = st.slider("Search Radius (km)", 5, 25, 25)
        
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        
        st.markdown("---")
        st.markdown("## � Alert Settings")
        
        enable_alerts = st.checkbox("Enable Alerts", value=True)
        
        if enable_alerts:
            alert_threshold = st.selectbox(
                "Alert when PM2.5 exceeds:",
                ["60 µg/m³ (Satisfactory)", "90 µg/m³ (Moderate)", "120 µg/m³ (Poor)"],
                index=1
            )
            
            st.markdown("### 📧 Email Notifications")
            
            enable_email_alerts = st.checkbox("Enable Email Alerts", value=False)
            if enable_email_alerts:
                alert_email = st.text_input("Email Address", placeholder="your@email.com", key="alert_email")
            
            if st.button("🧪 Test Email Alert", key="test_alerts"):
                if enable_email_alerts and st.session_state.get('alert_email', ''):
                    # Fetch real data for the test email
                    with st.spinner("Fetching data and sending test email..."):
                        # Get pollution data
                        test_pollution, test_stations, test_status = fetch_openaq_data(lat, lon, search_radius)
                        test_osm = fetch_osm_features(lat, lon)
                        
                        # Get prediction
                        test_source, test_conf, test_probs, _ = make_prediction(
                            test_pollution, test_osm, model, scaler, label_encoder, feature_cols
                        )
                        
                        # Get AQI category
                        pm25_v = test_pollution.get('pm25', 0) or 0
                        test_aqi, _, _ = get_aqi_category(pm25_v)
                        
                        # Prepare data for email
                        loc_data = {'name': location_name, 'lat': lat, 'lon': lon}
                        pred_data = {'source': test_source, 'confidence': test_conf, 'aqi_category': test_aqi}
                        
                        success, msg = send_email_alert(
                            st.session_state.get('alert_email', ''),
                            f"🌍 EnviroScan - Air Quality Report for {location_name}",
                            "test_alert",
                            location_data=loc_data,
                            pollution_data=test_pollution,
                            prediction_data=pred_data
                        )
                        
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Please enable email alerts and enter your email address.")
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        accuracy = model_info.get('test_accuracy', 0)
        accuracy_pct = accuracy * 100 if isinstance(accuracy, (int, float)) else 0
        st.markdown(f"**Accuracy:** {accuracy_pct:.1f}%")
        st.markdown(f"**Model:** {model_info.get('best_model_name', 'XGBoost')}")
    
    # Main content area
    if lat is None or lon is None:
        st.warning("Please select a location to begin analysis.")
        st.stop()
    
    # Fetch button
    if st.button("🔄 Fetch Air Quality Data", use_container_width=True):
        st.session_state['fetch_data'] = True
    
    if st.session_state.get('fetch_data', False) or auto_refresh:
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data
        status_text.text("📡 Fetching air quality data from OpenAQ...")
        progress_bar.progress(25)
        
        pollution_data, station_data, api_status = fetch_openaq_data(lat, lon, search_radius)
        
        progress_bar.progress(50)
        status_text.text("🗺️ Fetching geospatial features...")
        
        osm_features = fetch_osm_features(lat, lon)
        
        progress_bar.progress(75)
        status_text.text("🤖 Running AI prediction...")
        
        # Make prediction
        predicted_source, confidence, all_probs, features_used = make_prediction(
            pollution_data, osm_features, model, scaler, label_encoder, feature_cols
        )
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        # Store results in session state
        st.session_state['results'] = {
            'pollution_data': pollution_data,
            'station_data': station_data,
            'api_status': api_status,
            'osm_features': osm_features,
            'predicted_source': predicted_source,
            'confidence': confidence,
            'all_probs': all_probs,
            'features_used': features_used,
            'location': {'name': location_name, 'lat': lat, 'lon': lon}
        }
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        pollution_data = results['pollution_data']
        station_data = results['station_data']
        api_status = results['api_status']
        osm_features = results['osm_features']
        predicted_source = results['predicted_source']
        confidence = results['confidence']
        all_probs = results['all_probs']
        location_info = results['location']
        
        # API Status
        if api_status['success']:
            st.success(f"✅ Data fetched from {api_status['stations']} monitoring stations | Last updated: {api_status.get('timestamp', 'N/A')}")
        else:
            st.warning(f"⚠️ {api_status.get('error', 'Unknown error')}")
        
        # Alerts Section
        st.markdown("### 🚨 Real-Time Alerts")
        alerts = check_alerts(pollution_data)
        display_alerts(alerts)
        
        st.markdown("---")
        
        # Key Metrics Row
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        pm25_val = pollution_data.get('pm25', 0) or 0
        aqi_cat, aqi_color, _ = get_aqi_category(pm25_val)
        
        with col1:
            st.metric(
                label="🎯 Predicted Source",
                value=predicted_source or "N/A",
                delta=f"{confidence:.0f}% confidence" if isinstance(confidence, (int, float)) else None
            )
        
        with col2:
            st.metric(
                label="🌡️ PM2.5",
                value=f"{pm25_val:.1f} µg/m³" if isinstance(pm25_val, (int, float)) else "N/A",
                delta=f"{aqi_cat}",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                label="📍 Stations",
                value=len(station_data),
                delta="analyzed"
            )
        
        with col4:
            st.metric(
                label="🔍 Data Source",
                value="OpenAQ v3",
                delta="Real-time"
            )
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Source Distribution")
            if all_probs:
                fig_pie = create_source_pie_chart(all_probs, predicted_source, confidence, location_name)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Pollutant Levels")
            if pollution_data:
                fig_bar = create_pollutant_bar_chart(pollution_data, location_name)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Air Quality Index")
            fig_gauge = create_aqi_gauge(pm25_val, location_name)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Pollution Trend")
            # Simulated trend data (in production, fetch historical data)
            trend_data = []
            base_pm25 = pm25_val if pm25_val else 50
            for i in range(24):
                trend_data.append({
                    'timestamp': datetime.now() - timedelta(hours=23-i),
                    'pm25': base_pm25 + np.random.uniform(-20, 20),
                    'pm10': (pollution_data.get('pm10', 100)) + np.random.uniform(-30, 30),
                    'no2': (pollution_data.get('no2', 40)) + np.random.uniform(-10, 10)
                })
            fig_trend = create_trend_chart(trend_data)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        # Map Section
        st.markdown("### 🗺️ Interactive Map with Heatmap Overlay")
        
        map_obj = create_map(lat, lon, station_data, pollution_data, location_name)
        st_folium(map_obj, width=None, height=500)
        
        st.markdown("---")
        
        # Detailed Data Section
        with st.expander("📋 Detailed Station Data"):
            if station_data:
                station_rows = []
                for s in station_data:
                    pm25_val_s = s['data'].get('pm25')
                    pm10_val_s = s['data'].get('pm10')
                    no2_val_s = s['data'].get('no2')
                    station_rows.append({
                        'Station': str(s['name']),
                        'Distance (km)': float(s['distance_km']) if s['distance_km'] else 0.0,
                        'PM2.5': f"{pm25_val_s:.2f}" if isinstance(pm25_val_s, (int, float)) else 'N/A',
                        'PM10': f"{pm10_val_s:.2f}" if isinstance(pm10_val_s, (int, float)) else 'N/A',
                        'NO2': f"{no2_val_s:.2f}" if isinstance(no2_val_s, (int, float)) else 'N/A',
                    })
                df_stations = pd.DataFrame(station_rows)
                st.dataframe(df_stations, use_container_width=True)
            else:
                st.info("No station data available")
        
        # Download Section
        st.markdown("### 📥 Download Reports")
        
        col1, col2, col3, col4 = st.columns(4)
        
        prediction_results = {
            'source': predicted_source,
            'confidence': confidence,
            'aqi_category': aqi_cat,
            'aqi_color': aqi_color
        }
        
        with col1:
            csv_report = generate_report_csv(pollution_data, prediction_results, location_info, station_data)
            st.download_button(
                label="📄 CSV",
                data=csv_report,
                file_name=f"enviroscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            detailed_report = generate_detailed_report(
                pollution_data, prediction_results, location_info, station_data, osm_features
            )
            st.download_button(
                label="📝 TXT Report",
                data=detailed_report,
                file_name=f"enviroscan_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # JSON data export
            json_data = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'location': location_info,
                'pollution_data': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                                   for k, v in pollution_data.items()},
                'prediction': prediction_results,
                'stations': len(station_data)
            }, indent=2)
            st.download_button(
                label="📊 JSON",
                data=json_data,
                file_name=f"enviroscan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col4:
            # PDF report
            pdf_data = generate_pdf_report(pollution_data, prediction_results, location_info, station_data, osm_features)
            if pdf_data:
                st.download_button(
                    label="📕 PDF Report",
                    data=pdf_data,
                    file_name=f"enviroscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("Install reportlab for PDF: `pip install reportlab`")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p style="font-size: 1.2em;">🌍 <b>EnviroScan</b> - AI-Powered Environmental Monitoring</p>
        <p>Data sources: OpenAQ API v3 | OpenStreetMap | ML Model Accuracy: 92.26%</p>
        <p>Developed by Praveen S | © 2026</p>
        <p>Made with ❤️ for cleaner air</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
