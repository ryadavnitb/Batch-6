import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
from geopy.distance import geodesic
import math
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import json
import joblib
@st.cache_resource
def load_xgb_model():
    return joblib.load("models/xgb_model.joblib")

xgb_model = load_xgb_model()

MODEL_FEATURES = [
    "pm2_5","pm10","no2","o3","co","so2","nh3",
    "temperature","humidity","wind_speed","wind_deg"
]

SEVERITY_MAP = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Severe"
}

def prepare_model_input(weather, aqi):
    row = {
        "pm2_5": aqi.get("pm2_5",0),
        "pm10": aqi.get("pm10",0),
        "no2": aqi.get("no2",0),
        "o3": aqi.get("o3",0),
        "co": aqi.get("co",0),
        "so2": aqi.get("so2",0),
        "nh3": aqi.get("nh3",0),
        "temperature": weather.get("temperature",0),
        "humidity": weather.get("humidity",0),
        "wind_speed": weather.get("wind_speed",0),
        "wind_deg": weather.get("wind_deg",0)
    }
    return pd.DataFrame([row])[MODEL_FEATURES]


@st.cache_data(ttl=300)
def fetch_weather_cached(lat, lon):
    return fetch_weather(lat, lon)

@st.cache_data(ttl=300)
def get_aqi_cached(lat, lon):
    return get_aqi_data(lat, lon)
def clean_text(text):
    if isinstance(text, str):
        return (
            text.replace("–", "-")
                .replace("—", "-")
                .replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
        )
    return text


# ====================== CONFIG ======================
OPENWEATHER_API_KEY = "4ea3c9aff2c80251a5f1397a2c14e1e5"
SOURCE_INDEX = ["Vehicular", "Industrial", "Agricultural", "Waste", "Natural"]
SOURCE_COLORS = {"Vehicular": "red", "Industrial": "purple", "Agricultural": "yellow",
                 "Waste": "orange", "Natural": "green"}
MAX_SOURCES_PER_TYPE = 5
HIST_ROWS = 5
FEATURES = ["pm2_5","pm10","no2","o3","co","so2","nh3","temperature","humidity","wind_speed"]
VISUAL_LEVELS = ["Global", "Country", "State"]

# ====================== UTILITY ======================
def safe_text(s):
    if isinstance(s, str):
        return s.encode("latin-1", "replace").decode("latin-1")
    return str(s)

# ====================== WEATHER & AQI ======================
def fetch_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=10).json()
    return {"temperature": r["main"]["temp"], "humidity": r["main"]["humidity"],
            "wind_speed": r["wind"]["speed"], "wind_deg": r["wind"].get("deg", 0)}

def fetch_openweather_aqi(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=10).json()
    c = r["list"][0]["components"]

    return {
        "pm2_5": c.get("pm2_5", 0),
        "pm10": c.get("pm10", 0),
        "no2": c.get("no2", 0),
        "o3": c.get("o3", 0),
        "co": c.get("co", 0),
        "so2": c.get("so2", 0),
        "nh3": c.get("nh3", 0),
    }

def get_aqi_data(lat, lon):
    return fetch_openweather_aqi(lat, lon)

# ====================== OSM SOURCES ======================
def fetch_osm_sources(lat, lon, source_type, radius=7000, min_area=5000):
    overpass = "https://overpass.kumi.systems/api/interpreter"
    queries = {
        "Vehicular": '''[out:json];node["highway"](around:{r},{lat},{lon});out;''',
        "Industrial": '''[out:json];way["landuse"="industrial"](around:{r},{lat},{lon});out center;''',
        "Agricultural": '''[out:json];way["landuse"~"farmland|orchard"](around:{r},{lat},{lon});out center;''',
        "Waste": '''[out:json];way["landuse"~"construction|landfill|quarry"](around:{r},{lat},{lon});out center;''',
        "Natural": '''[out:json];way["landuse"~"forest|grass|park"](around:{r},{lat},{lon});out center;'''
    }
    q = queries[source_type].format(r=radius, lat=lat, lon=lon)
    try:
        r = requests.post(overpass, data={"data": q}, timeout=60).json()
    except:
        return []

    sources = []
    for e in r.get("elements", []):
        lat_lon = None
        if "lat" in e and "lon" in e:
            lat_lon = (e["lat"], e["lon"])
        elif "center" in e:
            lat_lon = (e["center"]["lat"], e["center"]["lon"])
        
        # Skip small natural polygons
        if lat_lon:
            if source_type == "Natural" and "tags" in e and "area" in e["tags"]:
                if float(e["tags"]["area"]) < min_area:
                    continue
            sources.append({"lat": lat_lon[0], "lon": lat_lon[1], "source": source_type})
    return sources

# ====================== WIND FACTOR ======================
def get_wind_factor(p_lat, p_lon, s_lat, s_lon, wind_deg, wind_speed):
    dlon = s_lon - p_lon
    dlat = s_lat - p_lat
    bearing = math.degrees(math.atan2(dlon, dlat)) % 360
    diff = min(abs(bearing - wind_deg), 360 - abs(bearing - wind_deg))
    return ((math.cos(math.radians(diff)) + 1)/2) * max(0.1, wind_speed/10)

# ====================== CONTRIBUTIONS ======================
def calculate_contributions(lat, lon, sources_by_category, weather):
    wind_speed = weather["wind_speed"]
    wind_deg = weather["wind_deg"]
    scale = max(1000, wind_speed*3600)
    contributions = []
    weights = {cat:0 for cat in SOURCE_INDEX}

    for cat, sources in sources_by_category.items():
        for s in sources:
            aqi = get_aqi_data(s['lat'], s['lon'])
            d = geodesic((lat, lon), (s["lat"], s["lon"])).meters + 1
            weight = aqi["pm2_5"] * get_wind_factor(lat, lon, s["lat"], s["lon"], wind_deg, wind_speed) * np.exp(-d/scale)
            contributions.append({"lat": s["lat"], "lon": s["lon"], "source": cat, "weight": weight, "distance": d})
            weights[cat] += weight

    total_weight = sum(weights.values())
    for c in contributions:
        c["overall_pct"] = c["weight"]/total_weight*100 if total_weight else 0
        c["category_pct"] = c["weight"]/weights[c["source"]]*100 if weights[c["source"]] else 0

    category_pct = {cat: weights[cat]/total_weight*100 if total_weight else 0 for cat in SOURCE_INDEX}
    major = max(category_pct, key=category_pct.get)
    return major, category_pct, contributions

# ====================== HISTORICAL DATA ======================
def simulate_historical_data(lat, lon, n=HIST_ROWS):
    records=[]
    for _ in range(n):
        weather = fetch_weather(lat, lon)
        aqi = fetch_openweather_aqi(lat, lon)
        records.append({
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "wind_speed": weather["wind_speed"],
            "wind_deg": weather["wind_deg"],
            "pm2_5": aqi["pm2_5"],
            "no2": aqi["no2"],
            "co": aqi["co"],
            "o3": aqi["o3"],
            "Timestamp": pd.Timestamp.now()
        })
    return pd.DataFrame(records)


def draw_table(pdf, df, title, col_subset, col_width=26):
    pdf.add_page()
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, safe_txt(title), ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 7)
    for c in col_subset:
        pdf.cell(col_width, 6, safe_txt(c[:12]), border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 7)
    for _, row in df.iterrows():
        for c in col_subset:
            val = row[c]
            if isinstance(val, float):
                val = f"{val:.2f}"
            elif "timestamp" in c.lower():
                val = str(val)
            pdf.cell(col_width, 6, safe_txt(val)[:12], border=1)
        pdf.ln()



def safe_txt(x):
    return str(x).replace("–", "-").replace("—", "-").encode("latin-1", "ignore").decode("latin-1")

def create_pdf(df_location, df_sources, location):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # ---------------- SAFE TEXT ----------------
    def safe_txt(x):
        return str(x).replace("–", "-").replace("—", "-").encode("latin-1", "ignore").decode("latin-1")

    # ---------------- TITLE PAGE ----------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "Pollution Attribution & Severity Report", ln=True, align="C")
    pdf.ln(5)

    lat, lon = location
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, safe_txt(f"Selected Location: ({lat:.3f}, {lon:.3f})"), ln=True)
    pdf.ln(4)

    # ---------------- CLEAN DATA ----------------
    df_location = df_location.drop_duplicates().reset_index(drop=True)

    if df_sources is not None and not df_sources.empty:
        df_sources = df_sources.drop_duplicates().reset_index(drop=True)
    else:
        df_sources = None

    # ---------------- LOCATION SUMMARY ----------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Local Pollution Overview", ln=True)
    pdf.ln(3)

    loc_mean = df_location.mean(numeric_only=True)

    pm25 = loc_mean.get("pm2_5", 0)
    wind = loc_mean.get("wind_speed", 0)
    hum = loc_mean.get("humidity", 0)

    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(
        0,
        6,
        safe_txt(
            f"""
The average PM2.5 concentration at the selected location is {pm25:.2f} micrograms per cubic meter.
Wind speed averages {wind:.2f} meters per second, influencing dispersion of airborne pollutants.
Humidity levels around {hum:.1f}% can affect particulate suspension and secondary aerosol formation.
"""
        ).strip()
    )

    # ---------------- SOURCE ATTRIBUTION ----------------
    if df_sources is not None:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "2. Pollution Source Attribution", ln=True)
        pdf.ln(3)

        src_summary = (
            df_sources.groupby("source")
            .agg(
                overall_pct=("overall_pct", "mean"),
                avg_distance=("distance_m", "mean")
            )
            .sort_values("overall_pct", ascending=False)
        )

        dominant = src_summary.index[0]
        dom_pct = src_summary.iloc[0]["overall_pct"]

        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(
            0,
            6,
            safe_txt(
                f"""
The dominant pollution source category is {dominant}, contributing approximately
{dom_pct:.1f}% of the estimated pollution load.

The contribution model combines spatial proximity, pollutant intensity, and wind
direction influence. Sources closer to the location exert greater impact, while
wind dynamics amplify upwind emissions.
"""
            ).strip()
        )

        # ---------------- BAR CHART ----------------
        plt.figure(figsize=(4, 3))
        src_summary["overall_pct"].plot(kind="bar")
        plt.ylabel("Contribution (%)")
        plt.title("Average Source Contribution")
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.savefig(f.name, dpi=140)
            pdf.ln(4)
            pdf.image(f.name, w=120)
        plt.close()

        # ---------------- DISTANCE VS CONTRIBUTION ----------------
        plt.figure(figsize=(4, 3))
        plt.scatter(df_sources["distance_m"], df_sources["overall_pct"], alpha=0.6)
        plt.xlabel("Distance from Location (m)")
        plt.ylabel("Contribution (%)")
        plt.title("Distance vs Pollution Contribution")
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.savefig(f.name, dpi=140)
            pdf.ln(4)
            pdf.image(f.name, w=120)
        plt.close()

    # ---------------- CONCLUSION ----------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Conclusion", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(
        0,
        6,
        safe_txt(
            """
This report demonstrates a hybrid pollution assessment framework integrating
machine learning-based severity prediction with spatial source attribution.

The system combines real-time air quality data, meteorological conditions,
geographical proximity, and OpenStreetMap-derived land-use information to
produce interpretable pollution insights.

Although temporal variation is limited due to short sampling intervals, the
approach provides meaningful, explainable, and scalable pollution intelligence.
"""
        ).strip()
    )

    return pdf.output(dest="S").encode("latin-1", "ignore")



def safe_style(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)
    return df


# ====================== STREAMLIT UI ======================
st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Pollution Attribution", "Historical Data"])

# ----------------- PAGE 1: DASHBOARD -----------------
if page == "Dashboard":
    st.title("📊 Pollution Dashboard")
    st.caption("Select a parameter to visualize. Zoom from Global → Country → State")

    selected_feature = st.selectbox("Select Feature (None by default)", [None] + FEATURES)
    level = st.selectbox("Select Level of Visualization", ["Global", "Country", "State"])

    @st.cache_data
    def load_static_data():
        df = pd.read_csv("pollution_with_state.csv")

        numeric_cols = ["Latitude", "Longitude"] + FEATURES
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Country", "Latitude", "Longitude"])
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    df_static = load_static_data()
    map_dashboard = folium.Map(location=[20, 0], zoom_start=2)

    if selected_feature:
        # ---------------- GEOJSON ----------------
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = json.loads(requests.get(geojson_url).text)

        all_countries = [f["properties"]["name"] for f in geojson_data["features"]]

        # ---------------- GLOBAL ----------------
        if level == "Global":
            df_country = (
                df_static.groupby("Country")[selected_feature]
                .mean()
                .reset_index()
            )

            missing = list(set(all_countries) - set(df_country["Country"]))
            df_missing = pd.DataFrame(
                {"Country": missing, selected_feature: [0] * len(missing)}
            )

            df_country = pd.concat([df_country, df_missing], ignore_index=True)

            folium.Choropleth(
                geo_data=geojson_data,
                data=df_country,
                columns=["Country", selected_feature],
                key_on="feature.properties.name",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=selected_feature,
            ).add_to(map_dashboard)

        # ---------------- COUNTRY / STATE ----------------
        else:
            selected_country = st.selectbox(
                "Select Country",
                sorted(df_static["Country"].unique())
            )

            df_country = df_static[df_static["Country"] == selected_country]

            # Country-level choropleth
            country_avg = (
                df_country.groupby("Country")[selected_feature]
                .mean()
                .reset_index()
            )

            folium.Choropleth(
                geo_data=geojson_data,
                data=country_avg,
                columns=["Country", selected_feature],
                key_on="feature.properties.name",
                fill_color="YlOrRd",
                fill_opacity=0.6,
                line_opacity=0.2,
                legend_name=selected_feature,
            ).add_to(map_dashboard)

            # ---------------- STATE MARKERS ----------------
            df_states = (
                df_country.groupby("State", dropna=True)
                .mean(numeric_only=True)
                .reset_index()
                .dropna(subset=["Latitude", "Longitude"])
            )

            max_val = df_states[selected_feature].max() or 1

            for _, r in df_states.iterrows():
                folium.CircleMarker(
                    location=[r["Latitude"], r["Longitude"]],
                    radius=max(4, min(14, r[selected_feature] / max_val * 14)),
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=(
                        f"<b>{r['State']}</b><br>"
                        f"{selected_country}<br>"
                        f"{selected_feature}: {r[selected_feature]:.2f}"
                    ),
                ).add_to(map_dashboard)

            map_dashboard.location = [
                df_country["Latitude"].mean(),
                df_country["Longitude"].mean(),
            ]
            map_dashboard.zoom_start = 5

    map_dashboard.add_child(folium.LayerControl())
    map_dashboard.add_child(folium.LatLngPopup())
    st_folium(map_dashboard, height=600, use_container_width=True)


# ----------------- PAGE 2: POLLUTION ATTRIBUTION -----------------
elif page == "Pollution Attribution":

    st.title("🌍 Pollution Source Attribution")
    st.caption(
        "Click on the map to analyze pollution sources using AQI, weather, wind dynamics "
        "and ML-based severity prediction."
    )

    # ---------- BASE MAP ----------
    base_map = folium.Map(location=[0, 0], zoom_start=2)
    base_map.add_child(folium.LatLngPopup())
    click = st_folium(base_map, height=550, use_container_width=True)

    if click and click.get("last_clicked"):
        lat = click["last_clicked"]["lat"]
        lon = click["last_clicked"]["lng"]

        # ---------- FETCH DATA ----------
        weather = fetch_weather(lat, lon)
        aqi = get_aqi_data(lat, lon)

        # ---------- MODEL PREDICTION ----------
        model_input = prepare_model_input(weather, aqi)

        severity_idx = int(xgb_model.predict(model_input)[0])
        severity_label = SEVERITY_MAP[severity_idx]

        severity_conf = float(
            np.max(xgb_model.predict_proba(model_input))
        )

        # ---------- DISPLAY LOCAL CONDITIONS ----------
        st.subheader("📊 Local Conditions (Weather + AQI + Severity)")

        combined = {
            **{f"Weather | {k}": v for k, v in weather.items()},
            **{f"AQI | {k}": v for k, v in aqi.items()},
            "Predicted Severity": severity_label,
            "Severity Confidence": f"{severity_conf:.2f}"
        }

        df_combined = pd.DataFrame([combined])
        st.dataframe(
            safe_style(df_combined),
            use_container_width=True
        )

        # ---------- FETCH OSM SOURCES ----------
        sources_by_category = {}
        for cat in SOURCE_INDEX:
            src = fetch_osm_sources(lat, lon, cat)
            src = sorted(
                src,
                key=lambda s: geodesic((lat, lon), (s["lat"], s["lon"])).meters
            )
            sources_by_category[cat] = src[:MAX_SOURCES_PER_TYPE]

        # ---------- CONTRIBUTION MODEL ----------
        major, category_pct, contributions = calculate_contributions(
            lat, lon, sources_by_category, weather
        )

        # ---------- NATURAL SOURCE SUPPRESSION ----------
        DOM_THRESHOLD = 15  # %
        sorted_cats = sorted(category_pct.items(), key=lambda x: x[1], reverse=True)

        if sorted_cats and sorted_cats[0][0] == "Natural":
            top = sorted_cats[0][1]
            second = sorted_cats[1][1] if len(sorted_cats) > 1 else 0

            if top - second < DOM_THRESHOLD:
                category_pct.pop("Natural", None)
                contributions = [
                    c for c in contributions if c["source"] != "Natural"
                ]

                total = sum(category_pct.values()) or 1
                category_pct = {k: v / total * 100 for k, v in category_pct.items()}

        major = max(category_pct, key=category_pct.get) if category_pct else "None"

        # ---------- RESULT MAP ----------
        result_map = folium.Map(location=[lat, lon], zoom_start=13)

        folium.Marker(
            [lat, lon],
            popup=f"Selected Location<br>Severity: {severity_label}",
            icon=folium.Icon(color="blue")
        ).add_to(result_map)

        for c in contributions:
            popup = (
                f"{c['source']}<br>"
                f"Overall Contribution: {c['overall_pct']:.1f}%<br>"
                f"Category Contribution: {c['category_pct']:.1f}%<br>"
                f"Distance: {int(c['distance'])} m"
            )
            folium.Marker(
                [c["lat"], c["lon"]],
                popup=popup,
                icon=folium.Icon(color=SOURCE_COLORS[c["source"]])
            ).add_to(result_map)

        st.subheader(
            f"🗺️ Pollution Sources Map "
            f"(Major: {major} | Severity: {severity_label})"
        )
        st_folium(result_map, height=550, use_container_width=True)

        # ---------- CATEGORY TABLE ----------
        st.subheader("📊 Category-wise Contribution (%)")
        df_cat = pd.DataFrame([category_pct])
        st.dataframe(safe_style(df_cat), use_container_width=True)


        # ---------- SUMMARY ----------
        st.success(
            f"Dominant Source: **{major}**  \n"
            f"Predicted Severity: **{severity_label}** "
            f"(Confidence: {severity_conf:.2f})"
        )

        # ---------- SESSION STORAGE ----------
        st.session_state["last_analysis"] = {
            "location": (lat, lon),
            "weather": weather,
            "aqi": aqi,
            "severity": severity_label,
            "severity_confidence": severity_conf,
            "category_pct": category_pct,
            "contributions": contributions
        }

        
#-------------------PAGE 3: HISTORICAL DATA -----------------

elif page == "Historical Data":
    
    st.title("📥 Historical Data Preview & Download")
    st.caption("Pseudo-historical data generated from last Pollution Attribution run")

    # ----------------- VALIDATION -----------------
    if "last_analysis" not in st.session_state:
        st.warning("Please run Pollution Attribution first.")
        st.stop()

    analysis = st.session_state["last_analysis"]
    lat, lon = analysis["location"]
    contributions = analysis["contributions"]

    st.subheader(f"Selected Location: ({lat:.3f}, {lon:.3f})")

    # ----------------- PARAMETERS -----------------
    REPEATS = 3  # Safe upper limit (can increase to 5)

    # ----------------- LOCATION TEMPORAL DATA -----------------
    loc_records = []

    weather = fetch_weather_cached(lat, lon)
    aqi = get_aqi_cached(lat, lon)

    for i in range(REPEATS):
        loc_records.append({
            "timestamp": pd.Timestamp.utcnow() + pd.Timedelta(seconds=i * 15),
            "latitude": lat,
            "longitude": lon,
            **weather,
            **aqi
        })

    df_location_hist = pd.DataFrame(loc_records)
    df_location_hist["timestamp"] = pd.to_datetime(df_location_hist["timestamp"])

    st.subheader("Selected Location - Temporal Data")
    st.dataframe(safe_style(df_location_hist), use_container_width=True)

    # ----------------- SOURCE TEMPORAL DATA -----------------
    all_sources = []

    for cat in SOURCE_INDEX:

        cat_sources = [c for c in contributions if c["source"] == cat][:5]
        rows = []

        for c in cat_sources:
            weather = fetch_weather_cached(c["lat"], c["lon"])
            aqi = get_aqi_cached(c["lat"], c["lon"])

            for i in range(REPEATS):
                rows.append({
                    "timestamp": pd.Timestamp.utcnow() + pd.Timedelta(seconds=i * 15),
                    "source": cat,
                    "latitude": c["lat"],
                    "longitude": c["lon"],
                    "distance_m": float(c["distance"]),
                    "overall_pct": float(c["overall_pct"]),
                    "category_pct": float(c["category_pct"]),
                    **weather,
                    **aqi
                })

        if rows:
            df_cat = pd.DataFrame(rows)
            df_cat["timestamp"] = pd.to_datetime(df_cat["timestamp"])
            all_sources.append(df_cat)

            st.subheader(f"{cat} Sources - Temporal Data")
            st.dataframe(safe_style(df_cat), use_container_width=True)

            st.download_button(
                f"Download {cat} CSV",
                df_cat.to_csv(index=False),
                file_name=f"{cat.lower()}_sources_history.csv",
                mime="text/csv"
            )

    # ----------------- MERGED SOURCES -----------------
    df_all_sources = None
    if all_sources:
        df_all_sources = pd.concat(all_sources, ignore_index=True)

        st.subheader("All Predicted Sources (Combined)")
        st.dataframe(safe_style(df_all_sources.head(25)), use_container_width=True)

        st.download_button(
            "Download All Sources CSV",
            df_all_sources.to_csv(index=False),
            file_name="all_sources_history.csv",
            mime="text/csv"
        )

    # ----------------- LOCATION CSV -----------------
    st.download_button(
        "Download Selected Location CSV",
        df_location_hist.to_csv(index=False),
        file_name="selected_location_history.csv",
        mime="text/csv"
    )

    # ----------------- PDF EXPORT -----------------
    pdf_bytes = create_pdf(
        df_location=df_location_hist,
        df_sources=df_all_sources,
        location=(lat, lon)
    )

    st.download_button(
        "Download Full PDF Report",
        pdf_bytes,
        file_name="pollution_history_report.pdf",
        mime="application/pdf"
    )
