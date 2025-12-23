# import streamlit as st
# import pandas as pd
# import folium
# from folium.plugins import HeatMap

# st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")

# st.title("🌍 EnviroScan – AI Pollution Monitoring Dashboard")

# # =======================
# # LOAD DATA
# # =======================
# @st.cache_data
# def load_data():
#     return pd.read_csv(
#         r"C:\Users\mothe\OneDrive\Documents\Batch-6\enviroscan_week3_labeled_dataset.csv"
#     )

# df = load_data()

# # =======================
# # SIDEBAR FILTERS
# # =======================
# st.sidebar.header("🔍 Filters")

# selected_country = st.sidebar.selectbox(
#     "Select Country",
#     options=["All"] + sorted(df["Country"].unique().tolist())
# )

# selected_source = st.sidebar.selectbox(
#     "Select Pollution Source",
#     options=["All"] + sorted(df["pollution_source"].unique().tolist())
# )

# df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# min_date = df["Timestamp"].min().date()
# max_date = df["Timestamp"].max().date()

# selected_date = st.sidebar.date_input(
#     "Select Date",
#     value=min_date,
#     min_value=min_date,
#     max_value=max_date
# )

# selected_city = st.sidebar.selectbox(
#     "Select City",
#     options=["All"] + sorted(df["City"].unique().tolist())
# )

# st.sidebar.subheader("📍 Coordinate Filter (Optional)")

# lat_input = st.sidebar.number_input("Latitude", value=0.0, format="%.4f")
# lon_input = st.sidebar.number_input("Longitude", value=0.0, format="%.4f")
# radius_km = st.sidebar.slider("Radius (km)", 1, 50, 10)

# df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# start_date, end_date = st.sidebar.date_input(
#     "Select Date Range",
#     value=[df["Timestamp"].min().date(), df["Timestamp"].max().date()]
# )





# # =======================
# # APPLY FILTERS
# # =======================
# filtered_df = df.copy()

# if selected_country != "All":
#     filtered_df = filtered_df[filtered_df["Country"] == selected_country]

# if selected_source != "All":
#     filtered_df = filtered_df[
#         filtered_df["pollution_source"] == selected_source
#     ]
# filtered_df = filtered_df[
#     filtered_df["Timestamp"].dt.date == selected_date
# ]

# if selected_city != "All":
#     filtered_df = filtered_df[filtered_df["City"] == selected_city]

#     import numpy as np

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1, phi2 = np.radians(lat1), np.radians(lat2)
#     dphi = np.radians(lat2 - lat1)
#     dlambda = np.radians(lon2 - lon1)
#     a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
#     return 2 * R * np.arcsin(np.sqrt(a))

# if lat_input != 0.0 and lon_input != 0.0:
#     filtered_df = filtered_df[
#         haversine(
#             lat_input, lon_input,
#             filtered_df["Latitude"], filtered_df["Longitude"]
#         ) <= radius_km
#     ]
# filtered_df = filtered_df[
#     (filtered_df["Timestamp"].dt.date >= start_date) &
#     (filtered_df["Timestamp"].dt.date <= end_date)
# ]





# # =======================
# # REAL-TIME ALERT LOGIC
# # =======================
# def get_alert_level(aqi):
#     if aqi > 200:
#         return "CRITICAL"
#     elif aqi > 100:
#         return "WARNING"
#     else:
#         return "SAFE"

# filtered_df["Alert_Level"] = filtered_df["AQI Value"].apply(get_alert_level)

# # =======================
# # ALERT BANNER
# # =======================
# st.subheader("🚨 Real-Time Pollution Alerts")

# critical_count = (filtered_df["Alert_Level"] == "CRITICAL").sum()
# warning_count = (filtered_df["Alert_Level"] == "WARNING").sum()

# if critical_count > 0:
#     st.error(f"🚨 CRITICAL ALERT: {critical_count} locations exceed safe AQI levels")
# elif warning_count > 0:
#     st.warning(f"⚠️ WARNING: {warning_count} locations show elevated pollution")
# else:
#     st.success("✅ All monitored locations are within safe pollution limits")

# # =======================
# # METRICS
# # =======================
# st.subheader("📊 Dataset Overview")

# col1, col2, col3 = st.columns(3)

# col1.metric("Total Records", filtered_df.shape[0])
# col2.metric("Total Countries", filtered_df["Country"].nunique())
# col3.metric("Total Cities", filtered_df["City"].nunique())

# # =======================
# # BAR CHART
# # =======================
# st.subheader("🧪 Pollution Source Distribution")

# st.bar_chart(
#     filtered_df["pollution_source"].value_counts()
# )

# # =======================
# # ALERT SUMMARY TABLE + DOWNLOAD
# # =======================
# st.subheader("📋 Alert Summary")

# alert_df = filtered_df[
#     filtered_df["Alert_Level"] != "SAFE"
# ][["Country", "City", "AQI Value", "pollution_source", "Alert_Level"]]

# st.dataframe(alert_df)

# st.download_button(
#     label="⬇️ Download Alert Report (CSV)",
#     data=alert_df.to_csv(index=False),
#     file_name="pollution_alert_report.csv",
#     mime="text/csv"
# )

# # =======================
# # MAP SECTION
# # =======================
# st.subheader("🗺️ Pollution Source Map")

# if len(filtered_df) > 0:
#     map_df = filtered_df.sample(
#         min(1000, len(filtered_df)),
#         random_state=42
#     )

#     map_center = [
#         map_df["Latitude"].mean(),
#         map_df["Longitude"].mean()
#     ]

#     m = folium.Map(
#         location=map_center,
#         zoom_start=2,
#         tiles="cartodbpositron"
#     )

#     source_colors = {
#         "Industrial": "red",
#         "Vehicular": "blue",
#         "Agricultural": "green",
#         "Natural": "gray"
#     }

#     for _, row in map_df.iterrows():
#         folium.CircleMarker(
#             location=[row["Latitude"], row["Longitude"]],
#             radius=4,
#             color=source_colors.get(row["pollution_source"], "black"),
#             fill=True,
#             fill_opacity=0.6,
#             popup=(
#                 f"City: {row['City']}<br>"
#                 f"Country: {row['Country']}<br>"
#                 f"AQI: {row['AQI Value']}<br>"
#                 f"Source: {row['pollution_source']}<br>"
#                 f"Alert: {row['Alert_Level']}"
#             )
#         ).add_to(m)

#     HeatMap(
#         map_df[["Latitude", "Longitude", "AQI Value"]].values.tolist(),
#         radius=15,
#         blur=20,
#     ).add_to(m)

#     st.components.v1.html(m._repr_html_(), height=600)

# else:
#     st.warning("No data available for the selected filters.")


import streamlit as st
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap

st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")
st.title("🌍 EnviroScan – AI Pollution Monitoring Dashboard")

# =======================
# LOAD DATA
# =======================
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\mothe\OneDrive\Documents\Batch-6\enviroscan_week3_labeled_dataset.csv"
    )

df = load_data()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# =======================
# SIDEBAR FILTERS
# =======================
st.sidebar.header("🔍 Filters")

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["Country"].unique())
)

selected_city = st.sidebar.selectbox(
    "Select City",
    ["All"] + sorted(df["City"].unique())
)

selected_source = st.sidebar.selectbox(
    "Select Pollution Source",
    ["All"] + sorted(df["pollution_source"].unique())
)

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [df["Timestamp"].min().date(), df["Timestamp"].max().date()]
)

st.sidebar.subheader("📍 Coordinate Filter (Optional)")
lat_input = st.sidebar.number_input("Latitude", value=0.0, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=0.0, format="%.4f")
radius_km = st.sidebar.slider("Radius (km)", 1, 50, 10)

# =======================
# APPLY FILTERS
# =======================
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

if selected_city != "All":
    filtered_df = filtered_df[filtered_df["City"] == selected_city]

if selected_source != "All":
    filtered_df = filtered_df[filtered_df["pollution_source"] == selected_source]

filtered_df = filtered_df[
    (filtered_df["Timestamp"].dt.date >= start_date) &
    (filtered_df["Timestamp"].dt.date <= end_date)
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

if lat_input != 0.0 and lon_input != 0.0:
    filtered_df = filtered_df[
        haversine(
            lat_input, lon_input,
            filtered_df["Latitude"], filtered_df["Longitude"]
        ) <= radius_km
    ]

# =======================
# REAL-TIME ALERTS
# =======================
def get_alert_level(aqi):
    if aqi > 200:
        return "CRITICAL"
    elif aqi > 100:
        return "WARNING"
    else:
        return "SAFE"

filtered_df["Alert_Level"] = filtered_df["AQI Value"].apply(get_alert_level)

st.subheader("🚨 Real-Time Pollution Alerts")

critical_count = (filtered_df["Alert_Level"] == "CRITICAL").sum()
warning_count = (filtered_df["Alert_Level"] == "WARNING").sum()

if critical_count > 0:
    st.error(f"🚨 CRITICAL ALERT: {critical_count} locations exceed safe AQI levels")
    st.info("📧 Email/SMS alert triggered (demo mode)")
elif warning_count > 0:
    st.warning(f"⚠️ WARNING: {warning_count} locations show elevated pollution")
else:
    st.success("✅ All monitored locations are within safe pollution limits")

# =======================
# METRICS
# =======================
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", filtered_df.shape[0])
c2.metric("Countries", filtered_df["Country"].nunique())
c3.metric("Cities", filtered_df["City"].nunique())

# =======================
# 5️⃣ POLLUTANT TREND CHART
# =======================
st.subheader("📈 Pollution Trend Over Time")

trend_df = filtered_df.groupby(
    filtered_df["Timestamp"].dt.hour
)["AQI Value"].mean()

st.line_chart(trend_df)

# =======================
# SOURCE DISTRIBUTION (BAR + PIE)
# =======================
st.subheader("🧪 Pollution Source Distribution")
st.bar_chart(filtered_df["pollution_source"].value_counts())

fig, ax = plt.subplots()
ax.pie(
    filtered_df["pollution_source"].value_counts(),
    labels=filtered_df["pollution_source"].value_counts().index,
    autopct="%1.1f%%"
)
st.pyplot(fig)

# =======================
# 7️⃣ DAILY / WEEKLY REPORT EXPORT
# =======================
st.subheader("⬇️ Export Pollution Report")

st.download_button(
    "Download Full Pollution Report (CSV)",
    filtered_df.to_csv(index=False),
    file_name="daily_weekly_pollution_report.csv",
    mime="text/csv"
)

# =======================
# MAP + HEATMAP
# =======================
st.subheader("🗺️ Pollution Source Map")

if len(filtered_df) > 0:
    map_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)

    m = folium.Map(
        location=[map_df["Latitude"].mean(), map_df["Longitude"].mean()],
        zoom_start=2,
        tiles="cartodbpositron"
    )

    colors = {
        "Industrial": "red",
        "Vehicular": "blue",
        "Agricultural": "green",
        "Natural": "gray"
    }

    for _, row in map_df.iterrows():
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=4,
            color=colors.get(row["pollution_source"], "black"),
            fill=True,
            popup=f"{row['City']} | AQI {row['AQI Value']} | {row['Alert_Level']}"
        ).add_to(m)

    HeatMap(
        map_df[["Latitude", "Longitude", "AQI Value"]].values.tolist(),
        radius=15, blur=20
    ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=600)

else:
    st.warning("No data available for the selected filters.")
