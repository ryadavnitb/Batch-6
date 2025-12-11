import aiohttp
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import ast

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")  # OpenWeatherMap API key
LOCATIONS_FILE = "openaq_locations.csv"
OUTPUT_FILE = "pollution_weather_dataset.csv"
POLLUTANTS = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
BATCH_SIZE = 40  # concurrent requests

# AQI breakpoints
AQI_BREAKPOINTS = {
    "pm2_5": [(0, 12, "Good"), (12.1, 35.4, "Moderate"), (35.5, 55.4, "Unhealthy")],
    "pm10": [(0, 54, "Good"), (55, 154, "Moderate"), (155, 254, "Unhealthy")],
    "co": [(0, 4.4, "Good"), (4.5, 9.4, "Moderate"), (9.5, 12.4, "Unhealthy")],
    "no2": [(0, 53, "Good"), (54, 100, "Moderate"), (101, 360, "Unhealthy")],
    "so2": [(0, 35, "Good"), (36, 75, "Moderate"), (76, 185, "Unhealthy")],
    "o3": [(0, 54, "Good"), (55, 70, "Moderate"), (71, 85, "Unhealthy")]
}

def get_aqi_category(value, pollutant):
    if value is None:
        return None
    for low, high, cat in AQI_BREAKPOINTS[pollutant]:
        if low <= value <= high:
            return cat
    return "Unhealthy"

def extract_lat_lon(coord_str):
    try:
        coord = ast.literal_eval(coord_str)
        return float(coord['latitude']), float(coord['longitude'])
    except:
        return None, None

# Load locations
df_locations = pd.read_csv(LOCATIONS_FILE)
df_locations['latitude'], df_locations['longitude'] = zip(*df_locations['coordinates'].map(extract_lat_lon))
locations = df_locations.to_dict(orient='records')

# -------------------- Async fetch --------------------
async def fetch_pollution(session, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        async with session.get(url, timeout=10) as r:
            if r.status != 200:
                return {p: None for p in POLLUTANTS}, None
            data = await r.json()
            comp = data['list'][0]['components'] if data.get('list') else {p: None for p in POLLUTANTS}
            return comp, data['list'][0]['main']['aqi'] if data.get('list') else None
    except:
        return {p: None for p in POLLUTANTS}, None

async def fetch_weather(session, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        async with session.get(url, timeout=10) as r:
            if r.status != 200:
                return {
                    "temperature": None, "humidity": None, "wind_speed": None, "wind_deg": None, 
                    "timestamp": None, "city": None, "country": None
                }
            data = await r.json()
            return {
                "temperature": data.get("main", {}).get("temp"),
                "humidity": data.get("main", {}).get("humidity"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_deg": data.get("wind", {}).get("deg"),
                "timestamp": datetime.utcfromtimestamp(data.get("dt")).isoformat() if data.get("dt") else None,
                "city": data.get("name"),
                "country": data.get("sys", {}).get("country")
            }
    except:
        return {
            "temperature": None, "humidity": None, "wind_speed": None, "wind_deg": None, 
            "timestamp": None, "city": None, "country": None
        }

async def process_location(session, loc):
    lat, lon = loc.get('latitude'), loc.get('longitude')
    if lat is None or lon is None:
        return None

    # Fetch in parallel
    pollution_task = fetch_pollution(session, lat, lon)
    weather_task = fetch_weather(session, lat, lon)
    comp, aqi_overall = await pollution_task
    weather = await weather_task

    city = weather.get("city") or loc.get("city")
    country = weather.get("country") or loc.get("country")

    row = {
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "country": country
    }

    # Map pollutants + AQI
    for p in POLLUTANTS:
        val = comp.get(p)
        col = p.upper() if p != 'pm2_5' else 'PM25'
        row[f"{col} AQ"] = val
        row[f"{col} AQI Category"] = get_aqi_category(val, p)
    row["AQI"] = aqi_overall

    # Weather
    row.update({
        "temperature": weather.get("temperature"),
        "humidity": weather.get("humidity"),
        "wind_speed": weather.get("wind_speed"),
        "wind_deg": weather.get("wind_deg"),
        "timestamp": weather.get("timestamp")
    })

    return row

# -------------------- Main --------------------
async def main():
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(locations), BATCH_SIZE):
            batch = locations[i:i+BATCH_SIZE]
            tasks = [process_location(session, loc) for loc in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r])
            print(f"Processed {i+len(batch)} / {len(locations)} locations...")

    df = pd.DataFrame(results)

    # Column order
    final_columns = [
        "city","country","latitude","longitude",
        "PM25 AQ","PM25 AQI Category",
        "PM10 AQ","PM10 AQI Category",
        "CO AQ","CO AQI Category",
        "NO2 AQ","NO2 AQI Category",
        "SO2 AQ","SO2 AQI Category",
        "O3 AQ","O3 AQI Category",
        "AQI",
        "temperature","humidity","wind_speed","wind_deg","timestamp"
    ]
    for col in final_columns:
        if col not in df.columns:
            df[col] = None
    df = df[final_columns]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🔥 Saved final CSV: {OUTPUT_FILE}")
    print(f"Total locations processed: {len(df)}")

# -------------------- Run --------------------
asyncio.run(main())
