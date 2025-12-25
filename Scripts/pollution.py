import asyncio
import aiohttp
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

INPUT_FILE = "india_locations.csv"
OUTPUT_FILE = "india_air_quality.csv"

CONCURRENT_REQUESTS = 10   # safe for free tier

def aqi_category(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Satisfactory"
    elif value <= 200:
        return "Moderate"
    elif value <= 300:
        return "Poor"
    elif value <= 400:
        return "Very Poor"
    else:
        return "Severe"

async def fetch_aq(session, city, lat, lon, sem):
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )

    async with sem:
        async with session.get(url) as r:
            if r.status != 200:
                print(f"❌ Failed: {city}")
                return None

            data = (await r.json())["list"][0]
            comp = data["components"]
            aqi = data["main"]["aqi"]

            print(f"✅ {city}")

            return {
                "city": city,
                "latitude": lat,
                "longitude": lon,

                "PM25 AQ": comp["pm2_5"],
                "PM25 AQI Category": aqi_category(comp["pm2_5"]),

                "PM10 AQ": comp["pm10"],
                "PM10 AQI Category": aqi_category(comp["pm10"]),

                "CO AQ": comp["co"],
                "CO AQI Category": aqi_category(comp["co"]),

                "NO2 AQ": comp["no2"],
                "NO2 AQI Category": aqi_category(comp["no2"]),

                "SO2 AQ": comp["so2"],
                "SO2 AQI Category": aqi_category(comp["so2"]),

                "O3 AQ": comp["o3"],
                "O3 AQI Category": aqi_category(comp["o3"]),

                "AQI": aqi
            }

async def main():
    df = pd.read_csv(INPUT_FILE)
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_aq(session, row["city"], row["latitude"], row["longitude"], sem)
            for _, row in df.iterrows()
        ]

        results = await asyncio.gather(*tasks)

    results = [r for r in results if r is not None]
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("🎉 Saved:", OUTPUT_FILE)
    print(final_df.head())

if __name__ == "__main__":
    asyncio.run(main())
