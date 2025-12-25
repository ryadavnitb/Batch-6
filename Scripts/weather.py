import pandas as pd
import aiohttp
import asyncio
import os
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
INPUT_FILE = "india_locations.csv"
OUTPUT_FILE = "india_weather.csv"
API_URL = "https://api.openweathermap.org/data/2.5/weather"
CONCURRENT_REQUESTS = 10   # safe limit

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------- LOAD INPUT ----------------
df = pd.read_csv(INPUT_FILE)

if df.empty:
    print("❌ Input file empty")
    exit()

results = []
sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

# ---------------- FETCH FUNCTION ----------------
async def fetch_weather(session, row):
    async with sem:
        params = {
            "lat": row["latitude"],
            "lon": row["longitude"],
            "appid": API_KEY,
            "units": "metric"
        }

        try:
            async with session.get(API_URL, params=params, timeout=15) as resp:
                data = await resp.json()

                return {
                    "location_id": row["location_id"],
                    "city": row["city"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "wind_deg": data["wind"].get("deg"),
                    "timestamp": data["dt"]
                }

        except Exception as e:
            print(f"❌ Failed {row['city']}: {e}")
            return None

# ---------------- MAIN ----------------
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_weather(session, row) for _, row in df.iterrows()]
        responses = await asyncio.gather(*tasks)

        for r in responses:
            if r:
                results.append(r)

# ---------------- RUN ----------------
asyncio.run(main())

# ---------------- SAVE ----------------
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ FAST fetch completed → {OUTPUT_FILE}")
print(final_df.head())
