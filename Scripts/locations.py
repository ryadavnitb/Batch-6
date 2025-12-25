import requests
import pandas as pd

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

query = """
[out:json][timeout:180];
area["ISO3166-1"="IN"][admin_level=2]->.india;
node(area.india)["place"~"city|town"];
out body;
"""

print("🔄 Sending query to Overpass...")
r = requests.post(OVERPASS_URL, data=query)

print("HTTP Status:", r.status_code)

data = r.json()
elements = data.get("elements", [])

print("Elements fetched:", len(elements))

rows = []
for el in elements:
    if "lat" not in el or "lon" not in el:
        continue

    rows.append({
        "location_id": el["id"],
        "city": el.get("tags", {}).get("name"),
        "latitude": el["lat"],
        "longitude": el["lon"]
    })

df = pd.DataFrame(rows)
df.to_csv("india_locations.csv", index=False)

print("✅ Saved india_locations.csv")
print(df.head())
