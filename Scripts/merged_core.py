import pandas as pd

# Load CSVs
loc = pd.read_csv("india_locations.csv")
air = pd.read_csv("india_air_quality.csv")
weather = pd.read_csv("india_weather.csv")
features = pd.read_csv("india_features.csv")

# -------------------------------
# 1️⃣ Merge AIR using city + lat + lon
# -------------------------------
merged = loc.merge(
    air,
    on=["city", "latitude", "longitude"],
    how="outer"
)

# -------------------------------
# 2️⃣ Merge WEATHER using location_id
# -------------------------------
merged = merged.merge(
    weather,
    on=["location_id", "city", "latitude", "longitude"],
    how="outer"
)

# -------------------------------
# 3️⃣ Merge FEATURES using location_id
# -------------------------------
merged = merged.merge(
    features,
    on=["location_id", "city"],
    how="outer"
)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
merged.to_csv("india_merged_all_rows_columns.csv", index=False)

print("✅ FULL MERGE SUCCESSFUL")
print("Rows:", merged.shape[0])
print("Columns:", merged.shape[1])
