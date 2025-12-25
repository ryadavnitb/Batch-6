import pandas as pd
import osmnx as ox
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "india_locations.csv"     # CSV: location_id, city, latitude, longitude
OUTPUT_FILE = "india_features.csv"
NUM_THREADS = 20
SEARCH_RADIUS = 2000  # meters

TAGS = {
    'highway': ['primary', 'secondary', 'tertiary', 'residential'],
    'landuse': ['industrial', 'farmland', 'landfill'],
    'amenity': ['waste_disposal']
}

# ============================================================
# FUNCTIONS
# ============================================================

def get_utm_epsg(lon, lat):
    zone_number = int((lon + 180) / 6) + 1
    return 32600 + zone_number if lat >= 0 else 32700 + zone_number


def process_city(row):
    location_id = row['location_id']
    city = row['city']
    lat = row['latitude']
    lon = row['longitude']

    print(f"\n📍 Processing: {city}")

    city_geom = Point(lon, lat)

    try:
        gdf = ox.features_from_point(
            (lat, lon),
            tags=TAGS,
            dist=SEARCH_RADIUS
        )

        def count_feature(col, value=None):
            if gdf.empty or col not in gdf.columns:
                return 0
            return (gdf[col] == value).sum() if value else gdf[col].notna().sum()

        def nearest_distance(col, value=None):
            if gdf.empty or col not in gdf.columns:
                return None
            gdf_f = gdf[gdf[col] == value] if value else gdf
            if gdf_f.empty:
                return None

            epsg = get_utm_epsg(lon, lat)
            gdf_proj = gdf_f.to_crs(epsg=epsg)
            city_proj = ox.projection.project_geometry(
                city_geom, to_crs=gdf_proj.crs
            )[0]
            return round(gdf_proj.geometry.distance(city_proj).min(), 2)

        # ================= COUNTS =================
        road_cnt = count_feature("highway")
        ind_cnt = count_feature("landuse", "industrial")
        farm_cnt = count_feature("landuse", "farmland")
        land_cnt = count_feature("landuse", "landfill")
        dump_cnt = count_feature("amenity", "waste_disposal")

        # ================= DISTANCES =================
        road_dist = nearest_distance("highway")
        ind_dist = nearest_distance("landuse", "industrial")
        farm_dist = nearest_distance("landuse", "farmland")
        land_dist = nearest_distance("landuse", "landfill")
        dump_dist = nearest_distance("amenity", "waste_disposal")

        # ================= PRINT =================
        print(
            f"   🛣 Roads: {road_cnt} | "
            f"🏭 Industrial: {ind_cnt} | "
            f"🌾 Farmland: {farm_cnt} | "
            f"🗑 Landfill: {land_cnt} | "
            f"🚮 Dump: {dump_cnt}"
        )

        print(
            f"   📏 Distances (m) → "
            f"Road: {road_dist}, "
            f"Industrial: {ind_dist}, "
            f"Farmland: {farm_dist}, "
            f"Landfill: {land_dist}, "
            f"Dump: {dump_dist}"
        )

        return {
            "location_id": location_id,
            "city": city,
            "Road_Count": road_cnt,
            "Industrial_Count": ind_cnt,
            "Farmland_Count": farm_cnt,
            "Landfill_Count": land_cnt,
            "Dump_Site_Count": dump_cnt,
            "Distance_to_Nearest_Road_m": road_dist,
            "Distance_to_Nearest_Industrial_m": ind_dist,
            "Distance_to_Nearest_Farmland_m": farm_dist,
            "Distance_to_Nearest_Landfill_m": land_dist,
            "Distance_to_Nearest_Dump_m": dump_dist
        }

    except Exception as e:
        print(f"❌ Failed {city}: {e}")
        return {
            "location_id": location_id,
            "city": city,
            "Road_Count": 0,
            "Industrial_Count": 0,
            "Farmland_Count": 0,
            "Landfill_Count": 0,
            "Dump_Site_Count": 0,
            "Distance_to_Nearest_Road_m": 0,
            "Distance_to_Nearest_Industrial_m": 0,
            "Distance_to_Nearest_Farmland_m": 0,
            "Distance_to_Nearest_Landfill_m": 0,
            "Distance_to_Nearest_Dump_m": 0
        }


# ============================================================
# LOAD INPUT CSV
# ============================================================

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()

cities = df[["location_id", "city", "latitude", "longitude"]].dropna()

print(f"\n🚀 Total cities to process: {len(cities)}")

# ============================================================
# MULTI-THREAD EXECUTION
# ============================================================

results = []

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(process_city, row) for _, row in cities.iterrows()]
    for future in as_completed(futures):
        results.append(future.result())

# ============================================================
# SAVE OUTPUT
# ============================================================

df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n✅ COMPLETED SUCCESSFULLY")
print(f"📁 Output saved as: {OUTPUT_FILE}")
print(df_final.head())
