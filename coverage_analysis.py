import pandas as pd
import numpy as np

print("📡 Starting coverage analysis...")


# Vectorized Haversine Distance Function

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



# Load datasets

print("📂 Loading tower dataset...")
towers = pd.read_csv("data/towers_india.csv")

# 🔥 SPEED BOOST (VERY IMPORTANT)
# Reduce from ~10 lakh towers to manageable size
towers = towers.sample(50000, random_state=42)

print("📂 Loading train route...")
route = pd.read_csv("data/train_route.csv")

print(f"✅ Towers loaded (sampled): {len(towers)}")
print(f"✅ Route points loaded: {len(route)}")


# Prepare tower arrays (faster)

tower_lat = towers["lat"].values
tower_lon = towers["lon"].values


# Find nearest tower for each route point

print("📡 Computing nearest tower distances...")

nearest_distances = []

for idx, r in route.iterrows():
    distances = haversine_vectorized(
        r["lat"],
        r["lon"],
        tower_lat,
        tower_lon,
    )
    nearest_distances.append(np.min(distances))

    # progress indicator every 50 points
    if idx % 50 == 0:
        print(f"   Processed {idx}/{len(route)} route points")

route["nearest_tower_km"] = nearest_distances


# Simulate signal strength (path loss model)

print("📶 Estimating signal strength...")

route["signal_strength"] = -30 - (route["nearest_tower_km"] * 6)


# Signal quality classification

def signal_quality(dbm):
    if dbm > -70:
        return "Excellent"
    elif dbm > -90:
        return "Good"
    elif dbm > -110:
        return "Poor"
    else:
        return "No Signal"

route["quality"] = route["signal_strength"].apply(signal_quality)


# Save output

output_path = "data/coverage_analysis.csv"
route.to_csv(output_path, index=False)

print(f"✅ Coverage analysis saved to {output_path}")
print("🎉 Step 3 complete!")
