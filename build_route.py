import pandas as pd
import numpy as np

print("🚄 Generating realistic Indian train route...")

# Chennai Central (start)
start_lat, start_lon = 13.0827, 80.2707

# Bengaluru (end)
end_lat, end_lon = 12.9716, 77.5946

# number of route points
N = 400

# base interpolation
lats = np.linspace(start_lat, end_lat, N)
lons = np.linspace(start_lon, end_lon, N)

# add slight curvature (railways are not straight)
curve_strength = 0.08
curve = curve_strength * np.sin(np.linspace(0, 3*np.pi, N))

lons = lons + curve

route = pd.DataFrame({
    'lat': lats,
    'lon': lons
})

route.to_csv("data/train_route.csv", index=False)

print("✅ Full train_route.csv generated with", N, "points")