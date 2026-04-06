import pandas as pd
import folium

print("🗺️ Creating coverage map...")

# -------------------------------------------------
# Load coverage data
# -------------------------------------------------
data = pd.read_csv("data/coverage_analysis.csv")

print(f"✅ Loaded {len(data)} coverage points")

# -------------------------------------------------
# Create base map (center of India)
# -------------------------------------------------
center_lat = data["lat"].mean()
center_lon = data["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="OpenStreetMap"
)

# -------------------------------------------------
# Color function for signal quality
# -------------------------------------------------
def get_color(quality):
    if quality == "Excellent":
        return "green"
    elif quality == "Good":
        return "blue"
    elif quality == "Poor":
        return "orange"
    else:
        return "red"

# -------------------------------------------------
# Plot route points
# -------------------------------------------------
print("📍 Plotting route points...")

for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=get_color(row["quality"]),
        fill=True,
        fill_opacity=0.7,
        popup=f"Signal: {row['signal_strength']:.1f} dBm<br>Quality: {row['quality']}"
    ).add_to(m)

# -------------------------------------------------
# Draw route line
# -------------------------------------------------
print("🚄 Drawing train route...")

route_coords = data[["lat", "lon"]].values.tolist()

folium.PolyLine(
    route_coords,
    color="black",
    weight=2,
    opacity=0.6
).add_to(m)

# -------------------------------------------------
# Save map
# -------------------------------------------------
output_file = "coverage_map.html"
m.save(output_file)

print(f"✅ Map saved as {output_file}")
print("🎉 Step 4 complete!")