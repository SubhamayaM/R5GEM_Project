import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib

st.set_page_config(layout="wide")
st.title("🚄 AI Railway Network Coverage Analyzer (India)")


# Load model

@st.cache_resource
def load_model():
    return joblib.load("signal_model.pkl")

model = load_model()


# Haversine (vectorized)

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


# Load towers (sample for speed)

@st.cache_data
def load_towers():
    towers = pd.read_csv("data/towers_india.csv")
    towers = towers.sample(50000, random_state=42)
    return towers

towers = load_towers()
tower_lat = towers["lat"].values
tower_lon = towers["lon"].values


# Session state

if "points" not in st.session_state:
    st.session_state.points = []

# -------------------------------------------------
# Map UI
# -------------------------------------------------
st.subheader("📍 Click TWO points on the map")

m = folium.Map(location=[22.5, 78.9], zoom_start=5)

map_data = st_folium(m, height=500, width=900)

# -------------------------------------------------
# Capture clicks
# -------------------------------------------------
if map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if len(st.session_state.points) < 2:
        st.session_state.points.append((lat, lon))

# -------------------------------------------------
# Show selected points
# -------------------------------------------------
if len(st.session_state.points) == 2:
    start = st.session_state.points[0]
    end = st.session_state.points[1]

    st.success("✅ Start and End selected!")

    # -------------------------------------------------
    # Generate route
    # -------------------------------------------------
    route_lats = np.linspace(start[0], end[0], 300)
    route_lons = np.linspace(start[1], end[1], 300)

    route_df = pd.DataFrame({
        "lat": route_lats,
        "lon": route_lons
    })

    # -------------------------------------------------
    # Compute nearest tower distance
    # -------------------------------------------------
    nearest_distances = []

    for _, r in route_df.iterrows():
        dists = haversine_vectorized(
            r["lat"], r["lon"],
            tower_lat, tower_lon
        )
        nearest_distances.append(np.min(dists))

    route_df["nearest_tower_km"] = nearest_distances

    # -------------------------------------------------
    # Predict using ML model
    # -------------------------------------------------
    preds = model.predict(
        route_df[["lat", "lon", "nearest_tower_km"]]
    )

    route_df["signal_strength"] = preds

    # Quality label
    def quality(dbm):
        if dbm > -70:
            return "Excellent"
        elif dbm > -90:
            return "Good"
        elif dbm > -110:
            return "Poor"
        else:
            return "No Signal"

    route_df["quality"] = route_df["signal_strength"].apply(quality)

    # -------------------------------------------------
    # Create result map
    # -------------------------------------------------
    st.subheader("📡 Coverage Result")

    result_map = folium.Map(location=start, zoom_start=6)

    def get_color(q):
        return {
            "Excellent": "green",
            "Good": "blue",
            "Poor": "orange",
            "No Signal": "red"
        }[q]

    # draw route
    coords = route_df[["lat", "lon"]].values.tolist()

    folium.PolyLine(coords, color="black", weight=3).add_to(result_map)

    # draw points
    for _, row in route_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=get_color(row["quality"]),
            fill=True,
            fill_opacity=0.7
        ).add_to(result_map)

    st_folium(result_map, height=500, width=900)

# -------------------------------------------------
# Reset button
# -------------------------------------------------
if st.button("🔄 Reset Points"):
    st.session_state.points = []
