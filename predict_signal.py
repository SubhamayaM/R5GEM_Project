import joblib
import pandas as pd

print("📡 AI Signal Prediction System")

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("signal_model.pkl")
print("✅ Model loaded")

# -------------------------------------------------
# User input
# -------------------------------------------------
try:
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))
    dist = float(input("Enter distance to nearest tower (km): "))
except:
    print("❌ Invalid input")
    exit()

# -------------------------------------------------
# Prepare input
# -------------------------------------------------
input_df = pd.DataFrame([[lat, lon, dist]],
                        columns=["lat", "lon", "nearest_tower_km"])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
pred_signal = model.predict(input_df)[0]

# Quality classification
def signal_quality(dbm):
    if dbm > -70:
        return "Excellent"
    elif dbm > -90:
        return "Good"
    elif dbm > -110:
        return "Poor"
    else:
        return "No Signal"

quality = signal_quality(pred_signal)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("\n🔮 Prediction Result")
print(f"📶 Signal Strength: {pred_signal:.2f} dBm")
print(f"📡 Quality: {quality}")
print("🎉 Live prediction complete!")