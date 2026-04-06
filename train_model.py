import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("🤖 Training ML model...")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
data = pd.read_csv("data/coverage_analysis.csv")

print(f"✅ Loaded {len(data)} samples")

# -------------------------------------------------
# Feature selection
# -------------------------------------------------
X = data[["lat", "lon", "nearest_tower_km"]]
y = data["signal_strength"]

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Train model
# -------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✅ Model training complete")

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 MAE: {mae:.2f}")
print(f"📊 R2 Score: {r2:.3f}")

# -------------------------------------------------
# Save model
# -------------------------------------------------
joblib.dump(model, "signal_model.pkl")

print("💾 Model saved as signal_model.pkl")
print("🎉 Step 5 complete!")