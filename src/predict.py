import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "linear_model.joblib")

# Load model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Predict
y_pred = model.predict(X)

# Output
print("Sample predictions (original model):", y_pred[:5])
r2 = r2_score(y, y_pred)
print(f"R² Score (original model): {r2:.4f}")

