import joblib
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load("models/linear_model.joblib")

# Load data
data = fetch_california_housing()
X = data.data

# Predict
y_pred = model.predict(X)

# Print sample predictions
print("Sample predictions (original model):", y_pred[:5])
