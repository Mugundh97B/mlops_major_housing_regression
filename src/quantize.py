import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing

# Load the trained model
model = joblib.load("models/linear_model.joblib")

# Extract parameters
coef = model.coef_
intercept = model.intercept_

# Save raw (unquantized) parameters
os.makedirs("params", exist_ok=True)
joblib.dump({"coef": coef, "intercept": intercept}, "params/unquant_params.joblib")

# Quantization logic (safe for scalar values)
def quantize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        scale = 1.0  # Avoid divide-by-zero
        quantized = np.zeros_like(arr, dtype=np.uint8)
    else:
        scale = 255 / (arr_max - arr_min)
        quantized = ((arr - arr_min) * scale).astype(np.uint8)
    return quantized, scale, arr_min

# Quantize coefficients and intercept
q_coef, coef_scale, coef_min = quantize(coef)
q_intercept, intercept_scale, intercept_min = quantize(np.array([intercept]))

# Save quantized params
joblib.dump({
    "q_coef": q_coef,
    "q_intercept": q_intercept,
    "coef_scale": coef_scale,
    "coef_min": coef_min,
    "intercept_scale": intercept_scale,
    "intercept_min": intercept_min
}, "params/quant_params.joblib")

# Dequantize and inference
def dequantize(q_arr, scale, min_val):
    return (q_arr.astype(np.float32) / scale) + min_val

deq_coef = dequantize(q_coef, coef_scale, coef_min)
deq_intercept = dequantize(q_intercept, intercept_scale, intercept_min)[0]

# Run inference
data = fetch_california_housing()
X = data.data
y = data.target

y_pred = np.dot(X, deq_coef) + deq_intercept

# Print few outputs
print("Sample predictions (dequantized):", y_pred[:5])
