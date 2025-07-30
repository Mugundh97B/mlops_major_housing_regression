import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "linear_model.joblib")
UNQUANT_PATH = os.path.join(ROOT_DIR, "params", "unquant_params.joblib")
QUANT_PATH = os.path.join(ROOT_DIR, "params", "quant_params.joblib")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
coef = model.coef_
intercept = model.intercept_

os.makedirs(os.path.join(ROOT_DIR, "params"), exist_ok=True)
joblib.dump({"coef": coef, "intercept": intercept}, UNQUANT_PATH)


def quantize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        scale = 1.0
        quantized = np.zeros_like(arr, dtype=np.uint8)
    else:
        scale = 255 / (arr_max - arr_min)
        quantized = ((arr - arr_min) * scale).astype(np.uint8)
    return quantized, scale, arr_min

q_coef, coef_scale, coef_min = quantize(coef)
q_intercept, intercept_scale, intercept_min = quantize(np.array([intercept]))


joblib.dump({
    "q_coef": q_coef,
    "q_intercept": q_intercept,
    "coef_scale": coef_scale,
    "coef_min": coef_min,
    "intercept_scale": intercept_scale,
    "intercept_min": intercept_min
}, QUANT_PATH)


def dequantize(q_arr, scale, min_val):
    return (q_arr.astype(np.float32) / scale) + min_val

deq_coef = dequantize(q_coef, coef_scale, coef_min)
deq_intercept = dequantize(q_intercept, intercept_scale, intercept_min)[0]


data = fetch_california_housing()
X, y = data.data, data.target
y_pred = np.dot(X, deq_coef) + deq_intercept

print("Sample predictions (dequantized):", y_pred[:5])
r2_after = r2_score(y, y_pred)
print(f"R² Score after quantization: {r2_after:.4f}")
