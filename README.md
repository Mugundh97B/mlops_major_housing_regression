# mlops_major_housing_regression

This project is built for the **MLOps Major Assignment** at IIT Jodhpur. It demonstrates a complete end-to-end MLOps pipeline using **Linear Regression** on the **California Housing dataset**. The project includes model training, evaluation, manual quantization, Dockerization, and CI/CD with GitHub Actions.

---

## Model Summary

- **Algorithm**: Linear Regression (sklearn.linear_model.LinearRegression)
- **Dataset**: California Housing (from sklearn.datasets)
- **Performance**:
  - R² Score: 0.6062
  - MSE: 0.5243

---

## Testing

Unit tests for pipeline components are written using `pytest`.

---

## Tests Included:
Dataset loading check

Model instance check

Model coefficient check

R² score validation (must be > 0.5).

---

## Manual Quantization
Performed manually in src/quantize.py:

Extracts model coefficients and intercept

Saves raw parameters to params/unquant_params.joblib

Applies unsigned 8-bit quantization

Saves quantized parameters to params/quant_params.joblib

Performs inference with dequantized weights
Sample predictions (dequantized): [3.05, -2.59, 2.09, 1.50, 0.66]


---

## Dockerization
Dockerfile builds a lightweight container that:

Installs required packages

Sample predictions (original model): [4.13, 3.97, 3.67, 3.24, 2.41]


---

## GitHub Actions
Three CI jobs triggered on every push to main:

Job	Description
Test-suite               ---	Runs unit tests with pytest. 
Train-and-quantize	     ---  Trains & quantizes model. 
Build-and-test-container ---	Builds Docker image and runs it. 

