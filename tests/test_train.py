import pytest
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

def test_data_loading():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0
    assert data.target.shape[0] > 0

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_training():
    model = joblib.load("models/linear_model.joblib")
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")

def test_model_accuracy():
    data = fetch_california_housing()
    X, y = data.data, data.target
    #print(f'the length - {len(X)} the lenth of the - {len(X[0])} the first item- {X[0]}' )
    #print("*********************************")
    #print(data)
    model = joblib.load("models/linear_model.joblib")
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5  # Minimum performance threshold
