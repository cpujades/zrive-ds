import pandas as pd

from fastapi.testclient import TestClient
from unittest.mock import patch

from src.module_6.app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_check():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@patch("src.module_6.src.basket_model.feature_store.FeatureStore.get_features")
@patch("src.module_6.src.basket_model.basket_model.BasketModel.predict")
def test_predict(mock_predict, mock_get_features):
    sample_features = pd.DataFrame({
        "prior_basket_value": [35.20],
        "prior_item_count": [5],
        "prior_regulars_count": [2],
        "regulars_count": [3]
    }, index=["test_user"])
    mock_get_features.return_value = sample_features
    mock_predict.return_value = [123.45]

    response = client.post("/predict/test_user")

    assert response.status_code == 200
    assert response.json() == {"price": 123.45}


@patch("src.module_6.src.basket_model.feature_store.FeatureStore.get_features")
def test_predict_error(mock_get_features):
    mock_get_features.side_effect = Exception("User not found in store")

    response = client.post("/predict/test_user")

    assert response.status_code == 404
    assert "User not found in store" in response.text
