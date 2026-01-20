import pytest
import numpy as np
from src.inference import predict, predict_proba, load_model

def test_predict_shape(dummy_model, inference_test_data):
    preds = predict(dummy_model, inference_test_data)
    assert len(preds) == len(inference_test_data)

def test_predict_proba_shape(dummy_model, inference_test_data):
    """Testa se predict_proba retorna o formato correto"""
    probas = predict_proba(dummy_model, inference_test_data)
    assert probas.shape[0] == len(inference_test_data)
    assert probas.shape[1] == 2  # Classificação binária

def test_predict_proba_values(dummy_model, inference_test_data):
    """Testa se as probabilidades somam 1"""
    probas = predict_proba(dummy_model, inference_test_data)
    np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(inference_test_data)))

def test_predict_proba_no_support():
    """Testa erro quando modelo não suporta predict_proba"""
    class MockModel:
        def predict(self, X):
            return np.array([0, 1])
    
    model = MockModel()
    X = np.array([[1], [2]])
    
    with pytest.raises(AttributeError, match="não suporta predição de probabilidades"):
        predict_proba(model, X)

def test_predict_returns_array(dummy_model, inference_test_data):
    """Testa se predict retorna numpy array"""
    preds = predict(dummy_model, inference_test_data)
    assert isinstance(preds, np.ndarray)