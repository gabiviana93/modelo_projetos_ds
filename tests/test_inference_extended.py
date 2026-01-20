import pytest
import numpy as np
import os
import tempfile
from joblib import dump, load
from src.inference import predict, predict_proba, load_model

def test_load_model_from_file(tmp_path):
    """Testa carregar modelo de arquivo"""
    from sklearn.dummy import DummyClassifier
    
    # Criar e salvar modelo temporário
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    
    model_path = os.path.join(tmp_path, "test_model.pkl")
    dump(model, model_path)
    
    # Mock do MODEL_PATH
    import src.inference as inf_module
    original_path = inf_module.MODEL_PATH
    inf_module.MODEL_PATH = model_path
    
    try:
        loaded_model = load_model()
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    finally:
        inf_module.MODEL_PATH = original_path

def test_predict_with_different_data_types(dummy_model):
    """Testa predict com diferentes tipos de dados"""
    # Array 2D
    X_array = np.array([[1], [2], [3]])
    preds = predict(dummy_model, X_array)
    assert len(preds) == 3
    
    # Apenas certifique-se de que funciona
    assert preds is not None

def test_predict_proba_sum_to_one(dummy_model):
    """Testa se probabilidades somam 1"""
    X = np.array([[1], [2], [3]])
    probas = predict_proba(dummy_model, X)
    
    # Cada linha deve somar 1
    for row in probas:
        assert abs(sum(row) - 1.0) < 0.001

def test_predict_consistency(dummy_model):
    """Testa consistência de predições"""
    X = np.array([[1], [2]])
    
    # Fazer predições múltiplas vezes
    pred1 = predict(dummy_model, X)
    pred2 = predict(dummy_model, X)
    
    # Devem ser idênticas
    np.testing.assert_array_equal(pred1, pred2)

def test_predict_proba_consistency(dummy_model):
    """Testa consistência de probabilidades"""
    X = np.array([[1], [2]])
    
    # Fazer predições múltiplas vezes
    proba1 = predict_proba(dummy_model, X)
    proba2 = predict_proba(dummy_model, X)
    
    # Devem ser idênticas
    np.testing.assert_array_almost_equal(proba1, proba2)
