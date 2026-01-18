from joblib import load
from src.config import MODEL_PATH

def load_model():
    return load(MODEL_PATH)

def predict(model, X):
    return model.predict(X)

def predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        raise AttributeError("Esse modelo não suporta predição de probabilidades.")