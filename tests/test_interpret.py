import pytest
import numpy as np
import pandas as pd
from src.interpret import get_feature_names, plot_feature_importance, create_importance_dataframe

def test_get_feature_names():
    """Testa extração de nomes de features do preprocessador"""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['feature1', 'feature2']),
        ('cat', OneHotEncoder(sparse_output=False), ['category'])
    ])
    
    # Fit com dados fake
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'category': ['A', 'B', 'A']
    })
    preprocessor.fit(X)
    
    features = get_feature_names(preprocessor)
    
    assert isinstance(features, list)
    assert len(features) > 0

def test_create_importance_dataframe():
    """Testa criação de DataFrame com importâncias"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessing', StandardScaler()),
        ('model', XGBClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)
    
    df = create_importance_dataframe(pipeline)
    
    assert isinstance(df, pd.DataFrame)
    assert 'feature' in df.columns
    assert 'importance' in df.columns
    assert len(df) == 5

def test_create_importance_dataframe_with_preprocessor():
    """Testa criação de DataFrame com preprocessador"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), list(range(5)))
    ])
    
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', XGBClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)
    
    df = create_importance_dataframe(pipeline, preprocessor=preprocessor)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
