import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from joblib import dump
from src.config import MODEL_PATH, MODEL_PARAMS
from src.logger import setup_logger

logger = setup_logger(__name__)

def train_model(preprocessor, X_train=None, y_train=None, params=None):
    """
    Treina um modelo XGBoost com MLflow tracking.
    
    Args:
        preprocessor: Pipeline de pré-processamento
        X_train: Dados de treinamento (opcional)
        y_train: Labels de treinamento (opcional)
        params: Parâmetros do modelo (opcional, usa MODEL_PARAMS se None)
    
    Returns:
        Pipeline treinado
    """
    
    if params is None:
        params = MODEL_PARAMS
    
    # Registrar parâmetros no MLflow (se houver uma run ativa)
    if mlflow.active_run():
        mlflow.log_params(params)
    
    model = XGBClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )
    
    # Treinar modelo se dados forem fornecidos
    if X_train is not None and y_train is not None:
        logger.info("Iniciando treinamento", extra={
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'model_type': 'XGBClassifier'
        })
        pipeline.fit(X_train, y_train)
        if mlflow.active_run():
            mlflow.log_metric("training_samples", X_train.shape[0])
        logger.info("Treinamento concluído")

    return pipeline


def cross_validate_model(pipeline, X_train, y_train, n_splits=5):
    """
    Valida modelo com K-Fold estratificado.
    
    Realiza validação cruzada para estimar performance de forma robusta,
    reduzindo a variância causada pelo split inicial.
    
    Args:
        pipeline: Pipeline sklearn treinado ou não-treinado
        X_train: Features de treino
        y_train: Target de treino
        n_splits: Número de folds (padrão 5)
    
    Returns:
        dict com scores de cada métrica (média e desvio padrão)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'f1_weighted': 'f1_weighted',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
    }
    
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1  # Paralelizar para mais rápido
    )
    
    # Consolidar resultados
    summary = {}
    for metric_name in scoring.keys():
        test_scores = cv_results[f'test_{metric_name}']
        train_scores = cv_results[f'train_{metric_name}']
        
        summary[metric_name] = {
            'test_mean': float(test_scores.mean()),
            'test_std': float(test_scores.std()),
            'train_mean': float(train_scores.mean()),
            'train_std': float(train_scores.std()),
            'scores': test_scores.tolist()
        }
    
    # Log no MLflow se houver uma run ativa
    if mlflow.active_run():
        for metric_name, stats in summary.items():
            mlflow.log_metric(f"cv_{metric_name}_test_mean", stats['test_mean'])
            mlflow.log_metric(f"cv_{metric_name}_test_std", stats['test_std'])
    
    return summary

def save_model(model, model_name="xgboost-pipeline", X_example=None):
    """
    Salva o modelo localmente e registra no MLflow.
    
    Args:
        model: Pipeline treinado
        model_name: Nome do modelo no MLflow
        X_example: DataFrame de exemplo para inferência de signature
    """
    # Salvar localmente
    dump(model, MODEL_PATH)
    
    # Adicionar tags e descrição
    if mlflow.active_run():
        mlflow.set_tag("model_type", "xgboost_classifier")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("preprocessing", "StandardScaler + OneHotEncoder")
        mlflow.log_param("model_description", "XGBoost model pipeline with preprocessing")
    
    # Registrar artefato com descrição
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    
    # Registrar modelo no MLflow com input_example para inferência de signature
    if X_example is not None:
        mlflow.sklearn.log_model(
            model, 
            artifact_path="sklearn-model",
            registered_model_name=model_name,
            input_example=X_example.iloc[:1] if hasattr(X_example, 'iloc') else X_example[:1]
        )
    else:
        mlflow.sklearn.log_model(
            model, 
            artifact_path="sklearn-model",
            registered_model_name=model_name
        )

