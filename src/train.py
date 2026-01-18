import mlflow
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from joblib import dump
from src.config import MODEL_PATH, MODEL_PARAMS

def train_model(preprocessor, X_train=None, y_train=None):
    """
    Treina um modelo XGBoost com MLflow tracking.
    
    Args:
        preprocessor: Pipeline de pré-processamento
        X_train: Dados de treinamento (opcional)
        y_train: Labels de treinamento (opcional)
    
    Returns:
        Pipeline treinado
    """
    
    # Registrar parâmetros no MLflow (se houver uma run ativa)
    if mlflow.active_run():
        mlflow.log_params(MODEL_PARAMS)
    
    model = XGBClassifier(**MODEL_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )
    
    # Treinar modelo se dados forem fornecidos
    if X_train is not None and y_train is not None:
        pipeline.fit(X_train, y_train)
        if mlflow.active_run():
            mlflow.log_metric("training_samples", X_train.shape[0])

    return pipeline

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

