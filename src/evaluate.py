import mlflow
from sklearn.metrics import roc_auc_score, classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    mlflow.log_metric("roc_auc", roc_auc)
    
    # Extrair apenas as métricas numéricas para logar
    metrics_to_log = {}
    for class_label in ['0', '1']:
        if class_label in report:
            for metric_name in ['precision', 'recall', 'f1-score']:
                key = f"{class_label}_{metric_name}"
                metrics_to_log[key] = report[class_label][metric_name]
    
    # Log das métricas agregadas
    if 'macro avg' in report:
        for metric_name in ['precision', 'recall', 'f1-score']:
            key = f"macro_avg_{metric_name}"
            metrics_to_log[key] = report['macro avg'][metric_name]
    
    if 'weighted avg' in report:
        for metric_name in ['precision', 'recall', 'f1-score']:
            key = f"weighted_avg_{metric_name}"
            metrics_to_log[key] = report['weighted avg'][metric_name]
    
    mlflow.log_metrics(metrics_to_log)

    metrics = {
        "roc_auc": roc_auc,
        "report": report
    }

    return metrics