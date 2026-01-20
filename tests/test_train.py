from src.train import train_model
from src.features import build_preprocessor

def test_pipeline_creation(preprocessor_features, model_params):
    preprocessor = build_preprocessor(
        numeric_features=preprocessor_features["numeric_features"],
        categorical_features=preprocessor_features["categorical_features"]
    )

    pipeline = train_model(preprocessor, model_params)

    assert "preprocessing" in pipeline.named_steps
    assert "model" in pipeline.named_steps
