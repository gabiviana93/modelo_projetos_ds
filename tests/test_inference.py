from src.inference import predict

def test_predict_shape(dummy_model, inference_test_data):
    preds = predict(dummy_model, inference_test_data)

    assert len(preds) == len(inference_test_data)