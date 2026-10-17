"""Test image preprocessing and hand-digit predictions."""

import numpy as np
import pytest

from model import preprocess_img, predict_result


def test_preprocess_img():
    """Check the prepared image's shape and pixel values."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    assert processed_img.shape == (1, 224, 224, 3), (
        "Processed image shape should be (1, 224, 224, 3)"
    )

    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, (
        "Image pixel values should be normalized between 0 and 1"
    )


def test_predict_result():
    """Check that prediction returns an integer class index."""
    img_path = "test_images/4/Sign 4 (92).jpeg"
    processed_img = preprocess_img(img_path)
    prediction = predict_result(processed_img)

    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    assert isinstance(prediction, (int, np.integer)), (
        "Prediction should be an integer class index"
    )


def test_invalid_image_path():
    """Check that a missing image raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")


def test_image_shape_on_prediction():
    """Check that another prepared image produces an integer prediction."""
    img_path = "test_images/5/Sign 5 (86).jpeg"
    processed_img = preprocess_img(img_path)
    prediction = predict_result(processed_img)

    assert isinstance(prediction, (int, np.integer)), (
        "The prediction should be an integer"
    )


def test_model_predictions_consistency():
    """Check that repeated predictions for the same image agree."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)
    predictions = [predict_result(processed_img) for _ in range(5)]

    assert all(prediction == predictions[0] for prediction in predictions), (
        "Predictions for the same input should be consistent"
    )
