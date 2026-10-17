"""Check core application behaviour and image preprocessing."""

from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

import app as app_module
from model import preprocess_img


def make_image():
    """Return an in-memory RGB image with known pixel values."""
    image = BytesIO()
    Image.new("RGB", (32, 48), color=(255, 0, 0)).save(
        image,
        format="PNG",
    )
    image.seek(0)
    return image


def test_home_page_accessible(client):
    """The home page should render the upload form successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"<form" in response.data
    assert b'name="file"' in response.data or b"name='file'" in response.data


def test_valid_upload_returns_prediction(client, monkeypatch):
    """A valid image should display the controlled prediction."""
    predict_mock = Mock(return_value=7)
    monkeypatch.setattr(app_module, "predict_result", predict_mock)

    response = client.post(
        "/prediction",
        data={"file": (make_image(), "valid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed" not in response.data
    assert b"7" in response.data
    predict_mock.assert_called_once()

    prepared_image = predict_mock.call_args.args[0]
    assert prepared_image.shape == (1, 224, 224, 3)


def test_missing_upload_returns_error(client, monkeypatch):
    """A missing file should produce the legacy error response."""
    predict_mock = Mock()
    monkeypatch.setattr(app_module, "predict_result", predict_mock)

    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    predict_mock.assert_not_called()


@pytest.mark.parametrize(
    "contents",
    [b"", b"This is not an image."],
    ids=["empty-file", "non-image-file"],
)
def test_invalid_upload_returns_error(client, monkeypatch, contents):
    """Empty and non-image uploads should not reach prediction."""
    predict_mock = Mock()
    monkeypatch.setattr(app_module, "predict_result", predict_mock)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(contents), "invalid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    predict_mock.assert_not_called()


def test_prediction_failure_preserves_error_response(client, monkeypatch):
    """Unexpected prediction failures should retain the legacy response."""
    predict_mock = Mock(side_effect=RuntimeError("Prediction failed"))
    monkeypatch.setattr(app_module, "predict_result", predict_mock)

    response = client.post(
        "/prediction",
        data={"file": (make_image(), "valid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    predict_mock.assert_called_once()


def test_preprocessing_resizes_and_normalizes_image():
    """Preprocessing should produce a normalized RGB image batch."""
    prepared_image = preprocess_img(make_image())

    assert prepared_image.shape == (1, 224, 224, 3)
    assert np.issubdtype(prepared_image.dtype, np.floating)
    assert np.isfinite(prepared_image).all()

    np.testing.assert_allclose(prepared_image[0, :, :, 0], 1.0)
    np.testing.assert_allclose(prepared_image[0, :, :, 1:], 0.0)
