"""Provide shared test fixtures for the Flask application."""

import pytest
from app import app

@pytest.fixture
def client():
    """Provide a Flask test client for each test."""
    with app.test_client() as test_client:
        yield test_client
@pytest.fixture
def missing_upload_response(client):
    """Return the response to a prediction request without a file."""
    return client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )
