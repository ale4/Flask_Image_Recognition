"""Provide shared test fixtures for the Flask application."""

import pytest
from app import app


@pytest.fixture
def client():
    """Provide a Flask test client for each test."""
    with app.test_client() as test_client:
        yield test_client
