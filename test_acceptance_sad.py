"""Test the application's response when no image is uploaded."""


def test_acceptance_missing_file(client):
    """Check that a missing upload displays an error message."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
