"""Test the prediction route when an uploaded file is missing."""
# pylint: disable=duplicate-code

def test_missing_file(client):
    """Check that a missing file displays the expected error message."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
