"""Test the application's response when no image is uploaded."""


def test_acceptance_missing_file(missing_upload_response):
    """Check that a missing upload displays an error message."""
    assert missing_upload_response.status_code == 200
    assert b"File cannot be processed" in missing_upload_response.data
