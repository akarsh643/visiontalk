import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api import app

def test_root_endpoint():
    """Verify root documentation landing route."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "interactive_docs" in data

def test_health_endpoint():
    """Verify health check endpoint returns status and boolean flag."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "degraded"]

def test_predict_invalid_file_type():
    """Ensure API rejects non-image formats with HTTP 400."""
    with TestClient(app) as client:
        fake_txt_file = io.BytesIO(b"this is a dummy text file, not an image.")
        response = client.post(
            "/predict",
            files={"file": ("test.txt", fake_txt_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "detail" in response.json()

def test_predict_valid_image():
    """Ensure API processes a synthesized JPEG image and returns caption structure."""
    with TestClient(app) as client:
        # Create a minimal valid RGB image in memory
        img = Image.new("RGB", (224, 224), color="blue")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test_blue.jpg", img_byte_arr, "image/jpeg")},
            params={"beam": 1, "penalty": 0.5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "caption" in data
        assert isinstance(data["caption"], str)
        assert len(data["caption"]) > 0