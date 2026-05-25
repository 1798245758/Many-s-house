from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == "SUCCESS"

def test_404_returns_error():
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
