from fastapi.testclient import TestClient
from task2_server import app

client = TestClient(app)

def test_1_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print(">> Test 1 Passed!")

def test_2_predict_valid_billing():
    response = client.post("/predict", json={"text": "I was charged twice"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Billing"
    assert data["priority"] == "High"
    assert "confidence" in data
    print(">> Test 2 Passed!")

def test_3_predict_empty_validation_422():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    print(">> Test 3 Passed!")

def test_4_predict_batch_bonus():
    payload = {"tickets": ["My internet network is down", "Help with my account charge"]}
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    assert "batch_results" in response.json()
    assert len(response.json()["batch_results"]) == 2
    print(">> Test 4 Passed!")

if __name__ == "__main__":
    test_1_health_endpoint()
    test_2_predict_valid_billing()
    test_3_predict_empty_validation_422()
    test_4_predict_batch_bonus()