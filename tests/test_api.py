from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

def test_risk_assessment_endpoint():
    sample = {
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 32,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 5000, "BILL_AMT2": 4000, "BILL_AMT3": 3000, "BILL_AMT4": 2000, "BILL_AMT5": 1000, "BILL_AMT6": 500,
        "PAY_AMT1": 5000, "PAY_AMT2": 4000, "PAY_AMT3": 3000, "PAY_AMT4": 2000, "PAY_AMT5": 1000, "PAY_AMT6": 500
    }
    with TestClient(app) as client:
        res = client.post("/assess-risk", json=sample)
        assert res.status_code == 200
        data = res.json()
        assert data["risk_level"] in ["Low", "High"]
        assert "default_probability" in data
        assert len(data["top_adverse_factors"]) == 3