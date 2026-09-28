import pytest
from fastapi.testclient import TestClient

from api.main import app, forecaster

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Thailand" in body["countries"] and "Rice, paddy" in body["crops"]


def test_predict_known_year_returns_model_baseline_and_actual():
    r = client.post("/predict", json={"area": "Thailand", "item": "Rice, paddy", "year": 2012})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction_hg_ha"] > 0
    assert body["baseline_hg_ha"] == pytest.approx(31867)  # Thailand rice, 2011
    assert body["actual_hg_ha"] == 31865
    assert body["model_trained_through"] == 2008


def test_predict_next_year_has_no_actual():
    r = client.post("/predict", json={"area": "Thailand", "item": "Rice, paddy", "year": 2014})
    assert r.status_code == 200
    assert r.json()["actual_hg_ha"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"area": "Atlantis", "item": "Rice, paddy", "year": 2012},  # unknown country
        {"area": "Thailand", "item": "Coffee", "year": 2012},  # unknown crop
        {"area": "Thailand", "item": "Rice, paddy", "year": 2030},  # out of range
        {"area": "Thailand", "item": "Rice, paddy"},  # missing field
    ],
)
def test_invalid_input_is_422(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_known_country_and_crop_without_history_is_404():
    pairs = set(forecaster.panel[["Area", "Item"]].itertuples(index=False, name=None))
    area, item = next((a, i) for a in forecaster.areas for i in forecaster.items if (a, i) not in pairs)
    r = client.post("/predict", json={"area": area, "item": item, "year": 2012})
    assert r.status_code == 404
