"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from cvserve.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["champion"] == "v1-seed42"


def test_predict_by_shape():
    resp = client.post("/predict", json={
        "shape": "circle", "seed": 101, "request_id": "api-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "circle"
    assert 0.0 < data["confidence"] <= 1.0
    assert data["explanation"]
    assert data["served_by"] in ("champion", "challenger", "fallback")
    assert data["latency_ms"] >= 0.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6


def test_predict_by_pixels():
    # In-distribution pixels: a generated circle, flattened. (A
    # noiseless hand-drawn disc is genuinely out-of-distribution for a
    # model trained on noisy images — see test_clean_disc_is_flagged_ood.)
    from cvserve.images import generate_image

    image = generate_image("circle", 101)
    pixels = [v for row in image for v in row]
    resp = client.post("/predict", json={"pixels": pixels})
    assert resp.status_code == 200
    assert resp.json()["label"] == "circle"


def test_clean_disc_is_flagged_ood():
    # A noiseless hand-drawn disc: the model may misclassify it, but the
    # drift monitor must flag it as out-of-distribution.
    pixels = [0.1] * 1024
    for y in range(32):
        for x in range(32):
            if (x - 15.5) ** 2 + (y - 15.5) ** 2 <= 100:
                pixels[y * 32 + x] = 0.9
    resp = client.post("/predict", json={"pixels": pixels})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ood_flag"] is True
    assert data["drift_flag"] is True


def test_predict_rejects_bad_pixels():
    resp = client.post("/predict", json={"pixels": [0.5] * 10})
    assert resp.status_code == 422
    resp = client.post("/predict", json={"pixels": [2.0] * 1024})
    assert resp.status_code == 422


def test_predict_rejects_unknown_shape():
    resp = client.post("/predict", json={"shape": "hexagon"})
    assert resp.status_code == 422


def test_predict_requires_shape_or_pixels():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion"] == "v1-seed42"
    assert data["versions"]


def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "served" in resp.json()
    assert "drift" in resp.json()


def test_promote_without_challenger_is_conflict():
    resp = client.post("/promote", json={"baseline_accuracy": 0.6})
    assert resp.status_code == 409
