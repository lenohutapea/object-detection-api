"""API tests with a fake detector, so they run without the model file."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.detector import Detection, Result
from app.main import create_app


class FakeDetector:
    name = "fake"

    def __init__(self):
        self.last_conf = None

    def detect(self, image_bgr, conf_threshold):
        self.last_conf = conf_threshold
        h, w = image_bgr.shape[:2]
        dets = [
            Detection(2, "car", 0.91, 10, 10, 60, 40),
            Detection(5, "bus", 0.62, 70, 5, 150, 60),
            Detection(0, "person", 0.40, 20, 50, 35, 95),
        ]
        return Result([d for d in dets if d.confidence >= conf_threshold], w, h, 12.3)


def png(width=160, height=100) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((height, width, 3), 200, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def fake():
    return FakeDetector()


@pytest.fixture
def client(fake):
    return TestClient(create_app(detector=fake))


def post(client, path="/v1/detect", data=None, **params):
    return client.post(path, params=params, files={"file": ("test.png", data if data is not None else png(), "image/png")})


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["model"] == "fake"


def test_not_ready_without_model():
    c = TestClient(create_app(detector=None))
    assert c.get("/readyz").status_code == 503
    assert post(c).status_code == 503


def test_detect_returns_objects_counts_and_image_size(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 160, "height": 100}
    assert body["count"] == 3
    assert body["vehicle_count"] == 2
    assert body["counts"] == {"car": 1, "bus": 1, "person": 1}
    assert body["detections"][0]["label"] == "car"
    assert set(body["detections"][0]["box"]) == {"x1", "y1", "x2", "y2"}


def test_confidence_is_passed_to_the_model(client, fake):
    r = post(client, conf=0.6)
    assert r.json()["count"] == 2
    assert fake.last_conf == pytest.approx(0.6)


def test_vehicle_filter(client):
    body = post(client, labels="vehicles").json()
    assert {d["label"] for d in body["detections"]} == {"car", "bus"}


def test_explicit_label_filter(client):
    body = post(client, labels="person").json()
    assert body["count"] == 1 and body["vehicle_count"] == 0


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"labels": "spaceship"}, 422),
        ({"conf": 1.5}, 422),
        ({"data": b"not an image"}, 415),
        ({"data": b""}, 400),
    ],
)
def test_bad_requests_get_clear_errors(client, kwargs, status):
    r = post(client, **kwargs)
    assert r.status_code == status
    assert "detail" in r.json()


def test_annotated_image(client):
    r = post(client, "/v1/detect/annotated")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-detections"] == "3"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (100, 160)


def test_labels_endpoint(client):
    body = client.get("/v1/labels").json()
    assert len(body["labels"]) == 80
    assert "truck" in body["groups"]["vehicles"]


def test_metrics_are_exposed(client):
    post(client)
    text = client.get("/metrics").text
    assert 'detector_requests_total{route="/v1/detect",status="200"}' in text
    assert "detector_inference_duration_seconds_bucket" in text
    assert 'detector_objects_detected_total{label="car"}' in text


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Object Detection API" in r.text
