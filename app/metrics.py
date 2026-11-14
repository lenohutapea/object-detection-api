"""Prometheus metrics exposed at /metrics."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "detector_requests_total",
    "HTTP requests handled, by route and status code.",
    ["route", "status"],
)

REQUEST_SECONDS = Histogram(
    "detector_request_duration_seconds",
    "End-to-end request time: upload, decode, inference, response.",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2.5, 5),
)

INFERENCE_SECONDS = Histogram(
    "detector_inference_duration_seconds",
    "Time spent inside the ONNX model only.",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1),
)

DETECTIONS = Counter(
    "detector_objects_detected_total",
    "Objects detected, by label.",
    ["label"],
)

MODEL_READY = Gauge(
    "detector_model_ready",
    "1 when the model is loaded and the service can answer predictions.",
)
