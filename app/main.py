"""Object Detection API: FastAPI service around a YOLOX ONNX model."""

from __future__ import annotations

import logging
import time
from collections import Counter as Tally
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from . import __version__
from .detector import Detector, Result, YoloxOnnxDetector
from .drawing import draw
from .labels import COCO_CLASSES, VEHICLE_CLASSES
from .metrics import DETECTIONS, INFERENCE_SECONDS, MODEL_READY, REQUEST_SECONDS, REQUESTS
from .settings import settings

log = logging.getLogger("detector")
STATIC = Path(__file__).parent / "static"


def route_path(request: Request) -> str:
    """Route template such as /v1/detect, so metrics do not explode on unknown URLs."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "unmatched")
    return "unmatched"


def create_app(detector: Detector | None = None) -> FastAPI:
    """Build the app. Tests pass a fake detector; in production the ONNX model is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.detector is None:
            try:
                app.state.detector = YoloxOnnxDetector(settings.model_path, settings.iou_threshold, settings.ort_threads)
                log.info("model %s loaded", settings.model_path)
            except Exception:
                log.exception("could not load model from %s", settings.model_path)
        MODEL_READY.set(1 if app.state.detector else 0)
        yield

    app = FastAPI(
        title="Object Detection API",
        version=__version__,
        description="Detects 80 kinds of objects (people, cars, buses, trucks, motorcycles and more) in an image.",
        lifespan=lifespan,
    )
    app.state.detector = detector

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = route_path(request)
        if path != "/metrics":
            REQUESTS.labels(path, str(response.status_code)).inc()
            REQUEST_SECONDS.labels(path).observe(time.perf_counter() - start)
        return response

    # ------------------------------------------------------------ helpers

    def read_image(file: UploadFile) -> np.ndarray:
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > limit:
            raise HTTPException(413, f"Image is larger than {settings.max_upload_mb} MB.")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(415, "Could not read the image. Send a JPEG, PNG or WebP file.")
        if max(image.shape[:2]) > settings.max_image_side:
            raise HTTPException(413, f"Image sides must be at most {settings.max_image_side} px.")
        return image

    def parse_labels(labels: str | None) -> set[str] | None:
        if not labels:
            return None
        wanted = {x.strip().lower() for x in labels.split(",") if x.strip()}
        if "vehicles" in wanted:
            wanted = (wanted - {"vehicles"}) | VEHICLE_CLASSES
        unknown = wanted - set(COCO_CLASSES)
        if unknown:
            raise HTTPException(422, f"Unknown label(s): {', '.join(sorted(unknown))}. See GET /v1/labels.")
        return wanted

    def run(image: np.ndarray, conf: float | None, labels: str | None) -> Result:
        detector = app.state.detector
        if detector is None:
            raise HTTPException(503, "Model is not loaded yet.")
        wanted = parse_labels(labels)
        result = detector.detect(image, conf if conf is not None else settings.conf_threshold)
        dets = [d for d in result.detections if wanted is None or d.label in wanted]
        INFERENCE_SECONDS.observe(result.inference_ms / 1000)
        for d in dets:
            DETECTIONS.labels(d.label).inc()
        return Result(dets, result.width, result.height, result.inference_ms)

    conf_query = Query(None, ge=0.05, le=0.95, description="Minimum confidence (default 0.35).")
    labels_query = Query(None, description="Only return these labels, comma-separated, e.g. `car,bus` or `vehicles`.")

    # ------------------------------------------------------------ routes

    @app.get("/", include_in_schema=False)
    def demo_page():
        return FileResponse(STATIC / "index.html")

    @app.post("/v1/detect", summary="Detect objects and return them as JSON")
    def detect(file: UploadFile = File(...), conf: float | None = conf_query, labels: str | None = labels_query):
        image = read_image(file)
        result = run(image, conf, labels)
        counts = Tally(d.label for d in result.detections)
        return {
            "model": app.state.detector.name,
            "image": {"width": result.width, "height": result.height},
            "inference_ms": round(result.inference_ms, 1),
            "count": len(result.detections),
            "vehicle_count": sum(n for label, n in counts.items() if label in VEHICLE_CLASSES),
            "counts": dict(counts.most_common()),
            "detections": [d.to_dict() for d in result.detections],
        }

    @app.post(
        "/v1/detect/annotated",
        summary="Detect objects and return the image with boxes drawn",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    def detect_annotated(file: UploadFile = File(...), conf: float | None = conf_query, labels: str | None = labels_query):
        image = read_image(file)
        result = run(image, conf, labels)
        ok, jpeg = cv2.imencode(".jpg", draw(image, result.detections), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(500, "Could not encode the result image.")
        return Response(
            jpeg.tobytes(),
            media_type="image/jpeg",
            headers={"X-Detections": str(len(result.detections)), "X-Inference-Ms": f"{result.inference_ms:.1f}"},
        )

    @app.get("/v1/labels", summary="Labels the model can detect")
    def labels_list():
        return {"labels": list(COCO_CLASSES), "groups": {"vehicles": sorted(VEHICLE_CLASSES)}}

    @app.get("/healthz", summary="Liveness: the process is running")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz", summary="Readiness: the model is loaded and can serve predictions")
    def readyz():
        if app.state.detector is None:
            return JSONResponse({"status": "not ready", "reason": "model not loaded"}, status_code=503)
        return {"status": "ready", "model": app.state.detector.name}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
