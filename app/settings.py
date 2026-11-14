"""Settings from environment variables, so the same image runs anywhere."""

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/yolox_nano.onnx")
    conf_threshold: float = _float("CONF_THRESHOLD", 0.35)
    iou_threshold: float = _float("IOU_THRESHOLD", 0.45)
    ort_threads: int = _int("ORT_THREADS", 1)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 10)
    max_image_side: int = _int("MAX_IMAGE_SIDE", 4096)


settings = Settings()
