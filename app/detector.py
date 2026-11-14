"""YOLOX object detection with ONNX Runtime.

Pre- and post-processing follow the official YOLOX ONNX Runtime demo:
letterbox resize with gray padding, raw BGR pixels (no normalization),
grid decoding of the network output, then per-class non-maximum suppression.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .labels import COCO_CLASSES

STRIDES = (8, 16, 32)
PAD_VALUE = 114


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "box": {k: round(v, 1) for k, v in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2))},
        }


@dataclass(frozen=True)
class Result:
    detections: list[Detection]
    width: int
    height: int
    inference_ms: float


class Detector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray, conf_threshold: float) -> Result: ...


# ---------------------------------------------------------------- pre-processing

def letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad bottom/right with gray. Returns CHW float32 and the scale ratio."""
    h_in, w_in = size
    h, w = image_bgr.shape[:2]
    ratio = min(h_in / h, w_in / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((h_in, w_in, 3), PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return chw, ratio


# ---------------------------------------------------------------- post-processing

def decode_outputs(raw: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Turn raw YOLOX outputs (N, 4+1+C) into absolute cx, cy, w, h in input-image pixels."""
    grids, strides = [], []
    for stride in STRIDES:
        h, w = input_size[0] // stride, input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((h * w, 1), stride))
    grid = np.concatenate(grids, axis=0)
    stride = np.concatenate(strides, axis=0)
    if raw.shape[0] != grid.shape[0]:
        raise ValueError(f"model output has {raw.shape[0]} anchors, expected {grid.shape[0]} for input {input_size}")

    out = raw.astype(np.float32, copy=True)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression on xyxy boxes. Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(
    decoded: np.ndarray,
    ratio: float,
    image_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    labels: tuple[str, ...] = COCO_CLASSES,
) -> list[Detection]:
    """Score = objectness x class probability; keep the best class per box, then NMS per class."""
    img_h, img_w = image_size
    boxes_cxcywh = decoded[:, :4]
    scores_all = decoded[:, 4:5] * decoded[:, 5:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(class_ids)), class_ids]

    mask = scores >= conf_threshold
    if not mask.any():
        return []
    boxes_cxcywh, scores, class_ids = boxes_cxcywh[mask], scores[mask], class_ids[mask]

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)

    detections: list[Detection] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        for k in nms(boxes[idx], scores[idx], iou_threshold):
            i = idx[k]
            b = boxes[i]
            detections.append(Detection(int(cls), labels[int(cls)], float(scores[i]),
                                        float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# ---------------------------------------------------------------- ONNX Runtime model

class YoloxOnnxDetector:
    """Loads a YOLOX .onnx file once and serves predictions; safe to call from several threads."""

    def __init__(self, model_path: str, iou_threshold: float = 0.45, threads: int = 1):
        import onnxruntime as ort  # imported here so tests can run without the model

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = (int(inp.shape[2]), int(inp.shape[3]))
        self.iou_threshold = iou_threshold
        self.name = os.path.splitext(os.path.basename(model_path))[0]

    def detect(self, image_bgr: np.ndarray, conf_threshold: float) -> Result:
        h, w = image_bgr.shape[:2]
        tensor, ratio = letterbox(image_bgr, self.input_size)
        start = time.perf_counter()
        raw = self.session.run(None, {self.input_name: tensor[None, :, :, :]})[0][0]
        inference_ms = (time.perf_counter() - start) * 1000
        decoded = decode_outputs(raw, self.input_size)
        detections = postprocess(decoded, ratio, (h, w), conf_threshold, self.iou_threshold)
        return Result(detections, w, h, inference_ms)
