"""Unit tests for pre- and post-processing. No model file needed."""

import math

import numpy as np
import pytest

from app.detector import decode_outputs, letterbox, nms, postprocess
from app.labels import COCO_CLASSES

INPUT = (416, 416)
ANCHORS = sum((416 // s) ** 2 for s in (8, 16, 32))  # 3549


def blank_raw() -> np.ndarray:
    raw = np.zeros((ANCHORS, 85), dtype=np.float32)
    raw[:, 2:4] = math.log(1.0)  # w = h = stride
    return raw


def anchor_index(stride: int, gx: int, gy: int) -> int:
    offset = 0
    for s in (8, 16, 32):
        if s == stride:
            return offset + gy * (416 // s) + gx
        offset += (416 // s) ** 2
    raise ValueError(stride)


def put(raw, stride, gx, gy, size_px, obj, cls, prob, dx=0.5, dy=0.5):
    i = anchor_index(stride, gx, gy)
    raw[i, 0], raw[i, 1] = dx, dy
    raw[i, 2] = raw[i, 3] = math.log(size_px / stride)
    raw[i, 4] = obj
    raw[i, 5 + cls] = prob


# ------------------------------------------------------------------ letterbox

def test_letterbox_keeps_aspect_ratio_and_pads_gray():
    img = np.full((200, 400, 3), 255, dtype=np.uint8)  # wide image
    chw, ratio = letterbox(img, INPUT)
    assert chw.shape == (3, 416, 416)
    assert chw.dtype == np.float32
    assert ratio == pytest.approx(416 / 400)
    assert chw[0, 0, 0] == 255           # image area
    assert chw[0, 415, 0] == 114          # padding at the bottom


# ------------------------------------------------------------------ decoding

def test_decode_places_box_on_its_grid_cell():
    raw = blank_raw()
    put(raw, stride=8, gx=10, gy=5, size_px=80, obj=1, cls=2, prob=1)
    out = decode_outputs(raw, INPUT)
    i = anchor_index(8, 10, 5)
    assert out[i, 0] == pytest.approx((10 + 0.5) * 8)   # cx = 84
    assert out[i, 1] == pytest.approx((5 + 0.5) * 8)    # cy = 44
    assert out[i, 2] == pytest.approx(80)
    assert out[i, 3] == pytest.approx(80)


def test_decode_rejects_wrong_anchor_count():
    with pytest.raises(ValueError):
        decode_outputs(np.zeros((100, 85), dtype=np.float32), INPUT)


# ------------------------------------------------------------------ NMS

def test_nms_removes_overlapping_boxes_and_keeps_separate_ones():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, iou_threshold=0.45) == [0, 2]


def test_nms_keeps_everything_when_nothing_overlaps():
    boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15]], dtype=np.float32)
    assert sorted(nms(boxes, np.array([0.5, 0.6]), 0.45)) == [0, 1]


# ------------------------------------------------------------------ full post-processing

def test_postprocess_scores_labels_scales_and_deduplicates():
    raw = blank_raw()
    car = COCO_CLASSES.index("car")
    dog = COCO_CLASSES.index("dog")
    put(raw, 8, 10, 5, 80, obj=0.9, cls=car, prob=0.8)            # car, score 0.72
    put(raw, 8, 11, 5, 80, obj=0.9, cls=car, prob=0.7)            # same car, shifted 8 px -> suppressed
    put(raw, 32, 8, 8, 64, obj=0.95, cls=dog, prob=0.9)           # dog, score 0.855
    put(raw, 16, 2, 2, 32, obj=0.2, cls=car, prob=0.5)            # score 0.1 -> below threshold

    decoded = decode_outputs(raw, INPUT)
    dets = postprocess(decoded, ratio=0.5, image_size=(832, 832), conf_threshold=0.35, iou_threshold=0.45)

    assert [d.label for d in dets] == ["dog", "car"]               # sorted by confidence
    dog_det, car_det = dets
    assert dog_det.confidence == pytest.approx(0.855, abs=1e-4)
    assert car_det.confidence == pytest.approx(0.72, abs=1e-4)
    # car: cx 84, cy 44, 80x80 in the 416 input -> divided by ratio 0.5 -> original pixels
    assert (car_det.x1, car_det.y1, car_det.x2, car_det.y2) == pytest.approx((88, 8, 248, 168))


def test_postprocess_clips_boxes_to_the_image():
    raw = blank_raw()
    put(raw, 32, 0, 0, 200, obj=1, cls=0, prob=1)                  # large box at the top-left corner
    dets = postprocess(decode_outputs(raw, INPUT), 1.0, (300, 300), 0.35, 0.45)
    assert dets[0].x1 == 0 and dets[0].y1 == 0


def test_postprocess_returns_nothing_below_threshold():
    raw = blank_raw()
    put(raw, 8, 3, 3, 40, obj=0.3, cls=0, prob=0.3)
    assert postprocess(decode_outputs(raw, INPUT), 1.0, (416, 416), 0.35, 0.45) == []


def test_detection_to_dict_is_json_friendly():
    raw = blank_raw()
    put(raw, 8, 10, 5, 80, obj=1, cls=2, prob=1)
    d = postprocess(decode_outputs(raw, INPUT), 1.0, (416, 416), 0.35, 0.45)[0].to_dict()
    assert d["label"] == "car"
    assert set(d["box"]) == {"x1", "y1", "x2", "y2"}
