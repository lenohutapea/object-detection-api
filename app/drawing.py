"""Draw detections on an image for the /v1/detect/annotated endpoint."""

import cv2
import numpy as np

from .detector import Detection


def _color(class_id: int) -> tuple[int, int, int]:
    # Stable, well-spread colors per class (BGR)
    hue = (class_id * 37) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    thickness = max(2, round(min(out.shape[:2]) / 300))
    font_scale = max(0.5, min(out.shape[:2]) / 900)
    for d in detections:
        color = _color(d.class_id)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y_top = max(0, p1[1] - th - base - 4)
        cv2.rectangle(out, (p1[0], y_top), (p1[0] + tw + 6, y_top + th + base + 4), color, -1)
        cv2.putText(out, text, (p1[0] + 3, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out
