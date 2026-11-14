"""End-to-end check against a running container, using only the standard library.

    python scripts/smoke_test.py http://localhost:8000 path/to/image.jpg [expected_label]

Waits until /readyz answers, sends the image to /v1/detect and /v1/detect/annotated,
and fails unless the expected label is found.
"""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def wait_ready(base: str, seconds: int = 90) -> dict:
    deadline = time.time() + seconds
    while True:
        try:
            with urllib.request.urlopen(f"{base}/readyz", timeout=3) as r:
                return json.load(r)
        except (urllib.error.URLError, ConnectionError, OSError):
            if time.time() > deadline:
                raise SystemExit(f"service at {base} not ready after {seconds}s")
            time.sleep(2)


def post_image(url: str, path: str) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers.get("Content-Type", ""), r.read()


def main() -> None:
    base, image = sys.argv[1].rstrip("/"), sys.argv[2]
    expected = sys.argv[3] if len(sys.argv) > 3 else None

    print("ready:", wait_ready(base))

    _, raw = post_image(f"{base}/v1/detect", image)
    result = json.loads(raw)
    print(f"detected {result['count']} objects in {result['inference_ms']} ms: {result['counts']}")
    if expected and expected not in result["counts"]:
        raise SystemExit(f"expected to detect '{expected}', got {result['counts']}")

    content_type, jpeg = post_image(f"{base}/v1/detect/annotated", image)
    if content_type != "image/jpeg" or not jpeg.startswith(b"\xff\xd8"):
        raise SystemExit(f"annotated endpoint returned {content_type}")
    print(f"annotated image: {len(jpeg)} bytes")
    print("smoke test passed")


if __name__ == "__main__":
    main()
