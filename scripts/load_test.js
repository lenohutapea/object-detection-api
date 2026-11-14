// k6 load test: ramps up concurrent users so the HorizontalPodAutoscaler adds pods.
//   docker run --rm --network host -v "$PWD/scripts:/scripts" grafana/k6:0.54.0 run /scripts/load_test.js
import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://localhost";
const image = open(__ENV.IMAGE || "./dog.jpg", "b");

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "30s", target: 4 },   // warm up
        { duration: "2m", target: 12 },   // push CPU past the autoscaling target
        { duration: "1m", target: 12 },   // hold while new pods join
        { duration: "30s", target: 0 },   // ramp down
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],        // fewer than 1% errors
    http_req_duration: ["p(95)<2000"],     // 95% of requests under 2 s
  },
};

export default function () {
  const res = http.post(`${BASE_URL}/v1/detect`, {
    file: http.file(image, "dog.jpg", "image/jpeg"),
  });
  check(res, {
    "status is 200": (r) => r.status === 200,
    "objects found": (r) => r.status === 200 && r.json("count") > 0,
  });
  sleep(0.2);
}
