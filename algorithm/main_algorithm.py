"""
Daksha - Onboard Human Detection Algorithm
--------------------------------------------
Runs on the drone's companion computer (e.g. Raspberry Pi / Jetson Nano).
Reads frames from the onboard camera, detects people using OpenCV's HOG
person detector, and reports structured detection info that can be
forwarded to the flight controller / ground station:

    - is_person_present : bool
    - person_count       : int
    - detections         : list of per-person info
        - bbox           : (x, y, w, h) in pixels
        - size_px        : (width, height) in pixels
        - confidence     : detector weight score
        - position       : "left" | "center" | "right" (relative to frame)
        - est_distance_m : rough distance estimate (metres)
        - est_height_m   : rough real-world height estimate (metres)

Distance/height are estimated with the pinhole-camera approximation:
    distance = (real_height * focal_length) / pixel_height
using an assumed average adult height. This is a coarse estimate meant
for situational awareness, not precision measurement.

The display window renders a false-color thermal look by default (blue background,
detected people highlighted solid red) — a cosmetic remap of the regular camera feed,
not a real thermal sensor reading. Pass --no-thermal to see the raw camera feed instead.

Usage:
    python main_algorithm.py [--camera 0] [--focal-length 700] [--headless] [--no-thermal]
"""

import argparse
import json
import time

import cv2
import numpy as np

AVG_HUMAN_HEIGHT_M = 1.7  # assumed average person height used for distance estimation


def classify_position(center_x, frame_width):
    third = frame_width / 3
    if center_x < third:
        return "left"
    if center_x > 2 * third:
        return "right"
    return "center"


def build_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def _non_max_suppression(rects, weights, overlap_thresh=0.5):
    """HOG's detectMultiScale commonly fires several overlapping boxes on the
    same person; keep only the highest-confidence box per overlapping cluster."""
    if len(rects) == 0:
        return rects, weights

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=float)
    scores = np.array(weights, dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)

    keep = []
    while len(order) > 0:
        last = order[-1]
        keep.append(last)
        rest = order[:-1]
        xx1 = np.maximum(x1[last], x1[rest])
        yy1 = np.maximum(y1[last], y1[rest])
        xx2 = np.minimum(x2[last], x2[rest])
        yy2 = np.minimum(y2[last], y2[rest])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[rest]
        order = rest[overlap <= overlap_thresh]

    return rects[keep], weights[keep]


def _tighten_box(x, y, w, h):
    """The default HOG people detector pads its boxes well beyond the actual
    body outline on every side; shrink tightly toward the person's real
    silhouette instead of the raw detector box."""
    pad_w, pad_h_top, pad_h_bottom = int(w * 0.30), int(h * 0.16), int(h * 0.07)
    return (
        x + pad_w,
        y + pad_h_top,
        w - 2 * pad_w,
        h - pad_h_top - pad_h_bottom,
    )


def analyze_frame(hog, frame, focal_length_px):
    frame_height, frame_width = frame.shape[:2]
    rects, weights = hog.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.03, hitThreshold=0.3
    )
    rects, weights = _non_max_suppression(rects, weights)

    detections = []
    for index, ((raw_x, raw_y, raw_w, raw_h), weight) in enumerate(zip(rects, weights)):
        x, y, w, h = _tighten_box(raw_x, raw_y, raw_w, raw_h)
        center_x = x + w / 2
        est_distance_m = None
        if h > 0:
            est_distance_m = round((AVG_HUMAN_HEIGHT_M * focal_length_px) / h, 2)

        detections.append(
            {
                "id": index + 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "size_px": {"width": int(w), "height": int(h)},
                "confidence": round(float(weight), 3),
                "position": classify_position(center_x, frame_width),
                "est_distance_m": est_distance_m,
                "est_height_m": AVG_HUMAN_HEIGHT_M,
            }
        )

    return {
        "timestamp": time.time(),
        "frame_size": {"width": frame_width, "height": frame_height},
        "is_person_present": len(detections) > 0,
        "person_count": len(detections),
        "detections": detections,
    }


PERSON_COLORS = [
    (60, 220, 140), (61, 122, 255), (233, 238, 233),
    (198, 209, 142), (122, 176, 255), (255, 176, 122),
]


def color_for(index):
    return PERSON_COLORS[index % len(PERSON_COLORS)]


THERMAL_BLUE = np.array([200, 60, 20], dtype=np.float32)   # BGR
THERMAL_RED = np.array([20, 25, 235], dtype=np.float32)    # BGR
BLUE_OPACITY = 0.6
RED_OPACITY = 1.0


GRABCUT_MAX_DIM = 120  # the ROI is downscaled to this before segmenting, for speed
GRABCUT_ITERS = 3


def _segment_person(frame, box):
    """Refine a detection box into an actual body-shaped mask with GrabCut, so
    the highlight follows the person's outline instead of filling the whole
    rectangle. Returns (x0, y0, x1, y1, mask) or None if the box is unusable.

    The incoming box is the tightened one from _tighten_box, which sits inside
    the body. GrabCut needs real background around its seed rect to separate
    subject from scene, so widen the box back out before segmenting."""
    x, y, bw, bh = box
    grow_x, grow_y = int(bw * 0.45), int(bh * 0.15)
    x0, y0 = max(x - grow_x, 0), max(y - grow_y, 0)
    x1 = min(x + bw + grow_x, frame.shape[1])
    y1 = min(y + bh + grow_y, frame.shape[0])
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = frame[y0:y1, x0:x1]
    scale = min(1.0, GRABCUT_MAX_DIM / max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
    sh, sw = small.shape[:2]

    # seed rect = the detected body box mapped into this widened ROI, so the
    # band around it is treated as known background
    rx, ry = max(1, int((x - x0) * scale)), max(1, int((y - y0) * scale))
    rw, rh = min(int(bw * scale), sw - rx - 1), min(int(bh * scale), sh - ry - 1)
    if rw < 5 or rh < 5:
        return None
    rect = (rx, ry, rw, rh)

    mask = np.zeros((sh, sw), np.uint8)
    try:
        cv2.grabCut(
            small, mask, rect,
            np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
            GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None  # degenerate ROI; caller falls back to the plain rectangle

    foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    if scale < 1:
        foreground = cv2.resize(
            foreground, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
        )
    return x0, y0, x1, y1, foreground.astype(bool)


def to_thermal(frame, detections):
    """Everywhere is tinted blue by default. Red covers each detected person's
    segmented body outline — driven by the detection result, not by how bright
    that part of the scene happens to be."""
    h, w = frame.shape[:2]
    person_mask = np.zeros((h, w), dtype=bool)

    for det in detections:
        segmented = _segment_person(frame, det["bbox"])
        if segmented is None:
            x, y, bw, bh = det["bbox"]
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + bw, w), min(y + bh, h)
            if x1 > x0 and y1 > y0:
                person_mask[y0:y1, x0:x1] = True
        else:
            x0, y0, x1, y1, body = segmented
            person_mask[y0:y1, x0:x1] |= body

    overlay = np.where(person_mask[:, :, None], THERMAL_RED, THERMAL_BLUE)
    opacity = np.where(person_mask, RED_OPACITY, BLUE_OPACITY)[:, :, None]
    result = frame.astype(np.float32) * (1 - opacity) + overlay * opacity
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_overlay(frame, result):
    for det in result["detections"]:
        x, y, w, h = det["bbox"]
        color = color_for(det["id"] - 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # outline only, no fill
        label = f"#{det['id']} {det['position']} | {det['size_px']['width']}x{det['size_px']['height']}px"
        if det["est_distance_m"] is not None:
            label += f" | ~{det['est_distance_m']}m"
        cv2.putText(
            frame, label, (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    status = f"PERSON DETECTED ({result['person_count']})" if result["is_person_present"] else "NO PERSON"
    color = (60, 220, 140) if result["is_person_present"] else (100, 100, 100)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Daksha onboard human detection")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--focal-length", type=float, default=700.0,
                         help="approximate focal length in pixels for distance estimation")
    parser.add_argument("--headless", action="store_true",
                         help="run without opening a display window; prints JSON per frame")
    parser.add_argument("--no-thermal", dest="thermal", action="store_false",
                         help="show the raw camera feed instead of the thermal false-color look")
    parser.set_defaults(thermal=True)
    args = parser.parse_args()

    hog = build_detector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-style view
            result = analyze_frame(hog, frame, args.focal_length)

            if args.headless:
                print(json.dumps(result))
            else:
                display_frame = to_thermal(frame, result["detections"]) if args.thermal else frame
                display_frame = draw_overlay(display_frame, result)
                cv2.imshow("Daksha - Human Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
