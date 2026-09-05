import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO_PATH = "video1.mp4"
OUTPUT_FILE = "parking_slots_auto.json"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def order_points(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered.astype(int).tolist()


def polygon_area(points):
    contour = np.array(points, dtype=np.int32)
    return abs(cv2.contourArea(contour))


def centroid(points):
    pts = np.array(points, dtype=np.float32)
    return pts.mean(axis=0)


def is_duplicate(candidate, accepted, distance_threshold):
    center = centroid(candidate)
    area = polygon_area(candidate)

    for current in accepted:
        current_center = centroid(current)
        current_area = polygon_area(current)
        if np.linalg.norm(center - current_center) < distance_threshold:
            if abs(area - current_area) / max(area, current_area, 1) < 0.35:
                return True
    return False


def collect_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")

    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Highlight painted parking lines while reducing texture noise from asphalt.
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5,
    )

    edges = cv2.Canny(thresh, 80, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, thresh, closed


def detect_slots(frame, min_area, max_area):
    _, _, mask = preprocess(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 80:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(-1, 2)
        area = abs(cv2.contourArea(points))
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(points.astype(np.float32))
        width, height = rect[1]
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.3 or aspect_ratio > 4.8:
            continue

        box = cv2.boxPoints(rect).astype(int).tolist()
        ordered = order_points(box)

        if is_duplicate(ordered, detected, distance_threshold=35):
            continue

        detected.append(ordered)

    detected.sort(key=lambda pts: (centroid(pts)[1], centroid(pts)[0]))
    return detected, mask


def draw_preview(frame, slots):
    preview = frame.copy()
    for index, slot in enumerate(slots, start=1):
        points = np.array(slot, np.int32)
        cv2.polylines(preview, [points], True, (0, 255, 0), 2)

        cx, cy = centroid(slot).astype(int)
        cv2.putText(
            preview,
            str(index),
            (cx - 10, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"Detected slots: {len(slots)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def save_slots(slots, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(slots, file, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimental automatic parking-slot detector."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index used for automatic slot discovery",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1200,
        help="Minimum contour area to keep as a candidate slot",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=25000,
        help="Maximum contour area to keep as a candidate slot",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="JSON file where detected slots will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)

    frame = collect_frame(video_path, args.frame)
    slots, mask = detect_slots(frame, args.min_area, args.max_area)
    preview = draw_preview(frame, slots)

    print(f"Video: {video_path}")
    print(f"Frame index: {args.frame}")
    print(f"Detected slots: {len(slots)}")
    print("Controls: 's' save detected slots, 'q' quit")

    while True:
        cv2.imshow("Auto Slot Detection Preview", preview)
        cv2.imshow("Auto Slot Detection Mask", mask)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            save_slots(slots, args.output)
            print(f"Saved {len(slots)} slots to {args.output}")
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
