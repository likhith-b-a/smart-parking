"""
Interactive tuner for the adaptive-threshold occupancy detector (see
legacy/detection.py for the original technique this is recovered
from). Per slot: adaptive-threshold the frame to highlight texture
(cars are textured, asphalt is smooth), count thresholded pixels
inside the slot polygon, divide by the slot's own area — a ratio, not
a raw count, since slots in this lot span very different pixel sizes
across lanes at different distances from the camera.

Live-adjust the two trackbars until slots flip red/green correctly
across a range of frames, then press 's' to save detection_config.json
— app.py loads it if present.

Usage:
    python tune_detection.py [video_path]

Controls:
    d / a    step forward / back 30 frames
    s        save current trackbar values to detection_config.json
    q        quit
"""
import sys
import json
import cv2
import numpy as np

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else 'video1.mp4'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720
SLOTS_FILE = 'slots.json'
OUT_FILE = 'detection_config.json'
WINDOW = 'Tune Detection - a/d jump 30 frames, s=save, q=quit'


def nothing(x):
    pass


def main():
    with open(SLOTS_FILE) as f:
        raw_slots = json.load(f)
    slots = [np.array(s, dtype=np.int32) for s in raw_slots]
    areas = [max(cv2.contourArea(s), 1) for s in slots]

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar('Ratio x1000', WINDOW, 180, 1000, nothing)
    cv2.createTrackbar('Block Size', WINDOW, 25, 101, nothing)

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

        block_size = cv2.getTrackbarPos('Block Size', WINDOW)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        ratio_threshold = cv2.getTrackbarPos('Ratio x1000', WINDOW) / 1000

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 16
        )

        disp = frame.copy()
        occupied_count = 0
        for slot, area in zip(slots, areas):
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [slot], 255)
            cropped = cv2.bitwise_and(thresh, thresh, mask=mask)
            count = cv2.countNonZero(cropped)
            ratio = count / area
            occupied = ratio > ratio_threshold
            occupied_count += occupied

            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(disp, [slot], True, color, 2)
            cx, cy = slot.mean(axis=0).astype(int)
            cv2.putText(disp, f'{ratio:.2f}', (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(disp, f'Occupied: {occupied_count}/{len(slots)}  Frame {frame_idx}/{total_frames}',
                    (10, FIXED_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(WINDOW, disp)
        cv2.imshow('Threshold Mask', thresh)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('d'):
            frame_idx = min(frame_idx + 30, total_frames - 1)
        elif key == ord('a'):
            frame_idx = max(frame_idx - 30, 0)
        elif key == ord('s'):
            with open(OUT_FILE, 'w') as f:
                json.dump({'block_size': block_size, 'occupied_ratio': ratio_threshold}, f, indent=2)
            print(f'Saved block_size={block_size}, occupied_ratio={ratio_threshold} to {OUT_FILE}')
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
