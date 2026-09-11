from flask import Flask, render_template, Response, jsonify
import json
import os
from collections import deque
import cv2
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
VIDEO_PATH = 'video1.mp4'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720
SLOTS_FILE = 'slots.json'
CONFIG_FILE = 'detection_config.json'   # run tune_detection.py to create/tune
SKIP_FRAMES = 3            # run occupancy check every Nth frame, reuse result between

# Defaults if detection_config.json isn't present — tune_detection.py
# will overwrite these via the trackbar UI.
BLOCK_SIZE = 25
OCCUPIED_RATIO = 0.18
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        _cfg = json.load(f)
    BLOCK_SIZE = _cfg['block_size']
    OCCUPIED_RATIO = _cfg['occupied_ratio']

# Per-slot state must agree across this many of the last inference
# passes before the displayed state flips — smooths out single-frame
# noise instead of flickering red/green every inference step.
HISTORY_LEN = 5

# No general object detector: slots are fixed, hand-plotted polygons
# (see plot_slots.py). Occupancy recovered from legacy/detection.py's
# technique — adaptive-threshold the frame to highlight texture (a
# car's body/shadow lines vs smooth asphalt), count thresholded pixels
# inside each slot polygon. Self-contained per frame, no empty-lot
# reference frame needed at all (sidesteps that this lot never fully
# clears in the footage) and no domain-mismatched detector model.
#
# Counted as a ratio of the slot's own pixel area, not a raw count
# like the legacy version — this lot's lanes sit at very different
# distances from the camera, so slot pixel size varies a lot across
# lanes and a single raw-count threshold wouldn't transfer between them.
with open(SLOTS_FILE) as f:
    raw_slots = json.load(f)
slots = [np.array(polygon, dtype=np.int32) for polygon in raw_slots]
slot_areas = [max(cv2.contourArea(s), 1) for s in slots]
TOTAL_SLOTS = len(slots)

slot_histories = [deque(maxlen=HISTORY_LEN) for _ in slots]

# Global variable holding latest occupancy status, read by /api/status
parking_status = {'occupied': 0, 'total': TOTAL_SLOTS, 'slots': [0] * TOTAL_SLOTS}


def generate_frames():
    global parking_status
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0
    last_slot_states = [0] * TOTAL_SLOTS

    while True:
        # Loop video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
        frame_count += 1

        if frame_count % SKIP_FRAMES == 0 or frame_count == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 1)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, 16
            )

            states = []
            for i, slot in enumerate(slots):
                mask = np.zeros(thresh.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [slot], 255)
                cropped = cv2.bitwise_and(thresh, thresh, mask=mask)
                ratio = cv2.countNonZero(cropped) / slot_areas[i]
                raw_occupied = ratio > OCCUPIED_RATIO
                slot_histories[i].append(raw_occupied)
                states.append(1 if sum(slot_histories[i]) > len(slot_histories[i]) / 2 else 0)

            last_slot_states = states
            occupied = sum(last_slot_states)

            parking_status = {
                'occupied': occupied,
                'total': TOTAL_SLOTS,
                'slots': last_slot_states,
            }

        # Draw slot grid overlay (green=free, red=occupied)
        for slot_polygon, state in zip(slots, last_slot_states):
            color = (0, 0, 255) if state else (0, 255, 0)
            cv2.polylines(frame, [slot_polygon], True, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    return jsonify(parking_status)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
