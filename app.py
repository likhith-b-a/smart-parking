from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
VIDEO_PATH = 'video1.mp4'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720
TOTAL_SLOTS = 30          # configurable capacity of the lot
SKIP_FRAMES = 5           # run YOLO every Nth frame, reuse result between
VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck
CONF_THRESHOLD = 0.4

model = YOLO('yolov8n.pt')

# Global variable holding latest detection status, read by /api/status
parking_status = {'occupied': 0, 'total': TOTAL_SLOTS, 'vehicles': []}


def generate_frames():
    global parking_status
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0
    last_result = None

    while True:
        # Loop video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
        frame_count += 1

        if last_result is None or frame_count % SKIP_FRAMES == 0:
            results = model(frame, classes=VEHICLE_CLASSES, conf=CONF_THRESHOLD, verbose=False)
            last_result = results[0]

            vehicles = []
            for box in last_result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                vehicles.append({
                    'class': model.names[cls_id],
                    'confidence': round(conf, 2),
                    'bbox': [x1, y1, x2, y2]
                })

            occupied = min(len(vehicles), TOTAL_SLOTS)
            parking_status = {
                'occupied': occupied,
                'total': TOTAL_SLOTS,
                'vehicles': vehicles
            }

        annotated = last_result.plot(img=frame)

        ret, buffer = cv2.imencode('.jpg', annotated)
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
