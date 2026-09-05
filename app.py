from flask import Flask, render_template, Response, jsonify
import cv2
import json
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
VIDEO_PATH = 'video1.mp4'
JSON_FILE = 'parking_slots.json'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720

# Global variable to store current status of slots (0 = Free, 1 = Occupied)
# This allows the API to read the status calculated by the video loop
parking_status = [] 

def get_parking_data():
    with open(JSON_FILE, 'r') as f:
        return json.load(f)

parking_data = get_parking_data()

# Initialize status list
parking_status = [0] * len(parking_data)

def generate_frames():
    global parking_status
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Trackbar values (Hardcoded here for web simplicity, 
    # but you can use the values you found in the previous step)
    PIXEL_THRESHOLD = 300 
    
    while True:
        # Loop video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 16
        )

        temp_status = []

        for index, slot in enumerate(parking_data):
            points = np.array(slot, np.int32)
            mask = np.zeros(img_thresh.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            cropped = cv2.bitwise_and(img_thresh, img_thresh, mask=mask)
            count = cv2.countNonZero(cropped)

            if count > PIXEL_THRESHOLD:
                # Occupied
                color = (0, 0, 255) # Red
                temp_status.append(1)
            else:
                # Free
                color = (0, 255, 0) # Green
                temp_status.append(0)

            # Draw on the frame
            cv2.polylines(frame, [points], True, color, 2)
        
        # Update the global status for the API
        parking_status = temp_status

        # Encode frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    # Return the current status as JSON
    occupied = parking_status.count(1)
    total = len(parking_status)
    return jsonify({
        'occupied': occupied,
        'total': total,
        'slots': parking_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)