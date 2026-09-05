import cv2
import json
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = 'clip8_fixed.mp4'
JSON_FILE = 'parking_slots.json'

# MUST match the size you used in the annotation tool
FIXED_WIDTH = 1280
FIXED_HEIGHT = 720

def main():
    # 1. Load the Parking Slot Coordinates
    with open(JSON_FILE, 'r') as f:
        parking_data = json.load(f)
    print(f"Loaded {len(parking_data)} parking slots.")

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # Optional: Loop video forever for testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 3. RESIZE FRAME (Critical Step)
        # If we don't resize, the coordinates from the JSON won't match the pixels
        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

        # 4. Draw the Lots
        for index, slot in enumerate(parking_data):
            # Convert list of points to NumPy array (Required by OpenCV)
            points = np.array(slot, np.int32)
            points = points.reshape((-1, 1, 2))

            # Draw the polygon (Blue, thickness 2)
            cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)

            # Optional: Add ID number to the center of the slot
            # Calculate center point for text placement
            moment = cv2.moments(points)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                cv2.putText(frame, str(index + 1), (cx - 5, cy + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)

        cv2.imshow("Parking Management System", frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()