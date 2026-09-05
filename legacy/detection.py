import cv2
import json
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = 'clip6_fixed.mp4'
JSON_FILE = 'parking_slots.json'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720

class ParkingDetector:
    def __init__(self):
        with open(JSON_FILE, 'r') as f:
            self.parking_data = json.load(f)
        
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Create a GUI window for tweaking parameters
        cv2.namedWindow("Control Panel")
        cv2.resizeWindow("Control Panel", 400, 100)
        # 1. Threshold: How many white pixels define a "Car"?
        cv2.createTrackbar("Pixel Threshold", "Control Panel", 300, 2000, self.nothing)
        # 2. Sensitivity: For adaptive thresholding (Light/Dark sensitivity)
        # Odd numbers only for adaptiveThreshold block size!
        cv2.createTrackbar("Block Size", "Control Panel", 25, 50, self.nothing) 

    def nothing(self, x):
        pass

    def check_parking_space(self, img_processed, img_display):
        occupied_count = 0
        
        pixel_threshold = cv2.getTrackbarPos("Pixel Threshold", "Control Panel")
        
        for index, slot in enumerate(self.parking_data):
            # 1. Extract the Polygon
            points = np.array(slot, np.int32)
            
            # 2. Create a mask for this specific parking spot
            # (This isolates the spot from the rest of the image)
            mask = np.zeros(img_processed.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            # 3. Bitwise AND to get the actual pixels of the spot
            # Now we have an image that is BLACK everywhere except this one spot
            cropped = cv2.bitwise_and(img_processed, img_processed, mask=mask)
            
            # 4. Count non-zero (white) pixels inside this spot
            count = cv2.countNonZero(cropped)
            
            # 5. Logic: Car or No Car?
            # Green = Empty, Red = Occupied
            if count > pixel_threshold:
                color = (0, 0, 255) # Red
                occupied_count += 1
                status = "Busy"
            else:
                color = (0, 255, 0) # Green
                status = "Free"

            # Draw the visual feedback
            cv2.polylines(img_display, [points], True, color, 2)
            
            # Optional: Show the pixel count on screen (helps with tuning)
            # Find center to put text
            moment = cv2.moments(points)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                cv2.putText(img_display, str(count), (cx - 10, cy + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Update Dashboard Text
        cv2.putText(img_display, f'Occupied: {occupied_count} / {len(self.parking_data)}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    def run(self):
        while True:
            # Loop video
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            success, frame = self.cap.read()
            if not success: break
            
            # Resize
            frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
            
            # --- IMAGE PRE-PROCESSING ---
            # 1. Convert to Grayscale
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Gaussian Blur (Reduces noise/texture of asphalt)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
            
            # 3. Adaptive Threshold (The Magic Step)
            # Converts image to pure Black & White based on local lighting
            # If a pixel is brighter/darker than neighbors, it becomes white.
            # This helps ignoring shadows (which are smooth gradients).
            
            # Ensure block_size is odd and > 1
            block_size = cv2.getTrackbarPos("Block Size", "Control Panel")
            if block_size % 2 == 0: block_size += 1
            if block_size < 3: block_size = 3
                
            img_thresh = cv2.adaptiveThreshold(
                img_blur, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                block_size, 
                16
            )

            # Pass the processed (B&W) image to the checker
            self.check_parking_space(img_thresh, frame)
            
            cv2.imshow("Smart Parking Dashboard", frame)
            # Optional: View the threshold view to debug
            # cv2.imshow("Threshold View", img_thresh) 
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ParkingDetector()
    detector.run()