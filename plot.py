import cv2
import json
import os
import math

# --- CONFIGURATION ---
VIDEO_PATH = 'clip8_fixed.mp4'
OUTPUT_FILE = 'parking_slots.json'
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

class ParkingAnnotationTool:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame = None
        
        self.all_points = []      # List of all plotted coordinates [(x,y), ...]
        self.parking_lots = []    # List of lots (each lot is a list of 4 points)
        self.current_selection = [] # Temp storage for the 4 points you are currently selecting
        self.mode = "PLOTTING"    # Modes: "PLOTTING" or "CONNECTING"

        # Load existing data if available
        if os.path.exists(OUTPUT_FILE):
             # We only load the finished lots. Recovering raw points from lots is tricky 
             # without duplicates, so usually we start fresh or just append new lots.
             # For simplicity here: we load lots to view them, but we plot new points.
            with open(OUTPUT_FILE, 'r') as f:
                self.parking_lots = json.load(f)
            print(f"Loaded {len(self.parking_lots)} existing lots.")

        # Read and Resize Frame
        success, temp_frame = self.cap.read()
        if not success:
            raise ValueError("Could not read video.")
        self.frame = cv2.resize(temp_frame, (TARGET_WIDTH, TARGET_HEIGHT))

    def get_nearest_point(self, x, y):
        # Find the point closest to the mouse click (within 15 pixels)
        limit = 15
        nearest = None
        min_dist = float('inf')

        for pt in self.all_points:
            dist = math.hypot(pt[0] - x, pt[1] - y)
            if dist < limit and dist < min_dist:
                min_dist = dist
                nearest = pt
        return nearest

    def mouse_events(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # --- MODE 1: PLOTTING POINTS ---
            if self.mode == "PLOTTING":
                self.all_points.append([x, y])

            # --- MODE 2: CONNECTING DOTS ---
            elif self.mode == "CONNECTING":
                # Find which existing point was clicked
                pt = self.get_nearest_point(x, y)
                if pt:
                    if pt not in self.current_selection:
                        self.current_selection.append(pt)
                    
                        # If we have selected 4 points, save the lot automatically
                        if len(self.current_selection) == 4:
                            self.parking_lots.append(self.current_selection)
                            print(f"Lot defined! Total lots: {len(self.parking_lots)}")
                            self.current_selection = [] # Reset for next lot

        # Right Click: Undo
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "PLOTTING" and len(self.all_points) > 0:
                self.all_points.pop()
            elif self.mode == "CONNECTING" and len(self.current_selection) > 0:
                self.current_selection.pop()

    def run(self):
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_events)
        
        print(f"Started in PLOTTING mode. Click all corners first.")
        print(f"Press 'TAB' to switch to CONNECTING mode.")

        while True:
            img_copy = self.frame.copy()

            # DRAWING LOGIC
            
            # 1. Draw Finalized Lots (Green Lines)
            for lot in self.parking_lots:
                for i in range(4):
                    pt1 = tuple(lot[i])
                    pt2 = tuple(lot[(i+1)%4])
                    cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)
                    
            # 2. Draw All Plotted Points (Blue Circles)
            for pt in self.all_points:
                cv2.circle(img_copy, tuple(pt), 3, (255, 0, 0), -1) # Blue dots

            # 3. Draw Current Selection in Connecting Mode (Red Highlights)
            if self.mode == "CONNECTING":
                for i, pt in enumerate(self.current_selection):
                    cv2.circle(img_copy, tuple(pt), 5, (0, 0, 255), -1) # Red larger dot
                    if i > 0:
                        cv2.line(img_copy, tuple(self.current_selection[i-1]), tuple(pt), (0, 0, 255), 2)
            
            # 4. Show Mode Text
            cv2.putText(img_copy, f"MODE: {self.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Frame", img_copy)
            key = cv2.waitKey(1) & 0xFF

            # CONTROLS
            if key == 9: # TAB Key to switch modes
                if self.mode == "PLOTTING":
                    self.mode = "CONNECTING"
                    print("Switched to CONNECTING mode. Click 4 blue dots to make a lot.")
                else:
                    self.mode = "PLOTTING"
                    print("Switched back to PLOTTING mode.")
            
            if key == ord('z'): # Undo last created LOT
                if len(self.parking_lots) > 0:
                    removed = self.parking_lots.pop()
                    print("Removed last Lot.")

            if key == ord('s'): # Save
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(self.parking_lots, f, indent=4)
                print(f"Saved {len(self.parking_lots)} lots to {OUTPUT_FILE}")
                break

            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = ParkingAnnotationTool(VIDEO_PATH)
    print("--- CONTROLS ---")
    print("Left Click : Add Point (Plotting) / Select Point (Connecting)")
    print("TAB Key    : Switch between Plotting and Connecting")
    print("Right Click: Undo last point/selection")
    print("'z' Key    : Undo last completed Lot")
    print("'s' Key    : Save and Exit")
    print("'q' Key    : Quit")
    tool.run()