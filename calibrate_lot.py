"""
One-time tool: click the 4 corners of each parking lane on a sample
frame. Saves lot_boundary.json (a list of quadrilaterals, one per lane
— no per-slot polygons needed). Handles lots with driving aisles
between lanes (each lane calibrated separately so slots don't get
placed in the gaps).

Usage:
    python calibrate_lot.py [video_path]

For each lane: click its 4 corners, tracing the perimeter (any
starting corner, either direction — generate_slots.py figures out
orientation on its own). Press 'n' to confirm the lane and start the
next one, 'r' to reset the current lane's clicks, 'z' to undo the last
confirmed lane, 's' to save all confirmed lanes, 'q' to quit without
saving.
"""
import sys
import json
import cv2
import numpy as np

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else 'video1.mp4'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720
OUT_FILE = 'lot_boundary.json'
WINDOW = 'Calibrate Lot - click 4 corners per lane (n=next lane, s=save, q=quit)'

current_points = []
lanes = []
frame = None


def redraw():
    disp = frame.copy()
    for lane in lanes:
        cv2.polylines(disp, [np.array(lane, dtype=np.int32)], True, (0, 200, 0), 2)
    for i, (x, y) in enumerate(current_points):
        cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(disp, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(current_points) > 1:
        cv2.polylines(disp, [np.array(current_points, dtype=np.int32)],
                      len(current_points) == 4, (0, 255, 255), 2)
    cv2.putText(disp, f'Lanes confirmed: {len(lanes)}', (10, FIXED_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(WINDOW, disp)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
        current_points.append((x, y))
        redraw()


def main():
    global frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f'Could not read a frame from {VIDEO_PATH}')
        sys.exit(1)

    frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    print('Click 4 corners of a lane. n=confirm lane, r=reset current, z=undo last lane, s=save, q=quit')

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            current_points.clear()
            redraw()
        elif key == ord('n'):
            if len(current_points) != 4:
                print(f'Need exactly 4 points for this lane, have {len(current_points)}')
                continue
            lanes.append(list(current_points))
            current_points.clear()
            print(f'Lane {len(lanes)} confirmed')
            redraw()
        elif key == ord('z'):
            if lanes:
                lanes.pop()
                print(f'Removed last lane, {len(lanes)} remain')
                redraw()
        elif key == ord('s'):
            if current_points and len(current_points) == 4:
                lanes.append(list(current_points))
                current_points.clear()
                print(f'Lane {len(lanes)} confirmed')
            if not lanes:
                print('No lanes confirmed yet')
                continue
            with open(OUT_FILE, 'w') as f:
                json.dump({'lanes': lanes, 'width': FIXED_WIDTH, 'height': FIXED_HEIGHT}, f, indent=2)
            print(f'Saved {len(lanes)} lane(s) to {OUT_FILE}')
            break
        elif key == ord('q'):
            print('Quit without saving')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
