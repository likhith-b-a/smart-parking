"""
Manual parking-slot plotter: click the 4 corners of each real slot
directly on a video frame. Second stage after calibrate_lot.py — no
detection model, no pitch estimation, just what you click is what you
get. calibrate_lot.py's lanes (lot_boundary.json) are drawn underneath
as a visual reference only, so you can see the lane you're plotting
inside of; they play no part in the math.

Saves/loads slots.json directly — same schema app.py already reads
(a flat list of 4-point [x,y] polygons), so no changes needed there.
Existing slots.json is loaded on start so you can resume or fix a few
slots without re-plotting the whole lot.

Usage:
    python plot_slots.py [video_path]

Controls:
    left click   add a corner to the slot in progress (4 max)
    n            confirm current slot (needs exactly 4 points), start next
    r            reset current slot's in-progress clicks
    z            undo last confirmed slot
    x            toggle delete mode; click inside any slot to remove it
    s            save all confirmed slots to slots.json
    q            quit without saving
"""
import sys
import json
import cv2
import numpy as np

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else 'video1.mp4'
FIXED_WIDTH, FIXED_HEIGHT = 1280, 720
BOUNDARY_FILE = 'lot_boundary.json'
OUT_FILE = 'slots.json'
WINDOW = 'Plot Slots - click 4 corners per slot (n=next, x=delete mode, s=save, q=quit)'

current_points = []
slots = []
lanes = []
delete_mode = False
frame = None


def redraw():
    disp = frame.copy()
    for lane in lanes:
        cv2.polylines(disp, [np.array(lane, dtype=np.int32)], True, (120, 120, 120), 1)
    for i, slot in enumerate(slots):
        pts = np.array(slot, dtype=np.int32)
        color = (0, 100, 255) if delete_mode else (0, 200, 0)
        cv2.polylines(disp, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(disp, str(i + 1), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for i, (x, y) in enumerate(current_points):
        cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(disp, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(current_points) > 1:
        cv2.polylines(disp, [np.array(current_points, dtype=np.int32)],
                      len(current_points) == 4, (0, 255, 255), 2)
    mode_txt = 'DELETE MODE (click a slot to remove it)' if delete_mode else f'Slots: {len(slots)}'
    cv2.putText(disp, mode_txt, (10, FIXED_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255) if delete_mode else (255, 255, 255), 1)
    cv2.imshow(WINDOW, disp)


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if delete_mode:
        for i, slot in enumerate(slots):
            pts = np.array(slot, dtype=np.float32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                slots.pop(i)
                print(f'Deleted slot, {len(slots)} remain')
                break
        redraw()
    elif len(current_points) < 4:
        current_points.append((x, y))
        redraw()


def main():
    global frame, delete_mode
    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f'Could not read a frame from {VIDEO_PATH}')
        sys.exit(1)
    frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

    try:
        with open(OUT_FILE) as f:
            loaded = json.load(f)
        slots.extend(loaded)
        print(f'Loaded {len(slots)} existing slots from {OUT_FILE}')
    except FileNotFoundError:
        pass

    try:
        with open(BOUNDARY_FILE) as f:
            lanes.extend(json.load(f)['lanes'])
        print(f'Loaded {len(lanes)} lane(s) from {BOUNDARY_FILE} as reference guide')
    except FileNotFoundError:
        print(f'No {BOUNDARY_FILE} found — run calibrate_lot.py first for a lane guide, '
              'or plot slots freehand without one')

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    print('Click 4 corners of a slot. n=confirm, r=reset current, z=undo last, '
          'x=toggle delete mode, s=save, q=quit')

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            current_points.clear()
            redraw()
        elif key == ord('n'):
            if len(current_points) != 4:
                print(f'Need exactly 4 points for this slot, have {len(current_points)}')
                continue
            slots.append(list(current_points))
            current_points.clear()
            print(f'Slot {len(slots)} confirmed')
            redraw()
        elif key == ord('z'):
            if slots:
                slots.pop()
                print(f'Removed last slot, {len(slots)} remain')
                redraw()
        elif key == ord('x'):
            delete_mode = not delete_mode
            current_points.clear()
            redraw()
        elif key == ord('s'):
            if current_points and len(current_points) == 4:
                slots.append(list(current_points))
                current_points.clear()
                print(f'Slot {len(slots)} confirmed')
            if not slots:
                print('No slots to save')
                continue
            with open(OUT_FILE, 'w') as f:
                json.dump(slots, f, indent=2)
            print(f'Saved {len(slots)} slot(s) to {OUT_FILE}')
            break
        elif key == ord('q'):
            print('Quit without saving')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
