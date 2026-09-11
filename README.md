# Smart Parking System

Flask dashboard that detects vehicles in a CCTV/parking-lot video feed using YOLOv8-OBB and shows live occupancy on a web UI.

## Setup

Windows:
```bat
install.bat
```

macOS/Linux:
```bash
./install.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

First run auto-downloads `yolov8s-obb.pt` (~23MB).

Place your video as `video1.mp4` in the project root (or edit `VIDEO_PATH` in `app.py`).

## Slot grid: `slots.json`

Without it, the app just counts raw detections against a hardcoded
`TOTAL_SLOTS = 30`. `slots.json` is a plain list of 4-point polygons,
one per real parking slot — `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` per
slot, in the app's 1280x720 frame. `app.py` picks it up automatically
if present, marking a slot occupied when a detected vehicle's center
falls inside its polygon.

Generate it automatically — only the lane boundaries are clicked
manually (once per lot), everything inside a lane is auto-tiled:

```bash
python calibrate_lot.py          # click 4 corners of each parking lane
python generate_slots.py         # measures real slot spacing per lane, tiles rotated slot quads -> slots.json
```

`calibrate_lot.py` opens the first video frame — click each lane's 4
corners (top-left, top-right, bottom-right, bottom-left), press `n` to
confirm and start the next lane, `s` to save all lanes once done.
Handles lots with multiple lanes separated by driving aisles (each
lane calibrated separately so slots don't land in the gaps).

`generate_slots.py` then samples the video and, for each lane, measures
the real gap between neighboring parked cars projected onto that
lane's own direction — not a raw car-size measurement, which doesn't
work here: this lot uses angled parking, so a car's own length is
foreshortened relative to the lane and doesn't match the real slot
pitch (car length ~70px vs true pitch ~30px on this video). It tiles
each lane into that many rotated slot quads, matching the lane's real
angle (bilinear-interpolated from its 4 corners) — no per-slot
clicking. `legacy/parking_slots.json` (hand-plotted, this repo's
original) is kept for reference/comparison; `slots.json` is the
generated one actually used by `app.py`.

## Run

```bash
python app.py
```

Open `http://localhost:5000`.

## How it works

- `app.py` reads frames from the video, runs YOLOv8s-**OBB** (`imgsz=1280`, `conf=0.1`) every 10th frame (`SKIP_FRAMES`), detecting DOTA classes `small vehicle`/`large vehicle`.
- OBB (oriented bounding box) draws rotated boxes that hug each car's actual angle — a plain axis-aligned box has to stretch to cover a rotated car's full diagonal extent, so it never fits tightly on an aerial/angled feed like this one. `conf` is lower than a COCO model would need since the DOTA-trained OBB model is less confident on this specific domain — still zero observed false positives at 0.1 on this lot.
- If `slots.json` exists, each slot polygon is marked occupied when a detected vehicle's center falls inside it (`cv2.pointPolygonTest`) — real per-slot state, not a raw count. Slot polygons are drawn green/red on the video, following each slot's actual rotation. Without it, falls back to raw vehicle count against `TOTAL_SLOTS`.
- `/api/status` returns:
  ```json
  {
    "occupied": 12,
    "total": 56,
    "slots": [0, 1, 1, 0, ...],
    "vehicles": [
      {"class": "small vehicle", "confidence": 0.79, "bbox": [x1, y1, x2, y2], "obb": [[x,y], [x,y], [x,y], [x,y]]}
    ]
  }
  ```
  `bbox` is an axis-aligned wrapper (min/max of the rotated corners) for consumers that want a simple box; `obb` is the tight rotated quad.

## Legacy

`legacy/` holds the old threshold-based approach (`detection.py`, `auto_detect_slots.py`, `plot.py`, `show.py`, `parking_slots.json`) — manual polygon slots + adaptive-threshold pixel counting. Kept for reference only, not used by `app.py`.
