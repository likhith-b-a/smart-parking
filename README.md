# Smart Parking System

Flask dashboard that detects vehicles in a CCTV/parking-lot video feed using YOLOv8 and shows live occupancy on a web UI.

## Setup

```bash
pip install -r requirements.txt
```

First run auto-downloads `yolov8n.pt` (~6MB).

Place your video as `video1.mp4` in the project root (or edit `VIDEO_PATH` in `app.py`).

## Run

```bash
python app.py
```

Open `http://localhost:5000`.

## How it works

- `app.py` reads frames from the video, runs YOLOv8n on every 5th frame (`SKIP_FRAMES`), detecting COCO classes car/bus/truck.
- Bounding boxes drawn on the MJPEG stream served at `/video_feed`.
- `/api/status` returns:
  ```json
  {
    "occupied": 12,
    "total": 30,
    "vehicles": [
      {"class": "car", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
    ]
  }
  ```
- `total` capacity is set via `TOTAL_SLOTS` in `app.py`; the dashboard fills that many tiles red/green by count (no fixed per-slot identity).

## Legacy

`legacy/` holds the old threshold-based approach (`detection.py`, `auto_detect_slots.py`, `plot.py`, `show.py`, `parking_slots.json`) — manual polygon slots + adaptive-threshold pixel counting. Kept for reference only, not used by `app.py`.
