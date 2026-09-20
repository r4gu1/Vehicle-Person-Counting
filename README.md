# Vehicle & Person Counting with Zone Intrusion Alerts

Python · OpenCV · YOLOv8 (Ultralytics) · ByteTrack

## Install

```bash
pip install ultralytics opencv-python numpy
```

YOLOv8 weights download automatically on first run (`yolov8n.pt`).

## Quick start

```bash
# 1. Draw counting lines and restricted zones on the first frame
python counter.py --source clip.mp4 --setup --config zones.json
#    left-click = add point | l = save 2 points as LINE | z = save 3+ points as ZONE
#    u = undo | s = save & continue | q = abort

# 2. Run
python counter.py --source clip.mp4 --config zones.json --save out.mp4

# 3. Evaluate against manual labels
python counter.py --source clip.mp4 --config zones.json --gt ground_truth.json --no-view
```

Live sources: `--source 0` (webcam) or `--source rtsp://...`.

## Config (`zones.json`)

```json
{
  "lines": [
    { "name": "gate_line", "p1": [50, 400], "p2": [1230, 400] }
  ],
  "zones": [
    {
      "name": "restricted_area",
      "points": [[400,200],[900,200],[900,600],[400,600]],
      "classes": ["person"],
      "min_dwell_s": 1.5,
      "cooldown_s": 10
    }
  ]
}
```

- `classes` — which groups trigger the zone (`person`, `vehicle`).
- `min_dwell_s` — object must stay inside this long before an alert (kills flicker).
- `cooldown_s` — same track won't re-alert within this window.

## Ground truth (`ground_truth.json`)

```json
{
  "person_in": 12, "person_out": 9,
  "vehicle_in": 31, "vehicle_out": 28,
  "zone_alerts": { "restricted_area": 4 }
}
```

## Outputs

| File | Content |
|---|---|
| `events.csv` | one row per line-crossing / zone-intrusion event |
| `summary.csv` | one row per run: frames, elapsed, avg FPS, totals, alerts |
| `out.mp4` | annotated video (boxes, IDs, trails, lines, zones, HUD) |

`events.csv` columns: `timestamp, frame, video_time_s, event_type, object_group,
class_name, track_id, target, direction, x, y, confidence`.

## How it works

1. **Detect** — YOLOv8 restricted to COCO classes person, bicycle, car, motorcycle, bus, truck.
2. **Track** — ByteTrack (`model.track(persist=True)`) gives stable IDs across frames.
3. **Count** — each track's bottom-centre anchor is kept in a short history; a crossing
   fires when the segment `prev→curr` intersects the line. Sign of the cross product
   before crossing gives direction (IN/OUT). Each ID counts once per line.
4. **Zones** — `cv2.pointPolygonTest` on the anchor, plus dwell-time and per-track
   cooldown so a loitering object doesn't spam alerts.
5. **Log** — every event flushed to CSV immediately.
6. **Evaluate** — `--gt` prints per-metric error and accuracy `1 - |pred-gt|/gt`,
   alongside measured average FPS.

## Tuning

| Symptom | Fix |
|---|---|
| Low FPS | `--imgsz 480`, `--model yolov8n.pt`, `--device 0`, `--no-view` |
| Missed small objects | `--imgsz 960`, `--model yolov8s.pt`, lower `--conf` |
| ID switches / double counts | `--tracker botsort.yaml`, raise `--conf` |
| Alert flicker | raise `min_dwell_s` |
