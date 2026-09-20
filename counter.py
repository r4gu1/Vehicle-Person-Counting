"""
Vehicle & Person Counting with Zone Intrusion Alerts
====================================================
Python + OpenCV + YOLOv8 (Ultralytics) + ByteTrack

Features
--------
1. Detect & track people and vehicles (car, motorcycle, bus, truck, bicycle).
2. Line-crossing counting (directional: IN / OUT) for any number of lines.
3. Polygon zone intrusion alerts (dwell-time based, with cooldown).
4. All events logged to CSV (events.csv) + per-run summary (summary.csv).
5. FPS measurement and accuracy evaluation against manually labeled ground truth.

Usage
-----
    pip install ultralytics opencv-python numpy shapely

    # interactively draw lines/zones on the first frame, save to config
    python counter.py --source clip.mp4 --setup --config zones.json

    # run the pipeline
    python counter.py --source clip.mp4 --config zones.json --save out.mp4

    # webcam / RTSP
    python counter.py --source 0 --config zones.json
    python counter.py --source rtsp://user:pass@ip:554/stream --config zones.json

    # evaluate counts against a manually labeled file
    python counter.py --source clip.mp4 --config zones.json --gt ground_truth.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# COCO class ids we care about -> logical group
CLASS_GROUPS: Dict[int, str] = {
    0: "person",
    1: "vehicle",   # bicycle
    2: "vehicle",   # car
    3: "vehicle",   # motorcycle
    5: "vehicle",   # bus
    7: "vehicle",   # truck
}
CLASS_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}
TARGET_CLASSES: List[int] = sorted(CLASS_GROUPS.keys())

COLORS = {
    "person": (0, 200, 255),
    "vehicle": (80, 220, 120),
    "line": (255, 180, 60),
    "zone": (60, 120, 255),
    "alert": (40, 40, 240),
    "text_bg": (25, 25, 25),
}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def side_of_line(p: Tuple[float, float],
                 a: Tuple[float, float],
                 b: Tuple[float, float]) -> float:
    """Signed area (cross product). >0 left of A->B, <0 right, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1p2 intersects segment p3p4."""
    d1 = side_of_line(p1, p3, p4)
    d2 = side_of_line(p2, p3, p4)
    d3 = side_of_line(p3, p1, p2)
    d4 = side_of_line(p4, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def point_in_polygon(p: Tuple[float, float], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0


# --------------------------------------------------------------------------- #
# Counting primitives
# --------------------------------------------------------------------------- #

@dataclass
class CountingLine:
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    # counts[direction][group]
    counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"in": defaultdict(int), "out": defaultdict(int)}
    )
    crossed_ids: set = field(default_factory=set)

    def check(self, track_id: int, prev_pt, curr_pt) -> Optional[str]:
        """Return 'in'/'out' if the track crossed this line on this frame."""
        if track_id in self.crossed_ids:
            return None
        if not segments_intersect(prev_pt, curr_pt, self.p1, self.p2):
            return None
        self.crossed_ids.add(track_id)
        # direction: sign flip of the signed area tells which way it went
        before = side_of_line(prev_pt, self.p1, self.p2)
        return "in" if before < 0 else "out"

    def total(self) -> int:
        return sum(self.counts["in"].values()) + sum(self.counts["out"].values())


@dataclass
class Zone:
    name: str
    points: List[Tuple[int, int]]
    classes: Tuple[str, ...] = ("person", "vehicle")
    min_dwell_s: float = 1.0        # must stay this long before alerting
    cooldown_s: float = 10.0        # don't re-alert the same track before this
    polygon: np.ndarray = field(init=False)
    _entered_at: Dict[int, float] = field(default_factory=dict)
    _last_alert: Dict[int, float] = field(default_factory=dict)
    active_ids: set = field(default_factory=set)
    alert_count: int = 0

    def __post_init__(self):
        self.polygon = np.array(self.points, dtype=np.int32)

    def update(self, track_id: int, group: str, point, now: float) -> bool:
        """Feed one track. Returns True when a new intrusion alert fires."""
        inside = group in self.classes and point_in_polygon(point, self.polygon)
        if not inside:
            self._entered_at.pop(track_id, None)
            self.active_ids.discard(track_id)
            return False

        self.active_ids.add(track_id)
        t0 = self._entered_at.setdefault(track_id, now)
        if now - t0 < self.min_dwell_s:
            return False
        if now - self._last_alert.get(track_id, -1e9) < self.cooldown_s:
            return False
        self._last_alert[track_id] = now
        self.alert_count += 1
        return True

    def forget(self, track_id: int) -> None:
        self._entered_at.pop(track_id, None)
        self._last_alert.pop(track_id, None)
        self.active_ids.discard(track_id)


# --------------------------------------------------------------------------- #
# CSV event logging
# --------------------------------------------------------------------------- #

class EventLogger:
    FIELDS = ["timestamp", "frame", "video_time_s", "event_type",
              "object_group", "class_name", "track_id", "target",
              "direction", "x", "y", "confidence"]

    def __init__(self, path: str):
        self.path = path
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new:
            self._w.writeheader()
        self.rows: List[dict] = []

    def log(self, **kw) -> None:
        row = {k: kw.get(k, "") for k in self.FIELDS}
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self._w.writerow(row)
        self._fh.flush()
        self.rows.append(row)

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Interactive setup: draw lines and zones on the first frame
# --------------------------------------------------------------------------- #

def interactive_setup(frame: np.ndarray, config_path: str) -> dict:
    """
    Left-click  : add point
    'l'         : finish current points as a LINE (needs exactly 2 points)
    'z'         : finish current points as a ZONE (needs >= 3 points)
    'u'         : undo last point
    's'         : save & quit      'q' : quit without saving
    """
    pts: List[Tuple[int, int]] = []
    lines: List[dict] = []
    zones: List[dict] = []
    win = "setup - l=line  z=zone  u=undo  s=save  q=quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for ln in lines:
            cv2.line(canvas, tuple(ln["p1"]), tuple(ln["p2"]), COLORS["line"], 2)
            cv2.putText(canvas, ln["name"], tuple(ln["p1"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["line"], 2)
        for z in zones:
            poly = np.array(z["points"], np.int32)
            cv2.polylines(canvas, [poly], True, COLORS["zone"], 2)
            cv2.putText(canvas, z["name"], tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["zone"], 2)
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 4, (255, 255, 255), -1)
            if i:
                cv2.line(canvas, pts[i - 1], p, (255, 255, 255), 1)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("l") and len(pts) == 2:
            lines.append({"name": f"line_{len(lines)+1}", "p1": pts[0], "p2": pts[1]})
            pts = []
        elif k == ord("z") and len(pts) >= 3:
            zones.append({"name": f"zone_{len(zones)+1}", "points": list(pts),
                          "classes": ["person", "vehicle"],
                          "min_dwell_s": 1.0, "cooldown_s": 10.0})
            pts = []
        elif k == ord("s"):
            break
        elif k in (ord("q"), 27):
            cv2.destroyWindow(win)
            raise SystemExit("setup aborted")

    cv2.destroyWindow(win)
    cfg = {"lines": lines, "zones": zones}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[setup] saved {len(lines)} line(s), {len(zones)} zone(s) -> {config_path}")
    return cfg


def load_config(path: str, frame_shape) -> Tuple[List[CountingLine], List[Zone]]:
    if path and os.path.exists(path):
        cfg = json.load(open(path, encoding="utf-8"))
    else:
        # sensible default: horizontal mid-line + centre rectangle zone
        h, w = frame_shape[:2]
        cfg = {
            "lines": [{"name": "main_line", "p1": [0, h // 2], "p2": [w, h // 2]}],
            "zones": [{"name": "restricted_area",
                       "points": [[int(w * .35), int(h * .25)], [int(w * .65), int(h * .25)],
                                  [int(w * .65), int(h * .75)], [int(w * .35), int(h * .75)]],
                       "classes": ["person", "vehicle"],
                       "min_dwell_s": 1.0, "cooldown_s": 10.0}],
        }
        print("[config] no config found - using defaults")

    lines = [CountingLine(l["name"], tuple(l["p1"]), tuple(l["p2"])) for l in cfg.get("lines", [])]
    zones = [Zone(z["name"], [tuple(p) for p in z["points"]],
                  tuple(z.get("classes", ["person", "vehicle"])),
                  float(z.get("min_dwell_s", 1.0)),
                  float(z.get("cooldown_s", 10.0)))
             for z in cfg.get("zones", [])]
    return lines, zones


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_zone(frame, zone: Zone, alerting: bool) -> None:
    color = COLORS["alert"] if alerting else COLORS["zone"]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone.polygon], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [zone.polygon], True, color, 2)
    x, y = zone.polygon[0]
    cv2.putText(frame, f"{zone.name} [{len(zone.active_ids)}]", (int(x), int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_panel(frame, lines: List[CountingLine], zones: List[Zone],
               fps: float, totals: Dict[str, int]) -> None:
    rows = [f"FPS: {fps:5.1f}",
            f"person IN/OUT: {totals['person_in']}/{totals['person_out']}",
            f"vehicle IN/OUT: {totals['vehicle_in']}/{totals['vehicle_out']}"]
    rows += [f"{z.name}: {z.alert_count} alerts" for z in zones]
    w = 300
    h = 24 * len(rows) + 16
    cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), COLORS["text_bg"], -1)
    for i, t in enumerate(rows):
        cv2.putText(frame, t, (22, 36 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run(args) -> dict:
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("empty stream")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.setup:
        interactive_setup(first, args.config)

    lines, zones = load_config(args.config, first.shape)
    logger = EventLogger(args.events)

    model = YOLO(args.model)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps,
                                 (first.shape[1], first.shape[0]))

    history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    last_seen: Dict[int, int] = {}
    totals = {"person_in": 0, "person_out": 0, "vehicle_in": 0, "vehicle_out": 0}
    unique_seen: Dict[str, set] = {"person": set(), "vehicle": set()}

    frame_idx = 0
    fps_window = deque(maxlen=30)
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t0 = time.perf_counter()
        vtime = frame_idx / src_fps

        results = model.track(
            frame,
            persist=True,
            classes=TARGET_CLASSES,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=args.tracker,
            device=args.device,
            verbose=False,
        )[0]

        alerting_zones: set = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cid, cf in zip(boxes, ids, clss, confs):
                group = CLASS_GROUPS.get(int(cid))
                if group is None:
                    continue
                cname = CLASS_NAMES.get(int(cid), str(cid))
                # anchor = bottom-centre (foot / wheel contact point)
                anchor = (float((x1 + x2) / 2), float(y2))
                history[tid].append(anchor)
                last_seen[tid] = frame_idx
                unique_seen[group].add(tid)

                # ---- line crossing -------------------------------------- #
                if len(history[tid]) >= 2:
                    prev = history[tid][-2]
                    for ln in lines:
                        direction = ln.check(tid, prev, anchor)
                        if direction:
                            ln.counts[direction][group] += 1
                            totals[f"{group}_{direction}"] += 1
                            logger.log(frame=frame_idx, video_time_s=round(vtime, 2),
                                       event_type="line_cross", object_group=group,
                                       class_name=cname, track_id=int(tid),
                                       target=ln.name, direction=direction,
                                       x=int(anchor[0]), y=int(anchor[1]),
                                       confidence=round(float(cf), 3))

                # ---- zone intrusion ------------------------------------- #
                for z in zones:
                    if z.update(int(tid), group, anchor, vtime):
                        alerting_zones.add(z.name)
                        logger.log(frame=frame_idx, video_time_s=round(vtime, 2),
                                   event_type="zone_intrusion", object_group=group,
                                   class_name=cname, track_id=int(tid),
                                   target=z.name, direction="",
                                   x=int(anchor[0]), y=int(anchor[1]),
                                   confidence=round(float(cf), 3))
                        print(f"[ALERT] {cname}#{tid} intruded {z.name} @ {vtime:.1f}s")

                # ---- draw ----------------------------------------------- #
                if not args.no_view or writer:
                    c = COLORS[group]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                    cv2.putText(frame, f"{cname} #{tid} {cf:.2f}",
                                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, c, 2, cv2.LINE_AA)
                    trail = list(history[tid])
                    for a, b in zip(trail, trail[1:]):
                        cv2.line(frame, (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])), c, 2)

        # forget tracks that disappeared (frees memory on long streams)
        stale = [t for t, f in last_seen.items() if frame_idx - f > int(src_fps * 5)]
        for t in stale:
            history.pop(t, None)
            last_seen.pop(t, None)
            for z in zones:
                z.forget(t)

        fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        fps = float(np.mean(fps_window))

        if not args.no_view or writer:
            for ln in lines:
                cv2.line(frame, ln.p1, ln.p2, COLORS["line"], 2)
                cv2.putText(frame, f"{ln.name} {ln.total()}",
                            (ln.p1[0] + 5, ln.p1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["line"], 2)
            for z in zones:
                draw_zone(frame, z, z.name in alerting_zones or bool(z.active_ids))
            draw_panel(frame, lines, zones, fps, totals)

        if writer:
            writer.write(frame)
        if not args.no_view:
            cv2.imshow("counting", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    elapsed = time.time() - t_start
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    summary = {
        "frames": frame_idx,
        "elapsed_s": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 2) if elapsed else 0.0,
        "unique_persons": len(unique_seen["person"]),
        "unique_vehicles": len(unique_seen["vehicle"]),
        **totals,
        "zone_alerts": {z.name: z.alert_count for z in zones},
        "line_counts": {ln.name: {d: dict(v) for d, v in ln.counts.items()} for ln in lines},
    }
    logger.close()
    write_summary(args.summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def write_summary(path: str, summary: dict) -> None:
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary.items()}
    flat["run_at"] = datetime.now().isoformat(timespec="seconds")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if new:
            w.writeheader()
        w.writerow(flat)


# --------------------------------------------------------------------------- #
# Evaluation against manually labeled clips
# --------------------------------------------------------------------------- #

def evaluate(summary: dict, gt_path: str) -> None:
    """
    ground_truth.json example:
    {
      "person_in": 12, "person_out": 9,
      "vehicle_in": 31, "vehicle_out": 28,
      "zone_alerts": {"restricted_area": 4}
    }
    """
    gt = json.load(open(gt_path, encoding="utf-8"))
    print("\n=== Count accuracy vs manual labels ===")
    print(f"{'metric':<22}{'pred':>6}{'gt':>6}{'err':>6}{'acc %':>8}")
    accs = []
    for key in ["person_in", "person_out", "vehicle_in", "vehicle_out"]:
        if key not in gt:
            continue
        pred, truth = summary.get(key, 0), gt[key]
        err = pred - truth
        acc = 100.0 * (1 - abs(err) / truth) if truth else (100.0 if pred == 0 else 0.0)
        accs.append(max(acc, 0.0))
        print(f"{key:<22}{pred:>6}{truth:>6}{err:>+6}{max(acc,0):>8.1f}")
    for zname, truth in gt.get("zone_alerts", {}).items():
        pred = summary["zone_alerts"].get(zname, 0)
        err = pred - truth
        acc = 100.0 * (1 - abs(err) / truth) if truth else (100.0 if pred == 0 else 0.0)
        accs.append(max(acc, 0.0))
        print(f"{('zone:' + zname):<22}{pred:>6}{truth:>6}{err:>+6}{max(acc,0):>8.1f}")
    if accs:
        print(f"\nMean count accuracy : {np.mean(accs):.1f}%")
    print(f"Average FPS         : {summary['avg_fps']}")


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Vehicle & person counting with zone intrusion alerts")
    p.add_argument("--source", default="0", help="video path, webcam index, or RTSP url")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt or custom weights")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--config", default="zones.json", help="lines/zones JSON")
    p.add_argument("--setup", action="store_true", help="draw lines/zones on frame 1 and save")
    p.add_argument("--events", default="events.csv")
    p.add_argument("--summary", default="summary.csv")
    p.add_argument("--save", default=None, help="write annotated mp4 to this path")
    p.add_argument("--gt", default=None, help="ground truth JSON for accuracy evaluation")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="'cpu', '0', 'mps' ...")
    p.add_argument("--no-view", action="store_true", help="headless (no window)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    s = run(a)
    if a.gt:
        evaluate(s, a.gt)
