#!/usr/bin/env python3
"""Review and adjust the curb lines that drive parking occupancy.

You usually do not need this. The collector fits the curb from where cars
actually park and writes `config/curb.yaml` itself. Reach for this tool when
the automatic fit is wrong — too few cars when it fitted, a delivery van parked
askew, or a curb that bends where a straight line will not do.

    python tools/curb_editor.py                # live frame from the camera
    python tools/curb_editor.py --latest       # newest collected timelapse frame
    python tools/curb_editor.py --image FILE

Draw each line along the gutter, where tyres meet the road — not along the car
bodies. A line may have any number of points, so a curve is fine.

Mouse       left click  add a point      right click  undo last point
Keys        a  auto-fit lines from the vehicles in this frame
            n  finish the current line      u  undo last point
            d  delete last line             r  reset everything
            b  toggle detection boxes       p  preview occupancy and gaps
            s  save to config/curb.yaml      q  quit
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.config import (curb_calib_path, curb_path, load_config,
                              load_curb_doc, save_curb_doc)
from streetmon.curb import CurbMonitor, autofit_curbs

WINDOW = "curb editor — click along the gutter, 'n' to finish, 's' to save"
PALETTE = [(0, 215, 255), (255, 170, 0), (140, 255, 140), (255, 120, 220)]
state = {"pts": [], "curbs": [], "dirty": False}


def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["pts"].append([float(x), float(y)])
        state["dirty"] = True
    elif event == cv2.EVENT_RBUTTONDOWN and state["pts"]:
        state["pts"].pop()
        state["dirty"] = True


def grab_frame(args, cfg):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        print(f"using {args.image}")
        return img
    if args.latest:
        files = sorted(glob.glob("data/*/session_*/timelapse/*.jpg"))
        if not files:
            raise SystemExit("no timelapse frames yet; run collect.py first")
        print(f"using {files[-1]}")
        return cv2.imread(files[-1])
    from streetmon.camera import CameraReader
    cam = CameraReader(cfg.camera)
    try:
        cam.start()
        if not cam.wait_first_frame(10.0):
            raise SystemExit("no frames from camera")
        _, frame = cam.latest()
        print(f"grabbed a live frame at {frame.shape[1]}x{frame.shape[0]}")
        return frame
    finally:
        cam.stop()


def render(base, dets, show_boxes, overlay_states, monitor):
    img = base.copy()
    if show_boxes and dets:
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (110, 110, 110), 1)
            cv2.circle(img, ((x1 + x2) // 2, y2), 5, (0, 255, 255), -1)

    if overlay_states and monitor is not None:
        monitor.draw(img, overlay_states)
    else:
        for i, c in enumerate(state["curbs"]):
            col = PALETTE[i % len(PALETTE)]
            pts = np.array(c["points"], dtype=np.int32)
            cv2.polylines(img, [pts], False, col, 3)
            for p in pts:
                cv2.circle(img, tuple(p), 5, col, -1)
            cv2.putText(img, c["name"], tuple(pts[0] + np.array([8, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    pts = state["pts"]
    if pts:
        arr = np.array(pts, dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [arr], False, (255, 255, 255), 2)
        for p in arr:
            cv2.circle(img, tuple(p), 5, (255, 255, 255), -1)

    lines = [f"{len(state['curbs'])} curb line(s)   "
             f"{len(pts)} point(s) in progress"
             + ("   [occupancy preview]" if overlay_states else ""),
             "a auto-fit  n finish  u undo  d del  r reset  b boxes  "
             "p preview  s save  q quit"]
    for i, t in enumerate(lines):
        y = 26 + i * 26
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (6, y - th - 7), (14 + tw, y + 7), (0, 0, 0), -1)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--no-load", action="store_true",
                    help="start empty instead of loading config/curb.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    base = grab_frame(args, cfg)
    h, w = base.shape[:2]

    if not args.no_load:
        doc = load_curb_doc()
        existing = doc.get("curbs") or []
        src = doc.get("frame_size")
        if existing and src and tuple(src) != (w, h):
            sx, sy = w / float(src[0]), h / float(src[1])
            print(f"rescaling from {src[0]}x{src[1]} to {w}x{h}")
            for c in existing:
                c["points"] = [[p[0] * sx, p[1] * sy] for p in c["points"]]
        state["curbs"] = existing
        if existing:
            print(f"loaded {len(existing)} curb line(s) from {curb_path()}")

    detector = dets = None
    show_boxes = True
    overlay_states = None
    monitor = None

    def ensure_detector():
        nonlocal detector, dets
        if detector is None:
            print("loading detector...")
            from streetmon.detector import Detector
            detector = Detector(cfg.detector)
            dets = detector.track(base)
            detector.reset_tracker()
            print(f"{len(dets)} objects detected")
        return dets

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, w), int(min(1600, w) * h / w))
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)
    ensure_detector()

    while True:
        cv2.imshow(WINDOW, render(base, dets, show_boxes, overlay_states, monitor))
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            if state["dirty"]:
                print("unsaved changes — 's' to save, or 'q' again to discard")
                state["dirty"] = False
                continue
            break
        elif key == ord("a"):
            found = autofit_curbs(ensure_detector(), (w, h),
                                  set(cfg.detector.vehicle_classes))
            if not found:
                print("not enough parked vehicles in this frame to fit a line")
            else:
                state["curbs"] = found
                state["dirty"] = True
                for f in found:
                    print(f"  fitted '{f['name']}' from {f['fitted_from']} "
                          f"vehicle(s), max residual {f['max_residual_px']} px")
        elif key == ord("n"):
            if len(state["pts"]) >= 2:
                used = {c["name"] for c in state["curbs"]}
                name = next((n for n in ("far", "near") if n not in used),
                            f"curb_{len(state['curbs']) + 1}")
                state["curbs"].append({"name": name,
                                       "points": [list(p) for p in state["pts"]]})
                print(f"added '{name}' with {len(state['pts'])} point(s)")
                state["pts"] = []
                state["dirty"] = True
            else:
                print("a curb line needs at least 2 points")
        elif key == ord("u") and state["pts"]:
            state["pts"].pop()
        elif key == ord("d") and state["curbs"]:
            print(f"deleted '{state['curbs'].pop()['name']}'")
            state["dirty"] = True
            overlay_states = None
        elif key == ord("r"):
            state["curbs"], state["pts"] = [], []
            state["dirty"] = True
            overlay_states = None
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("p"):
            if not state["curbs"]:
                print("no curb lines to preview")
                continue
            monitor = CurbMonitor(state["curbs"], (w, h), cfg.curb,
                                  vehicle_classes=cfg.detector.vehicle_classes,
                                  calib_path=curb_calib_path())
            veh = ensure_detector()
            for d in veh:
                monitor.observe_for_calibration(d)
            overlay_states = monitor.update(
                veh, parked_ids={d.track_id for d in veh})
            for st in overlay_states:
                print(f"  [{st.name}] {st.length_m:.1f} m visible, "
                      f"{st.occupied_m:.1f} m occupied, {st.free_m:.1f} m free, "
                      f"{len(st.parkable(float(cfg.curb.min_gap_m)))} parkable gap(s)"
                      + ("" if st.calibrated else "   (uncalibrated — px, not m)"))
        elif key == ord("s"):
            if state["pts"]:
                print("finish or undo the line in progress first ('n' or 'u')")
                continue
            path = save_curb_doc({"frame_size": [w, h], "curbs": state["curbs"]})
            state["dirty"] = False
            print(f"saved {len(state['curbs'])} curb line(s) to {path}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
