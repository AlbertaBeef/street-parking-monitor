#!/usr/bin/env python3
"""Live preview for aiming the camera and fixing focus/exposure.

Shooting through window glass is the reason this tool exists. Autofocus will
happily lock onto the pane, dirt on it, or your own reflection, and
auto-exposure swings hard when a bright vehicle enters the frame — both produce
data that looks fine in a spot check and is unusable in bulk.

Aim the camera, then walk focus and exposure to fixed values and press `p` to
print a YAML block to paste into config/site.yaml.

Keys
    f         toggle autofocus            [ ]   focus  -/+
    e         toggle auto-exposure        - =   exposure -/+
    z x       digital zoom -/+            d     toggle detection overlay
    g         toggle 3x3 framing grid     s     save a snapshot
    p         print YAML for site.yaml    q/ESC quit

The `sharp` readout is the variance of the Laplacian over the centre of the
frame: bigger is sharper. Maximise it while the street (not the glass) is in
focus.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.camera import CameraReader
from streetmon.config import curb_calib_path, load_config, load_curb_doc
from streetmon.detector import Detector, draw_detections
from streetmon.curb import CurbMonitor

WINDOW = "aim — street-parking-monitor"


def sharpness(frame: np.ndarray) -> float:
    """Laplacian variance over the central half of the frame."""
    h, w = frame.shape[:2]
    roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_grid(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    for i in (1, 2):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (255, 255, 255), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (255, 255, 255), 1)


def hud(frame, lines):
    for i, text in enumerate(lines):
        y = 28 + i * 26
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, y - th - 7), (16 + tw, y + 7), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--detect", action="store_true",
                    help="start with the detector overlay enabled")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.camera.device = args.device

    cam = CameraReader(cfg.camera).start()
    if not cam.wait_first_frame(10.0):
        print("no frames from camera", file=sys.stderr)
        cam.stop()
        return 2
    frame_size = cam.actual_size()

    detector = None
    show_detect = args.detect
    if show_detect:
        detector = Detector(cfg.detector)

    doc = load_curb_doc()
    cmon = None
    if doc.get("curbs"):
        src = doc.get("frame_size")
        cmon = CurbMonitor(doc["curbs"], frame_size, cfg.curb,
                           vehicle_classes=cfg.detector.vehicle_classes,
                           source_frame_size=tuple(src) if src else None,
                           calib_path=curb_calib_path())

    autofocus = bool(cfg.camera.autofocus)
    auto_exp = bool(cfg.camera.auto_exposure)
    focus = float(cfg.camera.focus) if cfg.camera.focus is not None else 0.0
    exposure = float(cfg.camera.exposure) if cfg.camera.exposure is not None else 156.0
    zoom = float(cfg.camera.zoom) if cfg.camera.zoom is not None else 100.0
    show_grid = False
    dets: list = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    print(__doc__)

    snap_dir = Path(__file__).resolve().parent.parent / "data" / "_aim"
    snap_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ts, frame = cam.latest()
        if frame is None:
            if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                break
            continue

        view = frame.copy()
        if show_detect and detector is not None:
            dets = detector.track(frame)
        if cmon is not None:
            states = cmon.update(dets, parked_ids={d.track_id for d in dets})
            cmon.draw(view, states)
        if dets and show_detect:
            draw_detections(view, dets)
        if show_grid:
            draw_grid(view)

        mean_v = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        hud(view, [
            f"{frame_size[0]}x{frame_size[1]}  sharp={sharpness(frame):7.1f}  "
            f"mean={mean_v:5.1f}",
            f"autofocus={'ON ' if autofocus else 'OFF'} focus={focus:6.1f}   "
            f"auto_exposure={'ON ' if auto_exp else 'OFF'} exposure={exposure:6.1f}"
            f"   zoom={zoom:.0f}",
            f"detect={'ON' if show_detect else 'OFF'}  "
            f"{len(dets)} objects   [f/[/]  e/-/=  z/x  d  g  s  p  q]",
        ])
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("f"):
            autofocus = not autofocus
            cam.set_prop(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            if not autofocus:
                cam.set_prop(cv2.CAP_PROP_FOCUS, focus)
        elif key in (ord("["), ord("]")) and not autofocus:
            focus = max(0.0, min(255.0, focus + (5 if key == ord("]") else -5)))
            cam.set_prop(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord("e"):
            auto_exp = not auto_exp
            cam.set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exp else 1)
            if not auto_exp:
                cam.set_prop(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key in (ord("-"), ord("=")) and not auto_exp:
            exposure = max(1.0, exposure * (1.15 if key == ord("=") else 0.87))
            cam.set_prop(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key in (ord("z"), ord("x")):
            zoom = max(100.0, min(500.0, zoom + (10 if key == ord("x") else -10)))
            cam.set_prop(cv2.CAP_PROP_ZOOM, zoom)
        elif key == ord("d"):
            show_detect = not show_detect
            if show_detect and detector is None:
                print("loading detector...")
                detector = Detector(cfg.detector)
            if not show_detect:
                dets = []
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("s"):
            from streetmon.storage import ts_name
            path = snap_dir / f"aim_{ts_name(ts)}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"saved {path}")
        elif key == ord("p"):
            print("\n# --- paste into config/site.yaml under `camera:` ---")
            print(f"  autofocus: {str(autofocus).lower()}")
            print(f"  focus: {None if autofocus else round(focus)}")
            print(f"  auto_exposure: {str(auto_exp).lower()}")
            print(f"  exposure: {None if auto_exp else round(exposure)}")
            print(f"  zoom: {round(zoom) if zoom != 100 else None}")
            print("# ---------------------------------------------------\n")

    cam.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
