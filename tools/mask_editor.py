#!/usr/bin/env python3
"""Draw the line where background removal starts, with a live preview.

Everything above the line you draw is treated as background and blacked out --
except vehicles and people, which instance segmentation carves back out, so a
car parked against the far curb survives even though its body sits well above
the line.

The collector's auto-fitted curb is loaded as a starting point, but it was
fitted to where cars' wheels meet the road, which is not necessarily where you
want the mask to begin. Drag it to the sidewalk edge you actually mean.

    python tools/mask_editor.py                 # newest collected still
    python tools/mask_editor.py --image FILE
    python tools/mask_editor.py --day 2026-11-09

Mouse    left click   add a point        right click  undo last point
         drag a point move it (grab within 20 px)
Keys     p  preview on/off (masked vs original)      c  clear the line
         [  ]  nudge the margin down/up              r  reload the fitted curb
         s  save to config/mask_line.yaml            q  quit
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.config import load_config

W, H = 1920, 1080
WINDOW = "mask editor"      # ASCII and short: Qt is fussy about lookup by name
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
PERSON_CLASS = 0

state = {"pts": [], "drag": None, "preview": True, "margin": 18, "dirty": False}


def on_mouse(event, x, y, flags, param):
    sx, sy = param["scale"]
    px, py = x / sx, y / sy
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (ax, ay) in enumerate(state["pts"]):
            if abs(ax - px) < 20 and abs(ay - py) < 20:
                state["drag"] = i
                return
        state["pts"].append([px, py])
        state["pts"].sort(key=lambda p: p[0])
        state["dirty"] = True
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
        state["pts"][state["drag"]] = [px, py]
        state["dirty"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        if state["drag"] is not None:
            state["pts"].sort(key=lambda p: p[0])
        state["drag"] = None
    elif event == cv2.EVENT_RBUTTONDOWN and state["pts"]:
        state["pts"].pop()
        state["dirty"] = True


def region_from(pts, margin):
    """Everything above the polyline (extended to both frame edges)."""
    if len(pts) < 2:
        return None
    p = sorted(pts, key=lambda q: q[0])
    poly = [[0, 0]]
    poly.append([0, p[0][1] + margin])
    for x, y in p:
        poly.append([x, y + margin])
    poly.append([W, p[-1][1] + margin])
    poly.append([W, 0])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m


def load_fitted(repo):
    f = repo / "config" / "curb.yaml"
    if not f.exists():
        return []
    doc = yaml.safe_load(f.read_text())
    far = next((c for c in doc.get("curbs", []) if c["name"] == "far"), None)
    if not far:
        return []
    fw, fh = doc.get("frame_size", [W, H])
    return [[p[0] * W / fw, p[1] * H / fh] for p in far["points"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None)
    ap.add_argument("--day", default=None)
    ap.add_argument("--seg-model", default="yolo11m-seg.pt")
    ap.add_argument("--display-width", type=int, default=1500)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    cfg = load_config()

    if args.image:
        path = args.image
    else:
        pat = f"data/{args.day or '2???-??-??'}/session_*/timelapse/*.jpg"
        files = sorted(glob.glob(str(repo / pat)))
        if not files:
            print("no timelapse frames found")
            return 1
        path = files[len(files) // 2] if args.day else files[-1]
    img = cv2.imread(path)
    if img is None:
        print(f"cannot read {path}")
        return 1
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    print(f"using {path}")

    print("running segmentation once (vehicles/people are preserved)...")
    from ultralytics import YOLO
    r = YOLO(args.seg_model).predict(img, imgsz=960, conf=0.25, verbose=False,
                                     device=cfg.detector.device)[0]
    keep = np.zeros((H, W), np.uint8)
    people = np.zeros((H, W), np.uint8)
    if r.masks is not None:
        for mk, c in zip(r.masks.data.cpu().numpy(),
                         r.boxes.cls.cpu().numpy().astype(int)):
            m = cv2.resize(mk.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)
            if c in VEHICLE_CLASSES:
                keep |= m
            elif c == PERSON_CLASS:
                people |= m
    if people.any():
        people = cv2.dilate(people, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)))
    print(f"  vehicles kept: {int(keep.sum())} px   "
          f"people removed everywhere: {int(people.sum())} px")

    saved = repo / "config" / "mask_line.yaml"
    if saved.exists():
        doc = yaml.safe_load(saved.read_text())
        fw, fh = doc.get("frame_size", [W, H])
        state["pts"] = [[p[0] * W / fw, p[1] * H / fh] for p in doc["points"]]
        state["margin"] = int(doc.get("margin", 18))
        print(f"loaded existing mask line ({len(state['pts'])} points)")
    else:
        state["pts"] = load_fitted(repo)
        print(f"starting from the auto-fitted curb ({len(state['pts'])} points)")

    dw = args.display_width
    dh = int(H * dw / W)
    scale = (dw / W, dh / H)

    # Realise the window before attaching anything to it. namedWindow alone
    # does not create it under the Qt backend -- it appears on the first
    # imshow, and even then not always within a single event-loop tick, so
    # pump a few and retry rather than assuming.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, dw, dh)
    blank = np.zeros((dh, dw, 3), np.uint8)
    attached = False
    for _ in range(40):
        cv2.imshow(WINDOW, blank)
        cv2.waitKey(30)
        try:
            cv2.setMouseCallback(WINDOW, on_mouse, {"scale": scale})
            attached = True
            break
        except cv2.error:
            continue
    if not attached:
        print("could not attach the mouse callback -- is DISPLAY set?")
        return 2
    print(__doc__)

    while True:
        region = region_from(state["pts"], state["margin"])
        view = img.copy()
        if state["preview"] and region is not None:
            view[(region == 1) & (keep == 0)] = 0
            view[people == 1] = 0        # people go everywhere, not just above
        elif region is not None:
            tint = view.copy()
            tint[region == 1] = (0, 0, 120)
            cv2.addWeighted(tint, 0.35, view, 0.65, 0, view)

        if len(state["pts"]) >= 2:
            p = np.array([[x, y + state["margin"]] for x, y in state["pts"]], np.int32)
            cv2.polylines(view, [p], False, (0, 220, 255), 3)
        for x, y in state["pts"]:
            cv2.circle(view, (int(x), int(y + state["margin"])), 9, (0, 220, 255), -1)
            cv2.circle(view, (int(x), int(y + state["margin"])), 9, (0, 0, 0), 2)

        pct = 0.0 if region is None else 100 * (((region == 1) & (keep == 0)).mean())
        lines = [f"{len(state['pts'])} points   margin {state['margin']} px   "
                 f"blacked out {pct:.0f}% of frame",
                 "L-click add / drag   R-click undo   p preview   [ ] margin   "
                 "c clear   r refit   s save   q quit"]
        for i, t in enumerate(lines):
            y = 34 + i * 34
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            cv2.rectangle(view, (10, y - th - 10), (20 + tw, y + 10), (0, 0, 0), -1)
            cv2.putText(view, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, cv2.resize(view, (dw, dh), interpolation=cv2.INTER_AREA))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            if state["dirty"]:
                print("unsaved changes — 's' to save, or 'q' again to discard")
                state["dirty"] = False
                continue
            break
        elif k == ord("p"):
            state["preview"] = not state["preview"]
        elif k == ord("c"):
            state["pts"] = []
            state["dirty"] = True
        elif k == ord("r"):
            state["pts"] = load_fitted(repo)
            state["dirty"] = True
        elif k == ord("["):
            state["margin"] = max(-200, state["margin"] - 4)
            state["dirty"] = True
        elif k == ord("]"):
            state["margin"] = min(400, state["margin"] + 4)
            state["dirty"] = True
        elif k == ord("s"):
            if len(state["pts"]) < 2:
                print("need at least 2 points")
                continue
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(
                "# Where background removal starts, drawn by hand in\n"
                "# tools/mask_editor.py. Everything ABOVE this line is blacked\n"
                "# out except vehicles and people, which segmentation keeps.\n"
                "# Coordinates are pixels at `frame_size` and rescale with it.\n"
                + yaml.safe_dump({"frame_size": [W, H],
                                  "margin": int(state["margin"]),
                                  "points": [[float(x), float(y)]
                                             for x, y in state["pts"]]},
                                 sort_keys=False))
            state["dirty"] = False
            print(f"saved {len(state['pts'])} points to {saved}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
