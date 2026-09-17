#!/usr/bin/env python3
"""Review tracked vehicles and label the parking-enforcement ones.

Stage 1 of the project is finding positives, and this is the tool that does it.
The unit of review is a **track**, not a crop: one vehicle visit, one decision,
with every crop of that vehicle shown side by side so a bad angle in one frame
does not cost you the sighting.

    python tools/review.py                 # triage candidates, unlabelled first
    python tools/review.py --all           # every vehicle track
    python tools/review.py --kinematic     # only the slow/stopping ones
    python tools/review.py --relabel       # include tracks already decided
    python tools/review.py --day 2026-09-17

Keys
    y      this IS a parking-enforcement vehicle
    n      it is not
    u      unsure -- comes back next time
    v      play the event clip this track appears in
    a / <-  previous track        d / ->  next track
    f       jump to the next unlabelled track
    q/ESC  quit (labels are saved as you go, not on exit)

Labels append to labels/labels.jsonl. Nothing is overwritten: relabelling adds
a new record and the most recent one wins, so changing your mind is safe and
the history stays auditable.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.config import load_config
from streetmon.labels import LabelStore, NOT_POLICE, POLICE, UNSURE
from streetmon.triage import analyse

WINDOW = "review — y=police  n=not  u=unsure  v=clip  a/d=prev/next  q=quit"
CLIP_WINDOW = "event clip — SPACE pause, a/d step, q close"
PANEL_W = 420
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

LABEL_COLOR = {POLICE: (0, 220, 0), NOT_POLICE: (90, 90, 200),
               UNSURE: (0, 200, 255), None: (140, 140, 140)}


def open_window(name: str, width: int, height: int) -> None:
    """Create a window and force the backend to realise it.

    cv2.createTrackbar fails with "NULL window handler" if the window has been
    named but never drawn to — namedWindow alone does not realise it under the
    Qt backend. Showing one placeholder frame and pumping the event loop does.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, np.full((max(height, 1), max(width, 1), 3), 24, np.uint8))
    cv2.waitKey(1)


def add_trackbar(name: str, window: str, maximum: int) -> bool:
    """Create a trackbar, degrading to keyboard-only navigation if it fails."""
    try:
        cv2.createTrackbar(name, window, 0, max(int(maximum), 1), lambda v: None)
        return True
    except cv2.error as exc:
        print(f"  (trackbar unavailable: {exc.err.strip() if hasattr(exc, 'err') else exc}")
        print("   — keyboard navigation still works)")
        return False


def session_key(session_dir: Path) -> str:
    return f"{session_dir.parent.name}/{session_dir.name}"


def load_tracks(root: Path, day: str | None, cfg) -> list[dict]:
    """Every vehicle track across sessions, with its crops resolved to paths."""
    pattern = f"{day or '*'}/session_*"
    items: list[dict] = []
    for sess in sorted(root.glob(pattern)):
        tf = sess / "logs" / "tracks.jsonl"
        if not tf.exists():
            continue
        # track_id -> event id, so the reviewer can watch the clip it came from
        ev_of: dict[int, str] = {}
        ef = sess / "logs" / "events.jsonl"
        if ef.exists():
            for line in ef.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for tid in e.get("track_ids", []):
                    ev_of.setdefault(int(tid), e["id"])
        for line in tf.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ci") not in VEHICLE_CLASSES:
                continue
            paths = []
            for name in r.get("crops", []):
                for sub in ("vehicle", "person"):
                    p = sess / "crops" / sub / name
                    if p.exists():
                        paths.append(p)
                        break
            if not paths:
                continue
            r["_session"] = session_key(sess)
            r["_dir"] = sess
            r["_crops"] = paths
            r["_event"] = ev_of.get(int(r["id"]))
            items.append(r)
    items.sort(key=lambda r: r["first_ts"])
    return items


def filmstrip(paths: list[Path], cfg, height: int, width: int) -> np.ndarray:
    """Up to six crops in a row, each annotated with its own triage scores."""
    canvas = np.full((height, width, 3), 24, np.uint8)
    use = paths[:6]
    if not use:
        return canvas
    cell_w = width // len(use)
    for i, p in enumerate(use):
        im = cv2.imread(str(p))
        if im is None:
            continue
        r = analyse(im, cfg.triage)
        scale = min((cell_w - 8) / im.shape[1], (height - 34) / im.shape[0])
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)),
                             max(1, int(im.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
        x = i * cell_w + (cell_w - im.shape[1]) // 2
        y = 30 + (height - 30 - im.shape[0]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        txt = f"w{r.white_frac:.2f} rb{r.red_band_frac:.3f} d{r.red_dashes}"
        col = (0, 255, 120) if r.is_police_candidate else (185, 185, 185)
        cv2.putText(canvas, txt, (i * cell_w + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        if i:
            cv2.line(canvas, (i * cell_w, 0), (i * cell_w, height), (60, 60, 60), 1)
    return canvas


def info_panel(rec: dict, label: str | None, idx: int, total: int,
               counts: dict, height: int) -> np.ndarray:
    import datetime as dt
    p = np.full((height, PANEL_W, 3), 18, np.uint8)
    y = [28]

    def line(text, color=(225, 225, 225), size=0.5, gap=22, bold=1):
        cv2.putText(p, text, (14, y[0]), cv2.FONT_HERSHEY_SIMPLEX, size,
                    color, bold, cv2.LINE_AA)
        y[0] += gap

    tri = rec.get("triage", {})
    line(f"track #{rec['id']}   {idx + 1} / {total}", (255, 255, 255), 0.62, 30, 2)
    line(rec["_session"], (150, 150, 150), 0.42)
    line(dt.datetime.fromtimestamp(rec["first_ts"]).strftime("%a %d %b  %H:%M:%S"),
         (150, 150, 150), 0.42, 26)

    line(f"class        {rec['c']}")
    line(f"motion       {rec.get('motion_class', '?')}")
    line(f"direction    {rec.get('direction', '?')}")
    line(f"duration     {rec['duration_sec']:.1f} s")
    line(f"speed (med)  {rec.get('median_speed_bl_s', 0):.2f} BL/s")
    line(f"travelled    {rec.get('net_bl', 0):.1f} body-lengths")
    line(f"stops        {rec.get('stops', 0)}", gap=26)

    kin = rec.get("kinematic_candidate")
    col = (0, 255, 120) if kin else (150, 150, 150)
    line(f"kinematic    {'FLAG ' + str(rec.get('kinematic_reason')) if kin else 'no'}",
         col)
    colour = tri.get("colour_candidate")
    col = (0, 255, 120) if colour else (150, 150, 150)
    line(f"colour       {'FLAG' if colour else 'no'}   "
         f"w{tri.get('best_white', 0):.2f} rb{tri.get('best_red_band', 0):.3f}", col)
    if kin and colour:
        line("BOTH CHANNELS AGREE", (0, 255, 255), 0.56, 28, 2)
    y[0] += 8

    line(f"crops        {len(rec['_crops'])}", (150, 150, 150), 0.44)
    line(f"event        {rec.get('_event') or '-'}", (150, 150, 150), 0.44, 30)

    cv2.rectangle(p, (10, y[0] - 6), (PANEL_W - 10, y[0] + 34),
                  LABEL_COLOR[label], 2)
    cv2.putText(p, (label or "unlabelled").upper(), (22, y[0] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, LABEL_COLOR[label], 2, cv2.LINE_AA)
    y[0] += 56

    line(f"police {counts.get(POLICE, 0)}   not {counts.get(NOT_POLICE, 0)}   "
         f"unsure {counts.get(UNSURE, 0)}", (170, 170, 170), 0.46, 24)
    line("y police   n not   u unsure", (140, 140, 140), 0.44, 20)
    line("v clip   a/d prev/next   f next new", (140, 140, 140), 0.44, 20)
    line("q quit", (140, 140, 140), 0.44)
    return p


def play_clip(path: Path, fps: float = 5.0) -> None:
    """Event-clip playback (VideoCapture pattern: trackbar read after waitKey)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"cannot open {path}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    open_window(CLIP_WINDOW, 1280, 720)
    has_bar = total > 1 and add_trackbar("Frame", CLIP_WINDOW, total - 1)

    frame_idx, frame, paused = -1, None, False
    while True:
        need_read = False
        seek_to = -1
        if has_bar:
            tb = cv2.getTrackbarPos("Frame", CLIP_WINDOW)
            if tb != frame_idx:
                seek_to, need_read = tb, True
        if not need_read and not paused:
            need_read = True
        if need_read:
            if seek_to >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
            ok, f = cap.read()
            if not ok:
                break
            frame = f
            frame_idx = seek_to if seek_to >= 0 else frame_idx + 1
            if has_bar:
                cv2.setTrackbarPos("Frame", CLIP_WINDOW, frame_idx)
        if frame is None:
            break

        view = frame.copy()
        status = f"{path.name}   frame {frame_idx}/{max(total - 1, 0)}"
        if paused:
            status += "   PAUSED"
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(CLIP_WINDOW, view)

        key = cv2.waitKey(30 if paused else max(1, int(1000 / src_fps))) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (32, ord("p")):
            paused = not paused
        elif key in (ord("d"), 83) and has_bar:
            paused = True
            if frame_idx < total - 1:
                cv2.setTrackbarPos("Frame", CLIP_WINDOW, frame_idx + 1)
        elif key in (ord("a"), 81) and has_bar:
            paused = True
            if frame_idx > 0:
                cv2.setTrackbarPos("Frame", CLIP_WINDOW, frame_idx - 1)
    cap.release()
    try:
        cv2.destroyWindow(CLIP_WINDOW)
    except cv2.error:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default="data")
    ap.add_argument("--day", default=None, help="restrict to YYYY-MM-DD")
    ap.add_argument("--all", action="store_true", help="every vehicle track")
    ap.add_argument("--kinematic", action="store_true",
                    help="only kinematic candidates")
    ap.add_argument("--colour", "--color", dest="colour", action="store_true",
                    help="only colour candidates")
    ap.add_argument("--relabel", action="store_true",
                    help="include tracks already labelled")
    ap.add_argument("--height", type=int, default=560)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    tracks = load_tracks(root, args.day, cfg)
    if not args.all:
        def is_cand(r):
            if args.kinematic:
                return bool(r.get("kinematic_candidate"))
            if args.colour:
                return bool(r.get("triage", {}).get("colour_candidate"))
            return bool(r.get("kinematic_candidate")
                        or r.get("triage", {}).get("colour_candidate"))
        tracks = [r for r in tracks if is_cand(r)]

    store = LabelStore.load()
    if not args.relabel:
        tracks = [r for r in tracks
                  if store.get(r["_session"], r["id"]) in (None, UNSURE)]

    if not tracks:
        print("nothing to review.")
        print(f"  labels so far: {store.counts()}")
        print("  try --all, or --relabel to revisit decided tracks.")
        return 0

    print(f"{len(tracks)} track(s) to review. Labels -> {store.path}")
    print(__doc__)

    strip_w = 1180
    open_window(WINDOW, strip_w + PANEL_W, args.height)
    has_bar = add_trackbar("Track", WINDOW, len(tracks) - 1)

    idx = 0
    while True:
        idx = max(0, min(len(tracks) - 1, idx))
        rec = tracks[idx]
        label = store.get(rec["_session"], rec["id"])
        view = np.hstack([
            filmstrip(rec["_crops"], cfg, args.height, strip_w),
            info_panel(rec, label, idx, len(tracks), store.counts(), args.height),
        ])
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(30) & 0xFF

        # Read the trackbar after waitKey so a drag is seen, not the value we set.
        if has_bar:
            tb = cv2.getTrackbarPos("Track", WINDOW)
            if tb != idx:
                idx = tb
                continue

        if key in (ord("q"), 27):
            break
        elif key in (ord("y"), ord("p")):
            store.put(rec["_session"], rec["id"], POLICE,
                      extra={"crops": [p.name for p in rec["_crops"]]})
            print(f"  #{rec['id']} -> POLICE")
            idx += 1
        elif key == ord("n"):
            store.put(rec["_session"], rec["id"], NOT_POLICE)
            idx += 1
        elif key == ord("u"):
            store.put(rec["_session"], rec["id"], UNSURE)
            idx += 1
        elif key in (ord("d"), 83):
            idx += 1
        elif key in (ord("a"), 81):
            idx -= 1
        elif key == ord("f"):
            nxt = next((i for i in range(idx + 1, len(tracks))
                        if store.get(tracks[i]["_session"], tracks[i]["id"]) is None),
                       None)
            if nxt is None:
                print("  no further unlabelled tracks")
            else:
                idx = nxt
        elif key == ord("v"):
            eid = rec.get("_event")
            if not eid:
                print(f"  #{rec['id']} has no event clip")
            else:
                clip = rec["_dir"] / "events" / f"{eid}.mp4"
                if clip.exists():
                    play_clip(clip, float(cfg.capture.event.fps))
                else:
                    print(f"  clip not found: {clip}")

        idx = max(0, min(len(tracks) - 1, idx))
        if has_bar:
            cv2.setTrackbarPos("Track", WINDOW, idx)

    cv2.destroyAllWindows()
    c = store.counts()
    print(f"\nlabels: police={c[POLICE]}  not={c[NOT_POLICE]}  unsure={c[UNSURE]}")
    print(f"stored in {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
