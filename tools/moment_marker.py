#!/usr/bin/env python3
"""Mark three moments in each confirmed-sighting clip: Left, Center, Right.

    Left    the vehicle entering (or leaving) frame on one side
    Center  the best view of it -- the frame worth holding on
    Right   the opposite edge

Those three indices give the montage exact in/out points and a hero frame,
replacing the timestamp arithmetic that previously mis-trimmed clips: an event
clip starts with ring-buffer pre-roll, so its frame 0 is some unknown distance
before the event's recorded start. Marked frames need no such guessing.

    python tools/moment_marker.py                  # all confirmed sightings
    python tools/moment_marker.py --video FILE     # one arbitrary clip
    python tools/moment_marker.py --label not_police

Navigate   SPACE play/pause    a / LEFT  back one frame    d / RIGHT  forward
           trackbar drag to seek (does not pause playback)
           , / .  jump 10 frames back / forward
Mark       l  Left        c  Center        r  Right        x  clear this clip
Clips      n  next clip   p  previous clip
           s  save (also saved automatically when you change clip or quit)
           q/ESC quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.labels import LabelStore, POLICE

WINDOW = "moment marker"
BAR = "Frame"


def moments_path(repo: Path) -> Path:
    return repo / "labels" / "moments.json"


def open_window(name: str, w: int, h: int) -> None:
    """Realise the window before attaching a trackbar.

    namedWindow does not create the window under the Qt backend -- it appears
    on the first imshow, and not always within one event-loop tick. Pump a few
    and retry rather than assuming; otherwise createTrackbar raises
    "NULL window handler".
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    blank = np.zeros((h, w, 3), np.uint8)
    for _ in range(40):
        cv2.imshow(name, blank)
        cv2.waitKey(30)
        try:
            cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
            return
        except cv2.error:
            continue


def collect_clips(repo: Path, label: str, video: str | None) -> list[dict]:
    if video:
        p = Path(video)
        return [{"session": "-", "event": p.stem, "clip": p}]
    store = LabelStore.load()
    found: dict[tuple, dict] = {}
    for r in store.decisions.values():
        if r["label"] != label:
            continue
        sess = repo / "data" / r["session"]
        ef = sess / "logs" / "events.jsonl"
        if not ef.exists():
            continue
        for line in ef.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if r["track_id"] not in e.get("track_ids", []):
                continue
            clip = sess / "events" / f"{e['id']}.mp4"
            if clip.exists():
                key = (r["session"], e["id"])
                it = found.setdefault(key, {"session": r["session"],
                                            "event": e["id"], "clip": clip,
                                            "start_ts": e["start_ts"],
                                            "tracks": []})
                it["tracks"].append(r["track_id"])
            break
    return sorted(found.values(), key=lambda c: c.get("start_ts", 0))


def rel(p: Path, repo: Path) -> str:
    """Repo-relative when possible -- a --video argument may live anywhere."""
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def draw_hud(frame, clip, idx, total, marks, paused, saved):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    ids = ",".join(f"#{t}" for t in clip.get("tracks", [])) or "-"
    cv2.putText(frame, f"{clip['event']}   {clip['session']}   {ids}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    status = f"frame {idx}/{max(total-1,0)}" + ("   PAUSED" if paused else "")
    cv2.putText(frame, status, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (200, 200, 200), 2, cv2.LINE_AA)

    x = 360
    for name, key, col in (("L", "left", (255, 170, 0)),
                           ("C", "center", (0, 230, 120)),
                           ("R", "right", (255, 120, 220))):
        v = marks.get(key)
        txt = f"{name} {v if v is not None else '-'}"
        on = v is not None
        cv2.putText(frame, txt, (x, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    col if on else (110, 110, 110), 2, cv2.LINE_AA)
        if v == idx:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
            cv2.rectangle(frame, (x - 6, 40), (x + tw + 6, 70), col, 2)
        x += 150

    if not saved:
        cv2.putText(frame, "unsaved", (w - 130, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)

    # timeline strip showing where the marks sit
    y0, y1 = h - 26, h - 8
    cv2.rectangle(frame, (14, y0), (w - 14, y1), (45, 45, 45), -1)
    span = max(total - 1, 1)
    for key, col in (("left", (255, 170, 0)), ("center", (0, 230, 120)),
                     ("right", (255, 120, 220))):
        v = marks.get(key)
        if v is None:
            continue
        px = 14 + int((w - 28) * v / span)
        cv2.rectangle(frame, (px - 3, y0), (px + 3, y1), col, -1)
    cur = 14 + int((w - 28) * idx / span)
    cv2.line(frame, (cur, y0 - 4), (cur, y1 + 4), (255, 255, 255), 2)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=POLICE)
    ap.add_argument("--video", default=None)
    ap.add_argument("--index", type=int, default=None,
                    help="open only clip N (1-based) from the confirmed list")
    ap.add_argument("--list", action="store_true",
                    help="print the numbered clip list and exit")
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    clips = collect_clips(repo, args.label, args.video)
    if not clips:
        print(f"no clips found for label {args.label}")
        return 1

    store_p = moments_path(repo)
    existing = json.loads(store_p.read_text()) if store_p.exists() else {}
    if args.list:
        for i, c in enumerate(clips, 1):
            m = existing.get(f"{c['session']}|{c['event']}", {}).get("marks", {})
            done = "".join(k[0].upper() if k in m else "-"
                           for k in ("left", "center", "right"))
            print(f"  {i}. {c['event']}  {c['session']}  [{done}]")
        return 0
    if args.index is not None:
        if not 1 <= args.index <= len(clips):
            print(f"--index must be 1..{len(clips)}")
            return 1
        clips = [clips[args.index - 1]]

    data = json.loads(store_p.read_text()) if store_p.exists() else {}
    print(f"{len(clips)} clip(s); {len(data)} already marked -> {store_p}")
    print(__doc__)

    ci = 0
    dirty = False

    def key_of(c):
        return f"{c['session']}|{c['event']}"

    def save():
        nonlocal dirty
        store_p.parent.mkdir(parents=True, exist_ok=True)
        store_p.write_text(json.dumps(data, indent=1, sort_keys=True))
        dirty = False
        print(f"  saved {len(data)} clip(s)")

    first = True
    while 0 <= ci < len(clips):
        clip = clips[ci]
        cap = cv2.VideoCapture(str(clip["clip"]))
        if not cap.isOpened():
            print(f"cannot open {clip['clip']}")
            ci += 1
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        marks = data.get(key_of(clip), {}).get("marks", {})

        if first:
            ok, probe = cap.read()
            dh = int(args.width * (probe.shape[0] / probe.shape[1])) if ok else 800
            open_window(WINDOW, args.width, dh)
            try:
                cv2.createTrackbar(BAR, WINDOW, 0, max(total - 1, 1), lambda v: None)
            except cv2.error as exc:
                print(f"  (trackbar unavailable: {exc}) -- keys still work")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            first = False
        else:
            try:
                cv2.setTrackbarMax(BAR, WINDOW, max(total - 1, 1))
                cv2.setTrackbarPos(BAR, WINDOW, 0)
            except cv2.error:
                pass

        frame_idx, frame, paused = -1, None, True
        advance = 0
        action = None
        while True:
            need_read = False
            seek_to = -1
            try:
                tb = cv2.getTrackbarPos(BAR, WINDOW)
            except cv2.error:
                tb = frame_idx
            # Rule 2: this reads the user's drag, because the previous
            # iteration's waitKey already processed the GUI events.
            if tb != frame_idx:
                seek_to, need_read = tb, True
            # Rule 3: a plain `if` -- an `elif` here would block playback
            # whenever the trackbar check ran.
            if not need_read and (not paused or advance):
                nxt = frame_idx + (advance or 1)
                if nxt != frame_idx + 1:
                    seek_to = max(0, min(total - 1, nxt))
                need_read = True
                advance = 0

            if need_read:
                if seek_to >= 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                ok, f = cap.read()
                if not ok:
                    paused = True
                    advance = 0
                    if frame is None:
                        break
                else:
                    frame = f
                    # Rule 1: track it ourselves; POS_FRAMES is the *next*
                    # frame, not the displayed one.
                    frame_idx = seek_to if seek_to >= 0 else frame_idx + 1
                    try:
                        cv2.setTrackbarPos(BAR, WINDOW, frame_idx)
                    except cv2.error:
                        pass

            if frame is None:
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    action = "quit"
                    break
                continue

            view = draw_hud(frame.copy(), clip, frame_idx, total, marks,
                            paused, not dirty)
            cv2.imshow(WINDOW, view)

            # Rule 5
            key = cv2.waitKey(30 if paused else max(1, int(1000 / fps))) & 0xFF

            if key in (ord("q"), 27):
                action = "quit"; break
            elif key == 32:
                paused = not paused            # Rule 4: only keys pause
            elif key in (ord("d"), 83):
                paused = True; advance = 1
            elif key in (ord("a"), 81):
                paused = True; advance = -1
            elif key == ord("."):
                paused = True; advance = 10
            elif key == ord(","):
                paused = True; advance = -10
            elif key in (ord("l"), ord("c"), ord("r")):
                name = {"l": "left", "c": "center", "r": "right"}[chr(key)]
                marks[name] = frame_idx
                data[key_of(clip)] = {
                    "session": clip["session"], "event": clip["event"],
                    "clip": rel(clip["clip"], repo),
                    "fps": round(fps, 3), "n_frames": total,
                    "tracks": clip.get("tracks", []), "marks": dict(marks),
                }
                dirty = True
                print(f"  {clip['event']}  {name} = frame {frame_idx}")
            elif key == ord("x"):
                marks.clear()
                data.pop(key_of(clip), None)
                dirty = True
                print(f"  {clip['event']}  cleared")
            elif key == ord("s"):
                save()
            elif key == ord("n"):
                action = "next"; break
            elif key == ord("p"):
                action = "prev"; break

        cap.release()
        if dirty:
            save()
        if action == "quit":
            break
        ci += 1 if action != "prev" else -1
        ci = max(0, ci)

    cv2.destroyAllWindows()
    done = sum(1 for v in data.values() if len(v.get("marks", {})) == 3)
    print(f"\n{done}/{len(clips)} clip(s) fully marked (L+C+R)")
    print(f"stored in {store_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
