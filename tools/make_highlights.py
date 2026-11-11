#!/usr/bin/env python3
"""Compile every confirmed parking-police sighting into one video.

Nine months of collection may yield a handful of passes, scattered across
sessions as isolated clips nobody will go hunting for. This stitches the
confirmed ones into a single file in chronological order, each introduced by a
card giving date, time, direction and speed, so the whole evidence set can be
watched end to end.

Two labels often point at one clip: a track that fragments and is re-acquired
gets two ids for one pass. Clips are therefore deduplicated, and the card names
every track id involved.

    python tools/make_highlights.py
    python tools/make_highlights.py --label parking_police --out /tmp/police.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.config import load_config
from streetmon.labels import LabelStore, POLICE
from streetmon.video import EventVideoWriter

W, H = 1920, 1080


def card(lines: list[tuple[str, float, tuple]], seconds: float, fps: float) -> list:
    """A title frame, repeated for `seconds`."""
    img = np.full((H, W, 3), 16, np.uint8)
    y = H // 2 - sum(int(44 * s) for _, s, _ in lines) // 2
    for text, scale, colour in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
        y += int(44 * scale) + 18
    return [img] * max(1, int(seconds * fps))


def banner(frame, text: str, sub: str = "") -> np.ndarray:
    h = 74 if sub else 50
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=POLICE)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--card-sec", type=float, default=2.0)
    ap.add_argument("--pad-sec", type=float, default=1.5,
                    help="context kept either side of the vehicle's presence")
    ap.add_argument("--full-clips", action="store_true",
                    help="keep whole clips instead of trimming to the vehicle")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / args.root
    cfg = load_config()

    store = LabelStore.load()
    picked = [r for r in store.decisions.values() if r["label"] == args.label]
    if not picked:
        print(f"no tracks labelled {args.label}")
        return 1

    # Resolve each label to its event clip, collapsing the fragmented pairs.
    clips: dict[tuple, dict] = {}
    for r in picked:
        sess = root / r["session"]
        ef = sess / "logs" / "events.jsonl"
        if not ef.exists():
            continue
        for line in ef.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if r["track_id"] not in e.get("track_ids", []):
                continue
            path = sess / "events" / f"{e['id']}.mp4"
            if not path.exists():
                break
            key = (r["session"], e["id"])
            entry = clips.setdefault(key, {"path": path, "ev": e, "tracks": [],
                                           "session": r["session"]})
            entry["tracks"].append(r["track_id"])
            break
    if not clips:
        print("no event clips found for those labels")
        return 1

    ordered = sorted(clips.values(), key=lambda c: c["ev"]["start_ts"])
    print(f"{len(picked)} labels -> {len(ordered)} distinct clips")

    # pull per-track kinematics for the cards
    def track_info(session: str, tid: int) -> dict:
        tf = root / session / "logs" / "tracks.jsonl"
        if not tf.exists():
            return {}
        for line in tf.read_text().splitlines():
            if f'"id":{tid},' in line:
                return json.loads(line)
        return {}

    out = Path(args.out) if args.out else (
        root / "highlights" /
        f"parking-police_{datetime.now():%Y%m%d}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = EventVideoWriter(out, (W, H), args.fps, quality=22)
    if not writer.ok:
        print("could not start the video writer")
        return 2

    total = 0
    for n, c in enumerate(ordered, 1):
        ev, sess = c["ev"], c["session"]
        when = datetime.fromtimestamp(ev["start_ts"])
        info = track_info(sess, c["tracks"][0])
        direction = info.get("direction", "?")
        speed = info.get("median_speed_bl_s")
        ids = ", ".join(f"#{t}" for t in sorted(c["tracks"]))
        print(f"  [{n}/{len(ordered)}] {when:%Y-%m-%d %H:%M:%S}  {ev['id']}  {ids}")

        for f in card([
            (f"Sighting {n} of {len(ordered)}", 1.1, (0, 220, 255)),
            (when.strftime("%A %d %B %Y   %H:%M:%S"), 0.95, (240, 240, 240)),
            (f"track {ids}    direction {direction}"
             + (f"    {speed:.2f} BL/s" if speed else ""), 0.7, (170, 170, 170)),
            (f"{sess}   {ev['id']}", 0.5, (120, 120, 120)),
        ], args.card_sec, args.fps):
            writer.write(f)
            total += 1

        # A clip runs 10-20 s but the vehicle crosses in about two. Keep only
        # the window where it is actually present, plus a little context --
        # otherwise most of the compilation is an empty street.
        lo_f, hi_f = 0, 10 ** 9
        if not args.full_clips:
            spans = [track_info(sess, t) for t in c["tracks"]]
            spans = [x for x in spans if x.get("first_ts")]
            if spans:
                a = min(x["first_ts"] for x in spans) - args.pad_sec
                b = max(x["last_ts"] for x in spans) + args.pad_sec
                lo_f = max(0, int((a - ev["start_ts"]) * args.fps))
                hi_f = int((b - ev["start_ts"]) * args.fps)

        cap = cv2.VideoCapture(str(c["path"]))
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < lo_f or idx > hi_f:
                idx += 1
                continue
            kept += 1
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            t = datetime.fromtimestamp(ev["start_ts"] + idx / max(args.fps, 1))
            banner(frame,
                   f"{n}/{len(ordered)}   {when:%a %d %b %Y}   {t:%H:%M:%S}",
                   f"track {ids}   {direction}   {ev['id']}")
            writer.write(frame)
            total += 1
            idx += 1
        cap.release()
        if not args.full_clips:
            print(f"        kept {kept} of {idx} frames "
                  f"({kept/max(args.fps,1):.1f}s of {idx/max(args.fps,1):.1f}s)")

    stats = writer.close()
    print(f"\nwrote {out}")
    print(f"  {total} frames, {total/args.fps:.0f}s, "
          f"{stats['bytes']/1e6:.1f} MB, {stats['encoder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
