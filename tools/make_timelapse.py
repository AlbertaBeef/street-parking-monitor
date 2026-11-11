#!/usr/bin/env python3
"""Render collected timelapse stills into an accelerated video.

Snapshots are taken every 10 s during enforcement hours and every 5 min
otherwise, so a single weekday is ~3,700 frames. Played at 30 fps that is a
300x speed-up: a ten-hour day becomes two minutes, which is the only practical
way to actually watch how the curb fills and empties.

    python tools/make_timelapse.py                     # most recent full day
    python tools/make_timelapse.py --day 2026-11-09
    python tools/make_timelapse.py --day '2026-11-*'   # several days
    python tools/make_timelapse.py --fps 60 --hours 8 18
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.video import EventVideoWriter


def stamp_of(path: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(os.path.basename(path)[:-4], "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--day", default=None,
                    help="YYYY-MM-DD, or a glob like '2026-11-*'")
    ap.add_argument("--hours", nargs=2, type=int, default=None,
                    metavar=("FROM", "TO"), help="keep only this hour range")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--quality", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-stamp", action="store_true")
    ap.add_argument("--interval", type=float, default=None, metavar="SEC",
                    help="keep at most one frame per SEC seconds. Capture runs "
                         "at 10 s in active hours and 300 s when idle, so this "
                         "thins the dense stretches without dropping the sparse "
                         "ones.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / args.root

    day = args.day
    if day is None:
        days = sorted(p.name for p in root.glob("2???-??-??") if p.is_dir())
        if not days:
            print("no data directories found")
            return 1
        day = days[-2] if len(days) > 1 else days[-1]   # last *complete* day
        print(f"no --day given; using the most recent complete day: {day}")

    files = sorted(glob.glob(str(root / day / "session_*" / "timelapse" / "*.jpg")))
    if not files:
        print(f"no timelapse frames for {day}")
        return 1

    picked = []
    for f in files:
        s = stamp_of(f)
        if s is None:
            continue
        if args.hours and not (args.hours[0] <= s.hour < args.hours[1]):
            continue
        picked.append((s, f))
    picked.sort()
    if not picked:
        print("no frames in that hour range")
        return 1

    if args.interval:
        thinned = []
        last = None
        for s_, f in picked:
            if last is None or (s_ - last).total_seconds() >= args.interval - 0.5:
                thinned.append((s_, f))
                last = s_
        print(f"thinned {len(picked)} -> {len(thinned)} frames "
              f"(>= {args.interval:g} s apart)")
        picked = thinned

    span = (picked[-1][0] - picked[0][0]).total_seconds()
    out_sec = len(picked) / args.fps
    print(f"{len(picked)} frames  "
          f"{picked[0][0]:%Y-%m-%d %H:%M} -> {picked[-1][0]:%H:%M}  "
          f"({span/3600:.1f} h of wall clock)")
    print(f"at {args.fps:g} fps -> {out_sec:.0f} s of video  "
          f"= {span/max(out_sec,1):.0f}x speed-up")

    first = cv2.imread(picked[0][1])
    if first is None:
        print("could not read the first frame")
        return 1
    h, w = first.shape[:2]
    if w != args.width:
        h = int(h * args.width / w)
        w = args.width
    h -= h % 2
    w -= w % 2

    out = Path(args.out) if args.out else (
        root / "highlights" / f"timelapse_{day.replace('*','all')}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = EventVideoWriter(out, (w, h), args.fps, quality=args.quality)
    if not writer.ok:
        print("could not start the video writer")
        return 2

    written = skipped = 0
    for i, (s, f) in enumerate(picked):
        img = cv2.imread(f)
        if img is None:
            skipped += 1
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if not args.no_stamp:
            label = s.strftime("%a %d %b %Y   %H:%M:%S")
            cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
            cv2.putText(img, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (255, 255, 255), 2, cv2.LINE_AA)
            # progress bar: where we are in the day
            frac = i / max(len(picked) - 1, 1)
            cv2.rectangle(img, (0, 46), (int(w * frac), 51), (0, 200, 255), -1)
        writer.write(img)
        written += 1
        if written % 500 == 0:
            print(f"  {written}/{len(picked)} frames...")

    stats = writer.close()
    print(f"\nwrote {out}")
    print(f"  {written} frames ({skipped} unreadable), "
          f"{written/args.fps:.0f}s, {stats['bytes']/1e6:.1f} MB, {stats['encoder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
