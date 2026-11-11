#!/usr/bin/env python3
"""Timelapse over the whole archive that slows into every confirmed sighting.

The archive is weeks of near-identical street, punctuated by a handful of
two-second events that are the entire point. A flat timelapse renders those
events as one blurred frame; playing the events alone throws away the context
of how rare they are. This does both: it races through the quiet stretches,
decelerates as a sighting approaches, cuts to the event clip at real speed,
then accelerates away.

Mechanics. A virtual clock walks the archive. Each output frame advances it by
`speed / fps` seconds, where `speed` ramps geometrically from `--max-speed`
down to the timelapse's own resolution as an event nears -- there is no point
going slower than the stills allow, it would just freeze. At the event the
source switches to the recorded clip, played frame-for-frame at real speed.

    python tools/make_story.py
    python tools/make_story.py --max-speed 6000 --ramp-min 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import json
import os
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streetmon.config import load_config
from streetmon.labels import LabelStore, POLICE
from streetmon.video import EventVideoWriter

W, H = 1920, 1080
TEAL = (208, 224, 32)    # BGR; RGB(32, 224, 208) -- reads as teal on black


def stamp_of(p: str):
    try:
        return dt.datetime.strptime(os.path.basename(p)[:-4], "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return None


def overlay(img, when: dt.datetime, speed: float, note: str = "",
            hold: bool = False):
    """Timestamp left, state label centred, log-scaled speed gauge right.

    The rate spans four orders of magnitude (1x to 10000x), so a linear bar
    would leave everything below ~1000x pinned at zero. A log scale gives each
    decade equal width, which makes the two-stage ramp legible: the outer
    stage walks one decade, the inner stage walks the rest.
    """
    Wf = img.shape[1]
    BANNER = 104
    BAR_W, BAR_H = 900, 26
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.95                       # shared by the clock and the label

    x1 = Wf - 20
    x0 = x1 - BAR_W
    by = 24

    cv2.rectangle(img, (0, 0), (Wf, BANNER), (0, 0, 0), -1)

    clock = when.strftime("%a %d %b %Y   %H:%M:%S")
    (cw, _), _ = cv2.getTextSize(clock, FONT, SCALE, 2)
    cv2.putText(img, clock, (20, 58), FONT, SCALE, (255, 255, 255), 2, cv2.LINE_AA)

    # Three states, not two. The hold is a dwell on one hand-marked frame --
    # playback has stopped, so calling it REAL TIME would be a lie: nothing is
    # moving at any rate at all.
    real = speed <= 1.5
    if hold:
        label, colour = "HOLD", TEAL
    elif real:
        label, colour = "REAL TIME", (0, 220, 255)
    else:
        label, colour = "ACCELERATED", (225, 225, 225)

    # centred in the gap between the clock and the gauge
    (lw, _), _ = cv2.getTextSize(label, FONT, SCALE, 2)
    lx = max(20 + cw + 24, (20 + cw + x0 - lw) // 2)
    cv2.putText(img, label, (lx, 58), FONT, SCALE, colour, 2, cv2.LINE_AA)

    cv2.rectangle(img, (x0, by), (x1, by + BAR_H), (52, 52, 52), -1)
    frac = min(1.0, max(0.0, math.log10(max(speed, 1.0)) / 4.0))
    if frac > 0:
        cv2.rectangle(img, (x0, by), (x0 + int(BAR_W * frac), by + BAR_H),
                      colour, -1)
    cv2.rectangle(img, (x0, by), (x1, by + BAR_H), (130, 130, 130), 1)

    for d, tick in enumerate(["1X", "10X", "100X", "1000X", "10000X"]):
        tx = x0 + int(BAR_W * d / 4)
        cv2.line(img, (tx, by), (tx, by + BAR_H), (130, 130, 130), 1)
        (tw, th), _ = cv2.getTextSize(tick, FONT, 0.62, 2)
        # the end ticks sit at the bar's edges, so a centred label would run
        # off the frame -- clamp them inside it
        tlx = min(max(tx - tw // 2, 6), Wf - tw - 6)
        cv2.putText(img, tick, (tlx, by + BAR_H + 8 + th), FONT,
                    0.62, (175, 175, 175), 2, cv2.LINE_AA)

    if note:
        (nw, _), _ = cv2.getTextSize(note, FONT, 0.7, 2)
        cv2.rectangle(img, (0, img.shape[0] - 46), (Wf, img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, note, ((Wf - nw) // 2, img.shape[0] - 16), FONT,
                    0.7, TEAL if hold else (0, 220, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-speed", type=float, default=9000.0,
                    help="speed-up far from any sighting")
    ap.add_argument("--outer-sec", type=float, default=1.0,
                    help="seconds of video for the OUTER ramp: cruise "
                         "(1 frame / 5 min) down to the stills' own 1 frame / 10 s")
    ap.add_argument("--inner-sec", type=float, default=1.0,
                    help="seconds of video for the INNER ramp: 1 frame / 10 s "
                         "down to real time, where the event clip takes over")
    ap.add_argument("--event-guard-sec", type=float, default=10.0,
                    help="stills this close to an event are skipped: the "
                         "vehicle is already in them, which gives the reveal "
                         "away during the approach")
    ap.add_argument("--cruise-sec", type=float, default=300.0,
                    help="wall-clock seconds per frame while cruising")
    ap.add_argument("--gap-skip-min", type=float, default=60.0,
                    help="jump over holes in collection longer than this")
    ap.add_argument("--hold-sec", type=float, default=1.0,
                    help="freeze on the last still before each event")
    ap.add_argument("--edge-pad-sec", type=float, default=0.5,
                    help="seconds of clip kept outside the hand-marked L/R "
                         "edges, so the vehicle is not cut off mid-entry")
    ap.add_argument("--center-hold-sec", type=float, default=1.0,
                    help="how long to dwell on the hand-marked Center frame, "
                         "the best view of the vehicle")
    ap.add_argument("--ignore-marks", action="store_true",
                    help="play each event clip whole, as before the clips "
                         "were hand-marked")
    ap.add_argument("--mask-background", action="store_true",
                    help="black out the far sidewalk and buildings, keeping "
                         "vehicles and people (instance segmentation + the "
                         "fitted curb line)")
    ap.add_argument("--seg-model", default="yolo11m-seg.pt")
    ap.add_argument("--keep-people", action="store_true",
                    help="do NOT mask people (default is to black them out "
                         "everywhere in the frame, not just in the background)")
    ap.add_argument("--people-dilate", type=int, default=9,
                    help="pixels to grow each person mask by, so segmentation "
                         "error does not leave a readable fringe")
    ap.add_argument("--bg-margin", type=float, default=18.0,
                    help="pixels below the curb line still treated as background")
    ap.add_argument("--skip-before", default="",
                    help="ignore stills earlier than this ISO timestamp "
                         "(empty: use everything on disk)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / args.root
    cfg = load_config()

    stills = []
    for p in glob.glob(str(root / "2???-??-??" / "session_*" / "timelapse" / "*.jpg")):
        s = stamp_of(p)
        if s:
            stills.append((s.timestamp(), p))
    stills.sort()
    if args.skip_before:
        cut = dt.datetime.fromisoformat(args.skip_before).timestamp()
        before = len(stills)
        stills = [(t, p) for t, p in stills if t >= cut]
        if before != len(stills):
            print(f"skipping {before - len(stills)} still(s) from before "
                  f"{args.skip_before} (camera re-aimed; different crop)")
    if not stills:
        print("no timelapse frames")
        return 1
    st_ts = [t for t, _ in stills]
    # the stills' own resolution is the floor: going slower just freezes
    gaps = sorted(b - a for a, b in zip(st_ts, st_ts[1:]) if 0 < b - a < 120)
    native = gaps[len(gaps) // 2] if gaps else 10.0
    floor_speed = native * args.fps
    print(f"{len(stills)} stills, native spacing {native:.0f}s "
          f"-> floor speed {floor_speed:.0f}x")

    store = LabelStore.load()
    evs = []
    for r in store.decisions.values():
        if r["label"] != POLICE:
            continue
        sess = root / r["session"]
        ef = sess / "logs" / "events.jsonl"
        if not ef.exists():
            continue
        for line in ef.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if r["track_id"] in e.get("track_ids", []):
                clip = sess / "events" / f"{e['id']}.mp4"
                if clip.exists():
                    evs.append({"start": e["start_ts"], "end": e["end_ts"],
                                "clip": clip, "id": e["id"],
                                "tracks": {r["track_id"]}, "sess": r["session"]})
                break
    merged = {}
    for e in evs:
        k = (e["sess"], e["id"])
        if k in merged:
            merged[k]["tracks"] |= e["tracks"]
        else:
            merged[k] = e
    evs = sorted(merged.values(), key=lambda e: e["start"])

    # Hand-marked moments (tools/moment_marker.py). These are the only
    # trustworthy in/out points: an event clip opens with ring-buffer pre-roll
    # of unknown length, so frame 0 sits some unknown distance before the
    # event's recorded start and timestamp arithmetic cannot locate the entry.
    mp = repo / "labels" / "moments.json"   # not root.parent: --root may move
    marks = json.loads(mp.read_text()) if mp.exists() else {}
    n_marked = 0
    for e in evs:
        m = marks.get(f"{e['sess']}|{e['id']}", {}).get("marks")
        if m and not args.ignore_marks and {"left", "center", "right"} <= set(m):
            e["marks"] = m
            n_marked += 1
    print(f"{len(evs)} confirmed sightings to slow into "
          f"({n_marked} hand-marked, {len(evs) - n_marked} played whole)")

    # Two ramps, both measured in output frames, with the breakpoint at the
    # stills' own resolution. Working in seconds-per-frame rather than a speed
    # multiple keeps the two stages comparable: the outer ramp closes the gap
    # between the cruise sampling and what the archive actually holds; the
    # inner ramp closes the gap between that and real time, at which point the
    # event clip -- real footage -- takes over.
    real_spf = 1.0 / args.fps

    def geo(a, b, n):
        n = max(2, int(n))
        return [a * (b / a) ** (i / (n - 1)) for i in range(n)]

    outer = geo(args.cruise_sec, native, args.outer_sec * args.fps)
    inner = geo(native, real_spf, args.inner_sec * args.fps)
    outer_span = sum(outer)
    inner_span = sum(inner)
    ramp_span = outer_span + inner_span
    print(f"outer ramp: {len(outer)} frames, 1/{args.cruise_sec:.0f}s -> 1/{native:.0f}s "
          f"per frame, covering {outer_span/60:.1f} min")
    print(f"inner ramp: {len(inner)} frames, 1/{native:.0f}s -> real time, "
          f"covering {inner_span:.0f} s")

    out = Path(args.out) if args.out else root / "highlights" / "story_police.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = EventVideoWriter(out, (W, H), args.fps, quality=23)
    if not writer.ok:
        print("could not start writer")
        return 2

    def still_at(t, avoid=None):
        """Nearest still to `t`, skipping any inside the window `avoid`.

        A still taken while the vehicle is in frame shows it -- verified on
        2026-10-30, where the frame 5 s after the trigger has the Rogue
        clearly entering. Showing that during the approach gives away the
        reveal, so the approach holds the last clean frame instead.
        """
        i = bisect_left(st_ts, t)
        cands = [j for j in (i - 1, i, i + 1) if 0 <= j < len(stills)]
        if avoid:
            lo, hi = avoid
            clean = [j for j in range(len(stills)) if not (lo <= st_ts[j] <= hi)]
            if clean:
                j = min(clean, key=lambda k: abs(st_ts[k] - t))
                return stills[j][1]
        if not cands:
            return stills[0][1]
        j = min(cands, key=lambda k: abs(st_ts[k] - t))
        return stills[j][1]

    # ---- background removal -------------------------------------------
    # The far sidewalk and the buildings behind it never carry information:
    # everything the project cares about happens on the road. Blacking them
    # out is a region test (above the fitted curb line) minus an instance
    # test (vehicles and people), so a car parked against the far curb -- whose
    # body sits well above that line -- survives intact.
    segmenter = None
    bg_region = None
    mask_cache: dict[str, tuple] = {}
    VEHICLE_CLASSES = {1, 2, 3, 5, 7}
    PERSON_CLASS = 0

    if args.mask_background:
        import yaml
        from ultralytics import YOLO
        # Prefer a hand-drawn line (tools/mask_editor.py) over the collector's
        # auto-fitted curb: the fit marks where wheels meet the road, which is
        # not always where you want the mask to begin.
        hand = repo / "config" / "mask_line.yaml"
        margin = args.bg_margin
        if hand.exists():
            doc = yaml.safe_load(hand.read_text())
            fw, fh = doc.get("frame_size", [W, H])
            pts = [[p[0] * W / fw, p[1] * H / fh] for p in doc["points"]]
            margin = float(doc.get("margin", args.bg_margin))
            src = "hand-drawn (config/mask_line.yaml)"
        else:
            doc_p = repo / "config" / "curb.yaml"
            if not doc_p.exists():
                print("no config/mask_line.yaml or curb.yaml -- run "
                      "tools/mask_editor.py to draw the boundary")
                return 1
            doc = yaml.safe_load(doc_p.read_text())
            far = next((c for c in doc["curbs"] if c["name"] == "far"), None)
            if far is None:
                print("curb.yaml has no 'far' line")
                return 1
            fw, fh = doc.get("frame_size", [W, H])
            pts = [[p[0] * W / fw, p[1] * H / fh] for p in far["points"]]
            src = "auto-fitted curb (config/curb.yaml)"

        pts = sorted(pts, key=lambda q: q[0])
        poly = [[0, 0], [0, pts[0][1] + margin]]
        poly += [[x, y + margin] for x, y in pts]
        poly += [[W, pts[-1][1] + margin], [W, 0]]
        bg_region = np.zeros((H, W), np.uint8)
        cv2.fillPoly(bg_region, [np.array(poly, np.int32)], 1)
        segmenter = YOLO(args.seg_model)
        print(f"background masking on, boundary from {src}: "
              f"{len(pts)} points, margin {margin:.0f}px, region is "
              f"{100*bg_region.mean():.0f}% of frame")

    def strip_background(img, cache_key=None):
        """Black out the background, keep vehicles, remove people entirely.

        Two separate rules, deliberately:
          * background  -- a region test (above the boundary) minus the vehicle
                           masks, so a car parked against the far curb survives
                           even though its body is inside that region.
          * people      -- unconditional, anywhere in the frame including the
                           road. A pedestrian on the near pavement is as
                           identifiable as one on the far one, so the region
                           test has nothing to do with it.

        Person masks are dilated: an instance mask is predicted at low
        resolution and upsampled, so its edge can leave a readable fringe.
        Growing it costs a few background pixels and removes that risk.
        """
        if segmenter is None:
            return img
        if cache_key is not None and cache_key in mask_cache:
            keep, people = mask_cache[cache_key]
        else:
            r = segmenter.predict(img, imgsz=960, conf=0.25, verbose=False,
                                  device=cfg.detector.device)[0]
            keep = np.zeros((H, W), np.uint8)
            people = np.zeros((H, W), np.uint8)
            if r.masks is not None:
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for mk, c in zip(r.masks.data.cpu().numpy(), clss):
                    m = cv2.resize(mk.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                    if c in VEHICLE_CLASSES:
                        keep |= m
                    elif c == PERSON_CLASS:
                        people |= m
            if args.people_dilate > 0 and people.any():
                k = int(args.people_dilate) | 1
                people = cv2.dilate(people, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k, k)))
            if cache_key is not None:
                mask_cache[cache_key] = (keep, people)
        img[(bg_region == 1) & (keep == 0)] = 0
        if not args.keep_people:
            img[people == 1] = 0
        return img

    def load(path):
        img = cv2.imread(path)
        if img is None:
            return None
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return strip_background(img, cache_key=path)

    t = st_ts[0]
    end = st_ts[-1]
    written = 0
    ev_i = 0

    def emit(img, when, speed, note="", hold=False):
        nonlocal written
        writer.write(overlay(img.copy(), when, speed, note, hold))
        written += 1

    while t < end:
        i = bisect_left(st_ts, t)
        if i < len(st_ts) and st_ts[i] - t > args.gap_skip_min * 60:
            t = st_ts[i]
            continue

        nxt = evs[ev_i]["start"] if ev_i < len(evs) else None

        # --- decelerate, then the event, then accelerate ------------------
        if nxt is not None and t >= nxt - ramp_span:
            e = evs[ev_i]
            when0 = dt.datetime.fromtimestamp(e["start"])
            ids = ", ".join(f"#{x}" for x in sorted(e["tracks"]))

            guard = (e["start"] - args.event_guard_sec,
                     e["end"] + args.event_guard_sec)
            target = e["start"] - args.event_guard_sec
            tt = max(t, target - ramp_span)
            for spf in outer:                            # 5 min -> 10 s per frame
                img = load(still_at(tt, guard))
                if img is not None:
                    emit(img, dt.datetime.fromtimestamp(tt), spf * args.fps)
                tt += spf
            for spf in inner:                            # 10 s -> real time
                img = load(still_at(tt, guard))
                if img is not None:
                    emit(img, dt.datetime.fromtimestamp(tt), spf * args.fps)
                tt += spf
            if args.hold_sec:
                img = load(still_at(target, guard))
                if img is not None:
                    for _ in range(int(args.hold_sec * args.fps)):
                        emit(img, when0, 1.0, f"PARKING POLICE   {ids}")

            cap = cv2.VideoCapture(str(e["clip"]))
            clip_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
            repeat = max(1, int(round(args.fps / clip_fps)))

            # Hand-marked clips are cut to the transit itself. L and R are
            # frame positions, not an order -- the vehicle may cross either
            # way -- so the window is min..max of the two, padded.
            mk = e.get("marks")
            if mk:
                pad = int(round(args.edge_pad_sec * clip_fps))
                lo = max(0, min(mk["left"], mk["right"]) - pad)
                hi = max(mk["left"], mk["right"]) + pad
                ctr = mk["center"]
                dwell = max(1, int(round(args.center_hold_sec * args.fps)))
            else:
                lo, hi, ctr, dwell = 0, 10 ** 9, -1, 0

            k = kept = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if k < lo:
                    k += 1
                    continue
                if k > hi:
                    break
                if fr.shape[1] != W or fr.shape[0] != H:
                    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                fr = strip_background(fr)        # clip frames are all distinct
                ft = dt.datetime.fromtimestamp(e["start"] + k / clip_fps)
                tag = f"PARKING POLICE   {ids}   {e['id']}"
                is_hold = (k == ctr)
                for _ in range(dwell if is_hold else repeat):
                    emit(fr, ft, 1.0, tag, hold=is_hold)
                kept += 1
                k += 1
            cap.release()

            tt = e["end"] + args.event_guard_sec
            for spf in reversed(inner):                  # real time -> 10 s
                img = load(still_at(tt, guard))
                if img is not None:
                    emit(img, dt.datetime.fromtimestamp(tt), spf * args.fps)
                tt += spf
            for spf in reversed(outer):                  # 10 s -> 5 min
                img = load(still_at(tt, guard))
                if img is not None:
                    emit(img, dt.datetime.fromtimestamp(tt), spf * args.fps)
                tt += spf
            how = (f"frames {lo}-{min(hi, k-1)} of {k}, hold on {ctr}"
                   if e.get("marks") else f"{k} frames, whole clip")
            print(f"  [{ev_i+1}/{len(evs)}] {when0:%Y-%m-%d %H:%M:%S} {ids} "
                  f"({how})")
            t = tt
            ev_i += 1
            continue

        # --- cruise --------------------------------------------------------
        img = load(still_at(t))
        if img is not None:
            emit(img, dt.datetime.fromtimestamp(t), args.cruise_sec * args.fps)
            if written % 1000 == 0:
                print(f"  {written} frames, at {dt.datetime.fromtimestamp(t):%Y-%m-%d %H:%M}")
        t += args.cruise_sec

    stats = writer.close()
    print(f"\nwrote {out}")
    print(f"  {written} frames, {written/args.fps:.0f}s, "
          f"{stats['bytes']/1e6:.1f} MB, {stats['encoder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
