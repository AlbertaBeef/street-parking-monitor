#!/usr/bin/env python3
"""Street parking monitor — phase 1 data collector.

Records four complementary streams, each answering a different question:

  timelapse   one full frame every `timelapse_sec`. The occupancy record — a
              parked car is dull frame to frame, but its arrival and departure
              times are the whole parking-space signal.

  events      an H.264 clip whenever a vehicle is actually moving or a new one
              appears: arrivals, departures, pass-bys. Includes pre-roll from a
              ring buffer, so the clip starts before the trigger fired.

  crops       per-track vehicle cutouts, rate-limited per track. The labelling
              pool for goal 2, the enforcement vehicle.

  tracks      one summary record per finished track: path, direction, speed in
              body-lengths/sec, stops, dwell. This is what makes a sighting
              searchable after the fact, and it drives the kinematic triage.

Two independent triage channels flag candidates for review. `triage.py` asks
what a vehicle looks like (white body, blue markings); `tracks.py` asks what it
does (patrol crawl, mid-street stops). Either one alone produces too many false
positives to review by hand; agreement between them is a strong signal.

Usage
    python collect.py                     # run until Ctrl-C
    python collect.py --duration 3600     # one hour
    python collect.py --display           # with a live preview window
    python collect.py --dry-run --duration 60   # exercise it all, write nothing
    python collect.py --ignore-schedule   # collect at full rate regardless
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streetmon.camera import CameraReader
from streetmon.config import (curb_calib_path, load_config, load_curb_doc,
                               save_curb_doc)
from streetmon.detector import Detector, draw_detections
from streetmon.schedule import Schedule
from streetmon.storage import SessionStore, iso
from streetmon.tracks import TrackRegistry
from streetmon.triage import analyse as triage_analyse
from streetmon.video import encoder_name
from streetmon.curb import CurbMonitor, autofit_curbs

log = logging.getLogger("collect")

_stop = False


def _handle_signal(signum, _frame):
    global _stop
    log.info("signal %s received — shutting down", signal.Signals(signum).name)
    _stop = True


def setup_console_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def attach_file_log(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def build_curb_monitor(cfg, frame_size) -> CurbMonitor | None:
    """Load curb lines from config, or return None so the caller can auto-fit."""
    doc = load_curb_doc()
    defs = doc.get("curbs") or []
    if not defs:
        return None
    src = doc.get("frame_size")
    return CurbMonitor(defs, frame_size, cfg.curb,
                       vehicle_classes=cfg.detector.vehicle_classes,
                       source_frame_size=tuple(src) if src else None,
                       calib_path=curb_calib_path())


def try_autofit_curbs(cfg, dets, frame_size) -> CurbMonitor | None:
    """Fit curb lines from the parked cars in view and persist them."""
    defs = autofit_curbs(dets, frame_size,
                         set(cfg.detector.vehicle_classes))
    if not defs:
        return None
    save_curb_doc({"frame_size": list(frame_size), "curbs": defs})
    for d in defs:
        log.info("auto-fitted curb '%s' from %d vehicle(s), max residual %.1f px",
                 d["name"], d["fitted_from"], d["max_residual_px"])
    log.info("wrote config/curb.yaml — adjust with tools/curb_editor.py, "
             "or delete it to re-fit")
    return build_curb_monitor(cfg, frame_size)


def crop_box(frame, xyxy, pad: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def draw_hud(frame, lines, origin=(10, 10)):
    x, y = origin
    for i, text in enumerate(lines):
        yy = y + 22 + i * 24
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x - 4, yy - th - 6), (x + tw + 6, yy + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run; 0 = until interrupted")
    ap.add_argument("--display", action="store_true", help="live preview window")
    ap.add_argument("--no-detect", action="store_true",
                    help="timelapse only — no model, events or crops")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    ap.add_argument("--ignore-schedule", action="store_true",
                    help="collect at full rate regardless of active hours")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--status-sec", type=float, default=60.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_console_logging(args.verbose)
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.camera.device = args.device

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ev, crops_cfg = cfg.capture.event, cfg.capture.crops
    schedule = Schedule(cfg.schedule)
    if args.ignore_schedule:
        schedule.enabled = False
        log.info("--ignore-schedule: active-hours gating off")

    detect_interval = 1.0 / max(float(cfg.capture.detect_fps), 0.01)
    event_interval = 1.0 / max(float(ev.fps), 0.01)

    # -------------------------------------------------------------- camera
    cam = CameraReader(cfg.camera,
                       ring_seconds=float(ev.pre_roll_sec) if ev.enabled else 0.0,
                       ring_fps=float(ev.fps))
    try:
        cam.start()
    except Exception as exc:
        log.error("camera failed to start: %s", exc)
        return 2
    if not cam.wait_first_frame(10.0):
        log.error("no frames from camera within 10 s")
        cam.stop()
        return 2
    frame_size = cam.actual_size()
    # Pixel thresholds in the config are written for a 1920-wide frame, so that
    # switching capture resolution does not silently change their meaning.
    px_scale = frame_size[0] / 1920.0
    min_box_px = int(float(crops_cfg.min_box_px) * px_scale)
    min_box_px_person = int(float(crops_cfg.min_box_px_person) * px_scale)
    tl_width = int(cfg.capture.timelapse_width)
    # Events are written at their own width: 4K is worth it for still crops
    # where fleet unit numbers might be legible, but not for the video.
    ev_width = int(ev.width) or frame_size[0]
    event_size = (ev_width, int(frame_size[1] * ev_width / frame_size[0]))
    log.info("capturing at %dx%d | event encoder: %s | timelapse %dpx | "
             "px thresholds x%.2f", *frame_size, encoder_name(), tl_width,
             px_scale)
    log.info("event clips at %dx%d", *event_size)

    # ------------------------------------------------------- detector/curb
    detector = None
    if not args.no_detect:
        try:
            detector = Detector(cfg.detector)
            # The first inference compiles ROCm kernels and can take ~15 s. Get
            # it out of the way now so it does not stall the first real sample
            # and skew every timestamp downstream.
            t_warm = time.time()
            _, warm = cam.latest()
            if warm is not None:
                detector.track(warm)
                detector.reset_tracker()
                detector.infer_ms_ema = None
            log.info("detector warmed up in %.1fs", time.time() - t_warm)
        except Exception as exc:
            log.error("detector failed to load: %s", exc)
            cam.stop()
            return 3
    curbs = build_curb_monitor(cfg, frame_size) if detector else None
    if detector and curbs is None:
        log.info("no curb lines yet — will fit them from the first parked cars")
    registry = TrackRegistry(cfg, expire_sec=float(cfg.kinematic.expire_sec))
    curb_states: list = []
    prev_parkable: dict[str, int] = {}
    pending_gap: dict[str, tuple] = {}
    next_curb_log = 0.0
    next_calib_save = 0.0

    # --------------------------------------------------------------- store
    store = SessionStore(cfg, root=args.root, dry_run=args.dry_run)
    file_handler = None
    if not args.dry_run:
        file_handler = attach_file_log(store.dir / "collect.log")
    store.write_meta({"frame_size": list(frame_size), "encoder": encoder_name()})

    event = None
    event_last_frame_ts = 0.0
    event_last_activity_ts = 0.0

    start = time.time()
    # Vehicles already parked at startup are not arrivals; suppress the
    # new-vehicle trigger until the standing scene has been enrolled.
    new_vehicle_grace_until = start + 5.0
    next_detect = start
    next_timelapse = start
    next_status = start + args.status_sec
    last_dets: list = []
    display_ok = bool(args.display)
    flagged_ids: set[int] = set()
    seen_vehicle_ids: set[int] = set()
    sample_count = 0
    colour_hits = 0
    last_reconnects = cam.reconnects
    was_active = None

    log.info("collecting — Ctrl-C to stop%s",
             f" (duration {args.duration:.0f}s)" if args.duration else "")

    try:
        while not _stop:
            now = time.time()
            if args.duration and now - start >= args.duration:
                log.info("duration reached")
                break

            wall = datetime.fromtimestamp(now)
            active = schedule.is_active(wall)
            if active != was_active:
                log.info("schedule: now %s", "ACTIVE" if active else "idle")
                was_active = active
            timelapse_sec = (float(cfg.capture.timelapse_sec) if active
                             else float(schedule.idle.timelapse_sec))
            events_on = ev.enabled and (active or bool(schedule.idle.events))
            crops_on = crops_cfg.enabled and (active or bool(schedule.idle.crops))

            # --- daily rotation -------------------------------------------
            if cfg.storage.rotate_daily and not args.dry_run:
                today = wall.strftime("%Y-%m-%d")
                if today != store.day:
                    log.info("date rolled to %s — new session", today)
                    if event is not None:
                        store.close_event(event, now)
                        event = None
                    store.close({"frame_size": list(frame_size)})
                    if file_handler:
                        logging.getLogger().removeHandler(file_handler)
                        file_handler.close()
                    store = SessionStore(cfg, start_ts=now, root=args.root)
                    file_handler = attach_file_log(store.dir / "collect.log")
                    store.write_meta({"frame_size": list(frame_size)})

            if cam.reconnects != last_reconnects:
                last_reconnects = cam.reconnects
                if detector:
                    detector.reset_tracker()
                for rec in registry.flush(frame_size):
                    store.log_json("tracks", rec)

            ts, frame = cam.latest()
            if frame is None:
                time.sleep(0.05)
                continue

            writable = store.disk_ok()
            if not writable and cfg.storage.retention_days:
                store.prune(int(cfg.storage.retention_days))
                writable = store.disk_ok(force=True)

            # --- detection sample -----------------------------------------
            ran_detection = False
            if detector and now >= next_detect:
                next_detect = max(now + detect_interval, next_detect + detect_interval)
                ran_detection = True
                sample_count += 1
                store.counts["detect_samples"] = sample_count

                dets = detector.track(frame)
                last_dets = dets
                new_vehicle = False
                any_moving = False
                any_person = False

                for d in dets:
                    if d.track_id is None:
                        continue
                    is_new = d.track_id not in registry.live
                    tr = registry.observe(d, ts, frame_size)
                    if is_new and detector.is_vehicle(d):
                        if (d.track_id not in seen_vehicle_ids
                                and now >= new_vehicle_grace_until):
                            new_vehicle = True
                        seen_vehicle_ids.add(d.track_id)
                    # Event clips use the strict test; crops and curb
                    # occupancy below keep using the sensitive `tr.moving`.
                    if tr.moving_confirmed and detector.is_vehicle(d):
                        any_moving = True
                    elif tr.moving and d.cls_id == 0 and ev.trigger_on_person:
                        # An enforcement officer inspects on foot as well as
                        # from the car, and someone walking to a parked vehicle
                        # is the best early warning that a gap is about to open.
                        any_person = True
                    if curbs is not None:
                        a = curbs.assign(d)
                        if a is not None:
                            tr.curbs_touched.add(a[0].name)

                    # --- crops --------------------------------------------
                    if not (crops_on and writable):
                        continue
                    is_vehicle = detector.is_vehicle(d)
                    kind = "vehicle" if is_vehicle else "person"
                    if kind == "person" and not crops_cfg.persons:
                        continue
                    bw, bh = d.wh
                    floor = min_box_px if is_vehicle else min_box_px_person
                    if min(bw, bh) < floor:
                        continue
                    if tr.crops_saved >= int(crops_cfg.max_per_track):
                        continue
                    # Rate-limit by novelty, not by clock. A parked car's crops
                    # are near-identical two seconds apart, so throttling them
                    # costs nothing. A vehicle crossing the frame gives a
                    # genuinely different view every sample — and may be gone in
                    # under a second. The confirmed enforcement sighting on
                    # 2026-09-18 transited in 0.73 s across 4 samples, and the
                    # flat 2 s interval kept only the first of them.
                    interval = (float(crops_cfg.min_interval_sec_moving)
                                if tr.moving else float(crops_cfg.min_interval_sec))
                    if ts - tr.last_crop_ts < interval:
                        continue

                    crop = crop_box(frame, d.xyxy, float(crops_cfg.pad))
                    if crop is None:
                        continue
                    fh_, fw_ = frame.shape[:2]
                    d.extra["edge"] = (d.xyxy[0] <= 2 or d.xyxy[1] <= 2
                                       or d.xyxy[2] >= fw_ - 2 or d.xyxy[3] >= fh_ - 2)
                    tri = None
                    if cfg.triage.enabled and is_vehicle:
                        tri = triage_analyse(crop, cfg.triage)
                        tr.best_white = max(tr.best_white, tri.white_frac)
                        tr.best_red = max(tr.best_red, tri.red_band_frac)
                        if tri.is_police_candidate and not tr.colour_candidate:
                            colour_hits += 1
                            log.warning("COLOUR CANDIDATE track #%d (%s) "
                                        "white=%.2f red_band=%.3f", d.track_id,
                                        d.name, tri.white_frac,
                                        tri.red_band_frac)
                            tr.colour_candidate = True
                            flagged_ids.add(d.track_id)

                    tr.crops_saved += 1
                    tr.last_crop_ts = ts
                    path = store.save_crop(ts, d, crop, kind, tr.crops_saved, tri)
                    if path:
                        tr.crop_files.append(path.name)

                # --- curb occupancy ---------------------------------------
                if curbs is None:
                    curbs = try_autofit_curbs(cfg, dets, frame_size)

                curb_snap = {}
                if curbs is not None:
                    # Only stationary vehicles occupy curb space. A car driving
                    # past would otherwise be reported as parked and would
                    # briefly erase a real gap.
                    parked_ids = {tid for tid, t in registry.live.items()
                                  if not t.moving}
                    for d in dets:
                        curbs.observe_for_calibration(d)
                    curb_states = curbs.update(dets, parked_ids=parked_ids)

                    min_gap = float(cfg.curb.min_gap_m)
                    for st in curb_states:
                        usable = st.parkable(min_gap)
                        n = len(usable)
                        prev = prev_parkable.get(st.name)
                        if prev is None:
                            prev_parkable[st.name] = n
                            continue
                        if n == prev:
                            pending_gap.pop(st.name, None)
                            continue
                        # Confirm over elapsed time, not sample count. A parked
                        # car occluded by a passing van drops out for a second
                        # or two; a 3-sample (0.75 s) threshold reported that as
                        # an 8.2 m space opening and closing again. Real cars
                        # take far longer to arrive or leave, so the wait is free.
                        pend = pending_gap.get(st.name)
                        if pend is None or pend[0] != n:
                            pending_gap[st.name] = (n, now)
                            continue
                        if now - pend[1] < float(cfg.curb.gap_confirm_sec):
                            continue
                        pending_gap.pop(st.name, None)
                        opened = n > prev
                        prev_parkable[st.name] = n
                        biggest = max((g.length for g in usable), default=0.0)
                        log.info("curb %s: %d parkable gap(s) %s (largest %.1f m, "
                                 "%.1f m free of %.1f m)", st.name, n,
                                 "opened" if opened else "closed", biggest,
                                 st.free_m, st.length_m)
                        if writable:
                            store.log_json("curb_events", {
                                "t": iso(ts), "ts": round(ts, 3),
                                "curb": st.name,
                                "event": "gap_opened" if opened else "gap_closed",
                                "n_parkable": n, "largest_gap_m": round(biggest, 2),
                                "free_m": round(st.free_m, 2),
                                "length_m": round(st.length_m, 2),
                                "calibrated": st.calibrated,
                                "gaps": [g.to_json() for g in usable]})

                    curb_snap = {s.name: {"free_m": round(s.free_m, 1),
                                          "occ_m": round(s.occupied_m, 1),
                                          "n": len(s.intervals),
                                          "parkable": len(s.parkable(min_gap))}
                                 for s in curb_states}

                    if writable and now >= next_curb_log:
                        next_curb_log = now + float(cfg.curb.log_interval_sec)
                        store.log_json("curb", {
                            "t": iso(ts), "ts": round(ts, 3),
                            "curbs": [s.to_json() for s in curb_states]})

                    if now >= next_calib_save:
                        next_calib_save = now + 300.0
                        curbs.save_calibration()

                if writable and sample_count % int(cfg.capture.log_every_n) == 0:
                    store.log_json("detections", {
                        "t": iso(ts), "ts": round(ts, 3), "n": sample_count,
                        "dets": [d.to_json() for d in dets], "curb": curb_snap})

                # --- event trigger ----------------------------------------
                if events_on and writable:
                    trigger = (any_moving or any_person
                               or (new_vehicle and ev.trigger_on_new_vehicle))
                    if trigger:
                        event_last_activity_ts = now
                    if event is None and trigger:
                        reason = ("motion" if any_moving else
                                  "new_vehicle" if new_vehicle else "person")
                        event = store.new_event(ts, reason, event_size,
                                                float(ev.fps), int(ev.quality))
                        log.info("event %s started (%s)", event["id"], reason)
                        for pts, pframe in cam.ring_snapshot(float(ev.pre_roll_sec)):
                            store.save_event_frame(event, pts, pframe)
                        event_last_frame_ts = now
                    if event is not None:
                        for d in dets:
                            if d.track_id is not None:
                                event["track_ids"].add(d.track_id)
                                if d.track_id in flagged_ids:
                                    event["flags"].add("colour_candidate")
                            event["classes"].add(d.name)

            if ran_detection:
                # detector.track() can take a while; everything below is
                # time-sensitive, so re-read the clock instead of trusting `now`.
                now = time.time()

                # --- finished tracks ----------------------------------------
                for rec in registry.reap(now, frame_size):
                    if writable:
                        store.log_json("tracks", rec)
                    if rec.get("kinematic_candidate"):
                        log.warning(
                            "KINEMATIC CANDIDATE track #%s (%s) %s — %s, "
                            "%.2f BL/s over %.1f BL, %d stop(s)",
                            rec["id"], rec["c"], rec["direction"],
                            rec["kinematic_reason"], rec.get("median_speed_bl_s", 0),
                            rec.get("net_bl", 0), rec.get("stops", 0))
                        if event is not None:
                            event["flags"].add("kinematic_candidate")
                    flagged_ids.discard(rec["id"])

            # --- event frames / end ---------------------------------------
            if event is not None:
                if now - event_last_frame_ts >= event_interval and writable:
                    store.save_event_frame(event, ts, frame)
                    event_last_frame_ts = now
                elapsed = now - event["start_ts"]
                idle = now - event_last_activity_ts
                if idle >= float(ev.post_roll_sec) or elapsed >= float(ev.max_sec):
                    rec = store.close_event(event, now)
                    log.info("event %s ended (%.1fs, %d frames, %.2f MB, %s)",
                             event["id"], elapsed, rec["frames"],
                             rec["bytes"] / 1048576, ",".join(rec["flags"]) or "-")
                    event = None

            # --- timelapse -------------------------------------------------
            if now >= next_timelapse:
                next_timelapse = now + timelapse_sec
                if writable:
                    # Events keep full resolution; the timelapse does not need
                    # it and would cost 4x the disk for no extra information.
                    tl = frame
                    if tl_width and frame.shape[1] > tl_width:
                        tl = cv2.resize(
                            frame, (tl_width,
                                    int(frame.shape[0] * tl_width / frame.shape[1])),
                            interpolation=cv2.INTER_AREA)
                    store.save_timelapse(ts, tl)

            # --- preview ---------------------------------------------------
            # The window is a convenience; collection is not. Started at boot
            # by systemd there is no X session yet, so the first imshow fails.
            # Degrade to headless once and carry on rather than taking the
            # collector down with it.
            if args.display and display_ok:
                try:
                    view = frame.copy()
                    if curbs is not None and curb_states:
                        curbs.draw(view, curb_states)
                    if last_dets:
                        draw_detections(view, last_dets, flagged_ids)
                    curb_line = "  ".join(
                        f"{s.name}: {s.free_m:.1f}m free, {len(s.parkable(float(cfg.curb.min_gap_m)))} parkable"
                        for s in curb_states) or "fitting curb lines..."
                    draw_hud(view, [
                        f"{wall:%Y-%m-%d %H:%M:%S}  [{'ACTIVE' if active else 'idle'}]",
                        curb_line,
                        f"tracks {len(registry.live)} live / {registry.completed} done"
                        f"   event {event['id'] if event else '-'}",
                        f"candidates: colour {colour_hits} / kinematic "
                        f"{registry.kinematic_hits}",
                    ])
                    w = min(1280, view.shape[1])
                    cv2.imshow("street-parking-monitor",
                               cv2.resize(view, (w, int(view.shape[0] * w / view.shape[1]))))
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        log.info("quit key pressed")
                        break
                except cv2.error as exc:
                    display_ok = False
                    log.warning("preview unavailable (%s) — continuing headless",
                                str(exc).strip().splitlines()[0][:90])

            # --- status ----------------------------------------------------
            if now >= next_status:
                next_status = now + args.status_sec
                curb_txt = " ".join(
                    f"{s.name}={s.free_m:.1f}m/{len(s.parkable(float(cfg.curb.min_gap_m)))}p"
                    for s in curb_states) or "curb=unfitted"
                log.info(
                    "up %.0fs [%s] | cam %d frames (%d reconnects) | %s | "
                    "tracks %d live/%d done | cand colour=%d kin=%d | "
                    "%s | infer %.0fms | suppressed %d/%d | free %.0fGB%s",
                    now - start, "ACTIVE" if active else "idle", cam.frames_read,
                    cam.reconnects, store.summary_line(), len(registry.live),
                    registry.completed, colour_hits, registry.kinematic_hits,
                    curb_txt, (detector.infer_ms_ema or 0.0) if detector else 0.0,
                    detector.suppressed_nested if detector else 0,
                    detector.suppressed_aspect if detector else 0,
                    store.free_gb(), "" if writable else " | WRITES PAUSED")

            if not ran_detection:
                sleep = max(0.0, next_detect - time.time())
                if sleep > 0:
                    time.sleep(min(sleep, 0.05))

    except Exception:
        log.exception("collector crashed")
        return 1
    finally:
        if event is not None:
            store.close_event(event, time.time())
        for rec in registry.flush(frame_size):
            store.log_json("tracks", rec)
        if curbs is not None:
            curbs.save_calibration()
        store.close({"frame_size": list(frame_size),
                     "suppressed_nested": detector.suppressed_nested if detector else 0,
                     "suppressed_aspect": detector.suppressed_aspect if detector else 0,
                     "camera_frames_read": cam.frames_read,
                     "camera_reconnects": cam.reconnects,
                     "vehicle_tracks_seen": len(seen_vehicle_ids),
                     "tracks_completed": registry.completed,
                     "colour_candidates": colour_hits,
                     "kinematic_candidates": registry.kinematic_hits})
        cam.stop()
        if args.display:
            cv2.destroyAllWindows()
        log.info("done. data in %s", store.dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
