"""On-disk layout for a collection session.

    data/<YYYY-MM-DD>/session_<HHMMSS>/
        meta.json               session config, host, camera, final counts
        timelapse/              periodic full frames  (parking occupancy over time)
        events/evt_000123/      burst frames around an arrival/departure/pass-by
        crops/vehicle/          per-track vehicle cutouts (phase-2 training set)
        crops/person/
        candidates/police/      copies of crops the colour triage flagged
        logs/*.jsonl            detections, occupancy, events, crops
        collect.log

Frames are JPEG, named `YYYYmmdd_HHMMSS_mmm.jpg`, so lexical order is time
order and a filename alone identifies the moment it came from.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from .video import EventVideoWriter, encoder_name

log = logging.getLogger(__name__)


def ts_name(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y%m%d_%H%M%S_") + f"{int(dt.microsecond / 1000):03d}"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


class SessionStore:
    def __init__(self, cfg: Any, start_ts: float | None = None,
                 root: str | Path | None = None, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.start_ts = start_ts if start_ts is not None else time.time()
        dt = datetime.fromtimestamp(self.start_ts)

        base = Path(root) if root else Path(cfg.storage.root)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent.parent / base
        self.root = base
        self.day = dt.strftime("%Y-%m-%d")
        self.session_id = dt.strftime("session_%H%M%S")
        self.dir = self.root / self.day / self.session_id

        self.timelapse_dir = self.dir / "timelapse"
        self.events_dir = self.dir / "events"
        self.crops_dir = self.dir / "crops"
        self.candidates_dir = self.dir / "candidates" / "police"
        self.logs_dir = self.dir / "logs"

        self.counts: dict[str, int] = {
            "timelapse": 0, "event_frames": 0, "events": 0,
            "crops": 0, "candidates": 0, "detect_samples": 0,
            "bytes_written": 0,
        }
        self._files: dict[str, TextIO] = {}
        self._event_seq = 0
        self._last_disk_check = 0.0
        self._disk_ok = True
        self._free_gb = 0.0

        if not dry_run:
            for d in (self.timelapse_dir, self.events_dir,
                      self.crops_dir / "vehicle", self.crops_dir / "person",
                      self.candidates_dir, self.logs_dir):
                d.mkdir(parents=True, exist_ok=True)
        log.info("session dir: %s%s", self.dir, "  (DRY RUN — nothing written)"
                 if dry_run else "")

    # --------------------------------------------------------------- logs

    def _logfile(self, name: str) -> TextIO | None:
        if self.dry_run:
            return None
        if name not in self._files:
            self._files[name] = open(self.logs_dir / f"{name}.jsonl", "a",
                                     encoding="utf-8")
        return self._files[name]

    def log_json(self, name: str, obj: dict) -> None:
        fh = self._logfile(name)
        if fh is None:
            return
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        # Flushed per record: this process is expected to be killed abruptly
        # (SIGTERM, power cut) and a truncated tail loses the interesting part.
        fh.flush()

    # ------------------------------------------------------------- frames

    def _imwrite(self, path: Path, img: np.ndarray, quality: int) -> bool:
        if self.dry_run:
            return True
        ok = cv2.imwrite(str(path), img,
                         [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            try:
                self.counts["bytes_written"] += path.stat().st_size
            except OSError:
                pass
        else:
            log.error("failed to write %s", path)
        return ok

    def save_timelapse(self, ts: float, frame: np.ndarray) -> Path | None:
        path = self.timelapse_dir / f"{ts_name(ts)}.jpg"
        if self._imwrite(path, frame, self.cfg.storage.jpeg_quality):
            self.counts["timelapse"] += 1
            return path
        return None

    # ------------------------------------------------------------- events

    def new_event(self, ts: float, reason: str, size: tuple[int, int],
                  fps: float, quality: int = 24) -> dict:
        """Open an event. Frames go to one H.264 clip; JPEGs only as fallback."""
        self._event_seq += 1
        eid = f"evt_{self._event_seq:06d}"
        event = {
            "id": eid, "start_ts": ts, "reason": reason, "frames": 0,
            "track_ids": set(), "classes": set(), "flags": set(),
            "writer": None, "dir": None,
        }
        if self.dry_run:
            self.counts["events"] += 1
            return event

        if encoder_name() != "none":
            event["path"] = self.events_dir / f"{eid}.mp4"
            event["writer"] = EventVideoWriter(event["path"], size, fps, quality)
            if not event["writer"].ok:
                log.warning("video encoder unavailable for %s — using JPEGs", eid)
                event["writer"] = None
        if event["writer"] is None:
            event["dir"] = self.events_dir / eid / "frames"
            event["dir"].mkdir(parents=True, exist_ok=True)
        self.counts["events"] += 1
        return event

    def save_event_frame(self, event: dict, ts: float, frame) -> bool:
        if self.dry_run:
            event["frames"] += 1
            self.counts["event_frames"] += 1
            return True
        if event["writer"] is not None:
            if not event["writer"].write(frame):
                return False
        else:
            path = event["dir"] / f"{ts_name(ts)}.jpg"
            if not self._imwrite(path, frame, self.cfg.storage.jpeg_quality):
                return False
        event["frames"] += 1
        self.counts["event_frames"] += 1
        return True

    def close_event(self, event: dict, end_ts: float) -> dict:
        stats = {"frames": event["frames"], "bytes": 0, "encoder": "jpeg"}
        if event["writer"] is not None:
            stats = event["writer"].close()
            self.counts["bytes_written"] += stats["bytes"]
        rec = {
            "id": event["id"],
            "reason": event["reason"],
            "start": iso(event["start_ts"]),
            "end": iso(end_ts),
            "start_ts": round(event["start_ts"], 3),
            "end_ts": round(end_ts, 3),
            "duration_sec": round(end_ts - event["start_ts"], 2),
            "frames": event["frames"],
            "encoder": stats["encoder"],
            "bytes": stats["bytes"],
            "flags": sorted(event["flags"]),
            "track_ids": sorted(event["track_ids"]),
            "classes": sorted(event["classes"]),
        }
        if event.get("path") is not None:
            rec["file"] = str(event["path"].relative_to(self.dir))
        self.log_json("events", rec)
        return rec

    # -------------------------------------------------------------- crops

    def save_crop(self, ts: float, det, crop: np.ndarray, kind: str,
                  seq: int, triage=None) -> Path | None:
        tid = det.track_id if det.track_id is not None else -1
        stem = f"trk{tid:06d}_{seq:02d}_{det.name}_{ts_name(ts)}"
        path = self.crops_dir / kind / f"{stem}.jpg"
        if not self._imwrite(path, crop, self.cfg.capture.crops.jpeg_quality):
            return None
        self.counts["crops"] += 1

        rec = {
            "t": iso(ts), "ts": round(ts, 3),
            "file": str(path.relative_to(self.dir)),
            "kind": kind, "seq": seq,
            "id": det.track_id, "c": det.name, "ci": det.cls_id,
            "p": round(float(det.conf), 3), "b": [int(v) for v in det.xyxy],
            # A box touching the frame border is a truncated vehicle. Kept (it
            # still carries colour evidence) but flagged, so phase-2 training
            # can exclude partial views.
            "edge": bool(det.extra.get("edge", False)),
        }
        if triage is not None:
            rec["triage"] = triage.to_json()
            if triage.is_police_candidate:
                cand = self.candidates_dir / f"{stem}.jpg"
                if not self.dry_run:
                    try:
                        shutil.copy2(path, cand)
                        self.counts["candidates"] += 1
                        rec["candidate_file"] = str(cand.relative_to(self.dir))
                    except OSError as exc:
                        log.error("candidate copy failed: %s", exc)
        self.log_json("crops", rec)
        return path

    # --------------------------------------------------------------- disk

    def free_gb(self) -> float:
        try:
            usage = shutil.disk_usage(self.root if self.root.exists()
                                      else self.root.parent)
            return usage.free / (1024 ** 3)
        except OSError:
            return float("inf")

    def disk_ok(self, force: bool = False) -> bool:
        """Cached free-space check; re-stats at most every 30 s."""
        now = time.time()
        if force or now - self._last_disk_check > 30.0:
            self._last_disk_check = now
            self._free_gb = self.free_gb()
            ok = self._free_gb >= float(self.cfg.storage.min_free_gb)
            if self._disk_ok and not ok:
                log.error("free space %.1f GB below min_free_gb=%.1f — "
                          "pausing writes", self._free_gb,
                          float(self.cfg.storage.min_free_gb))
            elif ok and not self._disk_ok:
                log.info("free space recovered (%.1f GB) — resuming writes",
                         self._free_gb)
            self._disk_ok = ok
        return self._disk_ok

    def prune(self, retention_days: int) -> int:
        """Delete whole day directories older than `retention_days`.

        Never called automatically unless retention_days > 0 AND the disk is
        below min_free_gb. Returns the number of day directories removed.
        """
        if retention_days <= 0 or self.dry_run:
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for day_dir in sorted(self.root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")):
            if not day_dir.is_dir() or day_dir.name == self.day:
                continue
            try:
                when = datetime.strptime(day_dir.name, "%Y-%m-%d").timestamp()
            except ValueError:
                continue
            if when < cutoff:
                log.warning("pruning %s (older than %d days)", day_dir, retention_days)
                shutil.rmtree(day_dir, ignore_errors=True)
                removed += 1
        return removed

    # ----------------------------------------------------------- lifecycle

    def write_meta(self, extra: dict | None = None, final: bool = False) -> None:
        meta = {
            "site": dict(self.cfg.site),
            "session_id": self.session_id,
            "day": self.day,
            "start": iso(self.start_ts),
            "start_ts": round(self.start_ts, 3),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "camera": dict(self.cfg.camera),
            "detector": dict(self.cfg.detector),
            "capture": json.loads(json.dumps(self.cfg.capture)),
            "triage": dict(self.cfg.triage),
            "counts": dict(self.counts),
            "final": final,
        }
        if final:
            meta["end"] = iso(time.time())
            meta["duration_sec"] = round(time.time() - self.start_ts, 1)
        if extra:
            meta.update(extra)
        if not self.dry_run:
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2,
                                                           default=str))

    def close(self, extra: dict | None = None) -> None:
        self.write_meta(extra, final=True)
        for fh in self._files.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        self._files.clear()
        log.info("session closed: %s", self.summary_line())

    def summary_line(self) -> str:
        c = self.counts
        mb = c["bytes_written"] / (1024 ** 2)
        return (f"timelapse={c['timelapse']} events={c['events']} "
                f"event_frames={c['event_frames']} crops={c['crops']} "
                f"candidates={c['candidates']} samples={c['detect_samples']} "
                f"written={mb:.1f}MB")
