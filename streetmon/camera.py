"""Threaded V4L2 capture.

A background thread drains the V4L2 queue continuously so the "latest" frame is
always fresh. Without this, sampling a 30 fps camera at 4 fps hands you frames
from several seconds ago — the driver's buffers are FIFO, not latest-wins.

The same thread maintains a short ring buffer, which is what lets an event clip
include the couple of seconds *before* the trigger fired.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraReader:
    def __init__(self, cfg: Any, ring_seconds: float = 0.0, ring_fps: float = 5.0):
        self.cfg = cfg
        self.ring_seconds = float(ring_seconds)
        self.ring_fps = float(ring_fps)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._frame: np.ndarray | None = None
        self._frame_ts: float = 0.0
        self._frame_seq: int = 0

        maxlen = max(1, int(self.ring_seconds * self.ring_fps) + 1)
        self._ring: collections.deque = collections.deque(maxlen=maxlen)
        self._last_ring_ts = 0.0

        self.frames_read = 0
        self.reconnects = 0
        self.read_failures = 0

    # ---------------------------------------------------------------- open

    def _open(self) -> cv2.VideoCapture:
        c = self.cfg
        cap = cv2.VideoCapture(c.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera device {c.device}")

        if c.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        # Ask for the shallowest queue the driver will give us; the reader thread
        # keeps it drained regardless.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c.autofocus else 0)
        if not c.autofocus and c.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, float(c.focus))
        # V4L2 convention: 3 = aperture priority (auto), 1 = manual.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if c.auto_exposure else 1)
        if not c.auto_exposure and c.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(c.exposure))
        if c.zoom is not None:
            cap.set(cv2.CAP_PROP_ZOOM, float(c.zoom))

        for _ in range(int(c.warmup_frames)):
            cap.read()

        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (got_w, got_h) != (c.width, c.height):
            log.warning(
                "camera negotiated %dx%d, config asked for %dx%d",
                got_w, got_h, c.width, c.height,
            )
        log.info("camera open: %dx%d @ %.0f fps (%s)",
                 got_w, got_h, cap.get(cv2.CAP_PROP_FPS), c.fourcc)
        return cap

    def actual_size(self) -> tuple[int, int]:
        with self._lock:
            if self._frame is not None:
                return self._frame.shape[1], self._frame.shape[0]
        return int(self.cfg.width), int(self.cfg.height)

    # -------------------------------------------------------------- thread

    def start(self) -> "CameraReader":
        self._cap = self._open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        self.wait_first_frame(timeout=10.0)
        return self

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            now = time.time()

            if not ok or frame is None:
                consecutive_failures += 1
                self.read_failures += 1
                if consecutive_failures >= 15:
                    log.error("camera read failed %d times — reconnecting",
                              consecutive_failures)
                    self._reconnect()
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            self.frames_read += 1

            with self._lock:
                self._frame = frame
                self._frame_ts = now
                self._frame_seq += 1
                if self._ring.maxlen and self.ring_seconds > 0:
                    if now - self._last_ring_ts >= 1.0 / max(self.ring_fps, 0.1):
                        self._ring.append((now, frame.copy()))
                        self._last_ring_ts = now

    def _reconnect(self) -> None:
        self.reconnects += 1
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        # Back off progressively; a USB camera that has fallen off the bus can
        # take a while to re-enumerate.
        for attempt in range(60):
            if self._stop.is_set():
                return
            delay = min(30.0, 1.0 * (attempt + 1))
            time.sleep(delay)
            try:
                self._cap = self._open()
                log.info("camera reconnected after %d attempt(s)", attempt + 1)
                return
            except Exception as exc:
                log.warning("reconnect attempt %d failed: %s", attempt + 1, exc)
        log.error("camera reconnect gave up")

    # ----------------------------------------------------------- accessors

    def wait_first_frame(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.02)
        return False

    def latest(self) -> tuple[float, np.ndarray | None]:
        """(timestamp, frame). The frame is a copy — callers may draw on it."""
        with self._lock:
            if self._frame is None:
                return 0.0, None
            return self._frame_ts, self._frame.copy()

    def latest_seq(self) -> int:
        with self._lock:
            return self._frame_seq

    def ring_snapshot(self, seconds: float | None = None) -> list[tuple[float, np.ndarray]]:
        """Buffered frames, oldest first, limited to the last `seconds`."""
        with self._lock:
            items = list(self._ring)
        if seconds is None:
            return items
        cutoff = time.time() - seconds
        return [(ts, f) for ts, f in items if ts >= cutoff]

    def set_prop(self, prop: int, value: float) -> bool:
        """Set a V4L2 property on the live capture (used by tools/aim.py)."""
        cap = self._cap
        if cap is None:
            return False
        with self._lock:
            return bool(cap.set(prop, float(value)))

    def get_prop(self, prop: int) -> float:
        cap = self._cap
        if cap is None:
            return float("nan")
        with self._lock:
            return float(cap.get(prop))

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self) -> "CameraReader":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
