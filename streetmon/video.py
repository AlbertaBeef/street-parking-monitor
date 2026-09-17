"""H.264 event clips via ffmpeg.

Event bursts are written as video rather than JPEG sequences. On this machine
the byte saving is real but secondary; the decisive argument is file count. A
busy day of JPEG bursts lands ~15,000 files, and a year of that makes backup,
rsync and even `ls` miserable. One clip per event keeps it to a few hundred
files a day.

Encoding goes to the Radeon iGPU through VAAPI when it is available, so a
long-running collector spends almost no CPU on it. If ffmpeg or VAAPI is
missing we fall back to libx264, and if ffmpeg is missing entirely the caller
falls back to writing JPEGs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VAAPI_DEVICE = "/dev/dri/renderD128"


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def has_vaapi() -> bool:
    exe = ffmpeg_path()
    if not exe or not Path(VAAPI_DEVICE).exists():
        return False
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
        return "h264_vaapi" in out
    except Exception:
        return False


@lru_cache(maxsize=1)
def encoder_name() -> str:
    if not ffmpeg_path():
        return "none"
    return "h264_vaapi" if has_vaapi() else "libx264"


class EventVideoWriter:
    """Feeds raw BGR frames to an ffmpeg subprocess over stdin."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 quality: int = 24, use_vaapi: bool | None = None):
        self.path = Path(path)
        self.size = size
        self.fps = max(float(fps), 1.0)
        self.frames = 0
        self.proc: subprocess.Popen | None = None
        self.encoder = "none"

        exe = ffmpeg_path()
        if not exe:
            return

        w, h = size
        # Odd dimensions break yuv420p; both our capture modes are even, but a
        # cropped source would not be.
        w, h = w - (w % 2), h - (h % 2)
        self.size = (w, h)
        vaapi = has_vaapi() if use_vaapi is None else (use_vaapi and has_vaapi())

        # Global options must precede the input; -vaapi_device is a global.
        globals_ = [exe, "-hide_banner", "-loglevel", "error", "-y"]
        source = ["-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-"]
        if vaapi:
            cmd = (globals_ + ["-vaapi_device", VAAPI_DEVICE] + source
                   + ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                      "-qp", str(int(quality))])
            self.encoder = "h264_vaapi"
        else:
            cmd = globals_ + source + ["-c:v", "libx264", "-preset", "veryfast",
                                       "-crf", str(int(quality)),
                                       "-pix_fmt", "yuv420p"]
            self.encoder = "libx264"
        cmd += ["-movflags", "+faststart", str(self.path)]

        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE)
        except Exception as exc:
            log.error("ffmpeg failed to start: %s", exc)
            self.proc = None
            self.encoder = "none"

    @property
    def ok(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def write(self, frame: np.ndarray) -> bool:
        if not self.ok:
            return False
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            import cv2
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
            return True
        except (BrokenPipeError, ValueError, OSError) as exc:
            log.error("event video write failed: %s", exc)
            self._drain_error()
            return False

    def _drain_error(self) -> None:
        if self.proc and self.proc.stderr:
            try:
                err = self.proc.stderr.read().decode(errors="replace").strip()
                if err:
                    log.error("ffmpeg: %s", err[:500])
            except Exception:
                pass

    def close(self) -> dict:
        if self.proc is None:
            return {"frames": 0, "bytes": 0, "encoder": "none"}
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not exit; killing")
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception as exc:
            log.error("closing ffmpeg: %s", exc)
        if self.proc.returncode not in (0, None):
            self._drain_error()
        size = self.path.stat().st_size if self.path.exists() else 0
        self.proc = None
        return {"frames": self.frames, "bytes": size, "encoder": self.encoder}
