"""Curbside parking as intervals on a line, not slots in a grid.

This street has no painted bays. Cars pack in wherever they fit, so "which
space is occupied" is not a well-posed question — the useful question is
"where along the curb is there a gap, and is it long enough to park in?"

So the curb is modelled as a polyline, parking as a set of occupied intervals
on its arc length, and free space as the complement of that set. A gap counts
as usable when it exceeds a real vehicle length plus room to manoeuvre.

Scale calibration uses the traffic itself. A passenger car is about 4.5 m
long, so every vehicle the camera sees is a ruler laid down at a known point
on the curb. Collecting (position, pixel-length) pairs over time and taking a
robust median per bin yields a pixels-per-metre function that varies along the
line exactly as perspective requires — no tape measure, no survey marks, and
it re-fits itself if the camera is ever nudged.

Caveats worth keeping in mind:
  * Pixel length approximates real length only while vehicles are side-on to
    the camera, which holds for a street running across the view.
  * A lone outlier (a bus, a motorcycle) would skew a mean, hence the median.
  * Gaps touching the edge of the frame are open-ended: we cannot see where
    they stop, so they are reported but marked, never measured.
"""

from __future__ import annotations

import bisect
import collections
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Nominal lengths in metres, used to turn detections into rulers.
CLASS_LENGTH_M = {"car": 4.5, "truck": 6.0, "bus": 11.0,
                  "motorcycle": 2.1, "bicycle": 1.7}
# Only these are trustworthy rulers: buses and trucks vary far too much.
CALIBRATION_CLASSES = {"car"}


@dataclass
class Interval:
    s0: float                 # arc length along the curb, metres
    s1: float
    track_id: int | None = None
    cls_name: str | None = None
    conf: float = 0.0

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass
class Gap:
    s0: float
    s1: float
    open_start: bool = False   # runs off the left edge of what we can see
    open_end: bool = False

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    @property
    def open_ended(self) -> bool:
        return self.open_start or self.open_end

    def to_json(self) -> dict:
        d = {"s0": round(self.s0, 2), "s1": round(self.s1, 2),
             "len_m": round(self.length, 2)}
        if self.open_ended:
            d["open"] = ("start" if self.open_start else "") + \
                        ("end" if self.open_end else "")
        return d


class CurbLine:
    """A polyline in image coordinates with arc-length parameterisation."""

    def __init__(self, name: str, points: list[list[float]]):
        self.name = name
        self.pts = np.asarray(points, dtype=np.float64)
        if len(self.pts) < 2:
            raise ValueError(f"curb {name}: need at least 2 points")
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total_px = float(self.cum[-1])

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Nearest point on the polyline: (arc length in px, perpendicular px)."""
        best_s, best_d = 0.0, float("inf")
        for i in range(len(self.pts) - 1):
            p, q = self.pts[i], self.pts[i + 1]
            v = q - p
            L2 = float(v @ v)
            if L2 <= 0:
                continue
            t = float(((np.array([x, y]) - p) @ v) / L2)
            t = max(0.0, min(1.0, t))
            proj = p + t * v
            d = math.hypot(x - proj[0], y - proj[1])
            if d < best_d:
                best_d = d
                best_s = float(self.cum[i] + t * self.seg_len[i])
        return best_s, best_d

    def point_at(self, s_px: float) -> tuple[float, float]:
        s_px = max(0.0, min(self.total_px, s_px))
        i = max(0, min(len(self.seg_len) - 1,
                       bisect.bisect_right(self.cum, s_px) - 1))
        t = (s_px - self.cum[i]) / max(self.seg_len[i], 1e-9)
        p = self.pts[i] + t * (self.pts[i + 1] - self.pts[i])
        return float(p[0]), float(p[1])

    def scaled(self, sx: float, sy: float) -> "CurbLine":
        return CurbLine(self.name, (self.pts * np.array([sx, sy])).tolist())


class ScaleModel:
    """Pixels-per-metre along one curb, learned from observed vehicles.

    Binned medians rather than a fitted curve: it makes no assumption about
    the camera's geometry, tolerates a bent curb, and degrades gracefully
    where samples are sparse.
    """

    def __init__(self, total_px: float, n_bins: int = 12,
                 min_samples: int = 8, max_per_bin: int = 400):
        self.total_px = float(total_px)
        self.n_bins = int(n_bins)
        self.min_samples = int(min_samples)
        self.max_per_bin = int(max_per_bin)
        # A bounded, recency-weighted window per bin rather than every sample
        # ever seen. Two reasons, and the second is the important one:
        #
        #   size   — one day of traffic produced 121,207 samples (2.8 MB),
        #            growing without bound, to compute twelve medians.
        #   truth  — keeping everything would quietly make this model
        #            *unable* to adapt. If the camera is nudged, a fresh
        #            sample has to outvote the entire history to move the
        #            median, so the scale would stay wrong indefinitely.
        #            A sliding window re-converges within a few hours.
        self.samples: list[collections.deque] = [
            collections.deque(maxlen=self.max_per_bin) for _ in range(self.n_bins)
        ]
        self._curve: list[float] | None = None
        self.global_px_per_m: float | None = None

    def _bin(self, s_px: float) -> int:
        f = 0.0 if self.total_px <= 0 else s_px / self.total_px
        return max(0, min(self.n_bins - 1, int(f * self.n_bins)))

    def add(self, s_px: float, px_len: float, length_m: float) -> None:
        if px_len <= 1 or length_m <= 0:
            return
        self.samples[self._bin(s_px)].append(px_len / length_m)
        self._curve = None

    @property
    def n_samples(self) -> int:
        return sum(len(b) for b in self.samples)

    def fit(self) -> bool:
        """Build the per-bin curve; returns True if usable."""
        allv = [v for b in self.samples for v in b]
        if not allv:
            return False
        self.global_px_per_m = float(np.median(allv))
        curve: list[float] = []
        for b in self.samples:
            curve.append(float(np.median(b)) if len(b) >= self.min_samples
                         else float("nan"))
        # Fill sparse bins from their neighbours, falling back to the global
        # median where a whole stretch of curb has never been parked on.
        arr = np.array(curve, dtype=float)
        idx = np.arange(self.n_bins)
        good = ~np.isnan(arr)
        if good.sum() == 0:
            arr[:] = self.global_px_per_m
        elif good.sum() < self.n_bins:
            arr = np.interp(idx, idx[good], arr[good])
        self._curve = arr.tolist()
        return True

    def px_per_m(self, s_px: float) -> float | None:
        if self._curve is None and not self.fit():
            return None
        return float(self._curve[self._bin(s_px)])

    def px_to_m(self, s_px: float) -> float | None:
        """Convert an arc-length position in px to metres by integrating 1/scale."""
        if self._curve is None and not self.fit():
            return None
        step = self.total_px / (self.n_bins * 4.0)
        metres, pos = 0.0, 0.0
        while pos < s_px:
            nxt = min(pos + step, s_px)
            scale = self._curve[self._bin(pos)]
            if scale > 0:
                metres += (nxt - pos) / scale
            pos = nxt
        return metres

    def to_json(self) -> dict:
        return {"total_px": self.total_px, "n_bins": self.n_bins,
                "min_samples": self.min_samples,
                "max_per_bin": self.max_per_bin,
                # Three decimals is far finer than the measurement is accurate
                # to, and keeps the file small enough to rewrite every 5 min.
                "samples": [[round(v, 3) for v in b] for b in self.samples],
                "global_px_per_m": self.global_px_per_m}

    @classmethod
    def from_json(cls, d: dict, px_scale: float = 1.0,
                  max_per_bin: int | None = None) -> "ScaleModel":
        """`px_scale` rescales a calibration measured at another resolution.

        Samples are pixels-per-metre, so they scale linearly with frame width.
        Without this, reusing a 1080p calibration at 4K would silently report
        every distance at half its true length.

        A file written before the per-bin cap existed is trimmed on load: the
        deque keeps the most recent `max_per_bin` of each bin.
        """
        cap = int(max_per_bin if max_per_bin is not None
                  else d.get("max_per_bin", 400))
        m = cls(d["total_px"] * px_scale, d.get("n_bins", 12),
                d.get("min_samples", 8), cap)
        for i, b in enumerate(d.get("samples") or []):
            if i < m.n_bins:
                m.samples[i].extend(v * px_scale for v in b)
        m.fit()
        return m


@dataclass
class CurbState:
    name: str
    # Two views of the same curb, deliberately kept apart:
    #   vehicles  — one interval per detected vehicle, never merged. This is
    #               the per-vehicle record that dwell-time analysis needs.
    #   intervals — the merged occupancy footprint, used only for gap maths.
    # Merging both together reported two touching cars as a single 11.5 m
    # "vehicle", which is nonsense as a measurement even though the gaps
    # derived from it were right.
    vehicles: list[Interval] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    length_m: float = 0.0
    occupied_m: float = 0.0
    calibrated: bool = False

    @property
    def free_m(self) -> float:
        return max(0.0, self.length_m - self.occupied_m)

    @property
    def largest_gap(self) -> Gap | None:
        closed = [g for g in self.gaps if not g.open_ended]
        return max(closed, key=lambda g: g.length) if closed else None

    def parkable(self, min_len_m: float) -> list[Gap]:
        return [g for g in self.gaps
                if not g.open_ended and g.length >= min_len_m]

    def to_json(self) -> dict:
        return {
            "curb": self.name,
            "calibrated": self.calibrated,
            "length_m": round(self.length_m, 2),
            "occupied_m": round(self.occupied_m, 2),
            "free_m": round(self.free_m, 2),
            "n_parked": len(self.vehicles),
            "n_footprints": len(self.intervals),
            "vehicles": [{"id": i.track_id, "c": i.cls_name,
                          "s0": round(i.s0, 2), "s1": round(i.s1, 2),
                          "len_m": round(i.length, 2)} for i in self.vehicles],
            "gaps": [g.to_json() for g in self.gaps],
        }


class CurbMonitor:
    """Maps detections onto curb lines and reports occupancy and gaps."""

    def __init__(self, curb_defs: list[dict], frame_size: tuple[int, int], cfg,
                 vehicle_classes: list[int] | None = None,
                 source_frame_size: tuple[int, int] | None = None,
                 calib_path: Path | None = None):
        self.cfg = cfg
        self.frame_size = frame_size
        # Pixel thresholds in the config are expressed at 1920 wide so that
        # changing capture resolution does not require re-tuning them.
        self.px_scale = frame_size[0] / 1920.0
        self.max_perp_px = float(cfg.max_perp_px) * self.px_scale
        self.vehicle_classes = set(vehicle_classes or [1, 2, 3, 5, 7])
        self.curbs: list[CurbLine] = []
        self.scales: dict[str, ScaleModel] = {}
        self.calib_path = calib_path

        w, h = frame_size
        sx = sy = 1.0
        if source_frame_size and tuple(source_frame_size) != tuple(frame_size):
            sx = w / float(source_frame_size[0])
            sy = h / float(source_frame_size[1])

        for cd in curb_defs:
            line = CurbLine(cd["name"], cd["points"])
            if (sx, sy) != (1.0, 1.0):
                line = line.scaled(sx, sy)
            self.curbs.append(line)
            self.scales[line.name] = ScaleModel(
                line.total_px, int(cfg.scale_bins), int(cfg.scale_min_samples),
                int(getattr(cfg, "scale_max_per_bin", 400)))

        if calib_path and Path(calib_path).exists():
            self.load_calibration(calib_path)
        log.info("curb monitor: %d line(s), %s",
                 len(self.curbs),
                 ", ".join(f"{c.name}={c.total_px:.0f}px" for c in self.curbs))

    # ------------------------------------------------------- calibration

    def load_calibration(self, path: Path) -> None:
        try:
            doc = json.loads(Path(path).read_text())
        except Exception as exc:
            log.warning("could not read curb calibration: %s", exc)
            return
        src_w = float(doc.get("frame_width") or 1920.0)
        px_scale = self.frame_size[0] / src_w
        if abs(px_scale - 1.0) > 1e-6:
            log.info("curb calibration was measured at %.0f px wide, now %d — "
                     "rescaling by %.3f", src_w, self.frame_size[0], px_scale)
        for name, d in doc.get("scales", {}).items():
            if name in self.scales:
                self.scales[name] = ScaleModel.from_json(
                    d, px_scale, int(getattr(self.cfg, "scale_max_per_bin", 400)))
        log.info("loaded curb calibration: %s",
                 ", ".join(f"{n}={m.n_samples} samples"
                           for n, m in self.scales.items()))

    def save_calibration(self, path: Path | None = None) -> None:
        path = Path(path or self.calib_path)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"frame_width": self.frame_size[0],
             "scales": {n: m.to_json() for n, m in self.scales.items()}},
            indent=1))

    def calibrated(self, name: str) -> bool:
        m = self.scales.get(name)
        return bool(m and m.n_samples >= int(self.cfg.scale_min_samples))

    # ------------------------------------------------------------ assign

    def assign(self, det) -> tuple[CurbLine, float, float, float] | None:
        """Which curb this vehicle sits on, and its pixel interval there.

        Uses the bottom corners of the box — where the vehicle meets the road —
        because the top of a tall vehicle projects well off its own footprint.
        """
        if det.cls_id not in self.vehicle_classes:
            return None
        x1, _y1, x2, y2 = det.xyxy
        best = None
        for curb in self.curbs:
            s_a, d_a = curb.project(x1, y2)
            s_b, d_b = curb.project(x2, y2)
            perp = (d_a + d_b) / 2.0
            if perp > self.max_perp_px:
                continue
            if best is None or perp < best[3]:
                best = (curb, min(s_a, s_b), max(s_a, s_b), perp)
        return best

    def observe_for_calibration(self, det) -> None:
        """Feed one side-on vehicle into the scale model as a ruler."""
        if det.name not in CALIBRATION_CLASSES:
            return
        a = self.assign(det)
        if a is None:
            return
        curb, s0, s1, _perp = a
        # A box clipped by the frame edge is not a whole vehicle, so it is not
        # a valid ruler.
        w, h = self.frame_size
        if det.xyxy[0] <= 2 or det.xyxy[2] >= w - 2:
            return
        self.scales[curb.name].add((s0 + s1) / 2.0, s1 - s0,
                                   CLASS_LENGTH_M[det.name])

    # ------------------------------------------------------------ update

    def update(self, dets, parked_ids: set[int] | None = None) -> list[CurbState]:
        """Occupancy and gaps per curb. Only `parked_ids` occupy space —
        a car driving past must not be reported as parked."""
        states: list[CurbState] = []
        for curb in self.curbs:
            scale = self.scales[curb.name]
            ok = scale.fit()
            state = CurbState(name=curb.name, calibrated=ok)
            total_m = scale.px_to_m(curb.total_px) if ok else curb.total_px
            state.length_m = float(total_m or 0.0)

            raw: list[Interval] = []
            for d in dets:
                if parked_ids is not None and d.track_id not in parked_ids:
                    continue
                a = self.assign(d)
                if a is None or a[0].name != curb.name:
                    continue
                _c, s0_px, s1_px, _p = a
                if ok:
                    s0 = scale.px_to_m(s0_px) or 0.0
                    s1 = scale.px_to_m(s1_px) or 0.0
                else:
                    s0, s1 = s0_px, s1_px
                raw.append(Interval(min(s0, s1), max(s0, s1),
                                    d.track_id, d.name, float(d.conf)))

            raw.sort(key=lambda i: i.s0)
            state.vehicles = list(raw)
            merged: list[Interval] = []
            for iv in raw:
                # Boxes of adjacent cars often touch; merging keeps the
                # occupied set honest rather than inventing slivers of space.
                if merged and iv.s0 <= merged[-1].s1:
                    if iv.s1 > merged[-1].s1:
                        merged[-1] = Interval(merged[-1].s0, iv.s1,
                                              merged[-1].track_id,
                                              merged[-1].cls_name,
                                              merged[-1].conf)
                else:
                    merged.append(iv)
            state.intervals = merged
            state.occupied_m = sum(i.length for i in merged)

            cursor = 0.0
            for iv in merged:
                if iv.s0 - cursor > 1e-6:
                    state.gaps.append(Gap(cursor, iv.s0,
                                          open_start=(cursor <= 1e-6)))
                cursor = max(cursor, iv.s1)
            if state.length_m - cursor > 1e-6:
                state.gaps.append(Gap(cursor, state.length_m, open_end=True))
            states.append(state)
        return states

    # -------------------------------------------------------------- draw

    def draw(self, frame: np.ndarray, states: list[CurbState]) -> np.ndarray:
        by_name = {s.name: s for s in states}
        for curb in self.curbs:
            state = by_name.get(curb.name)
            scale = self.scales[curb.name]
            cv2.polylines(frame, [curb.pts.astype(np.int32)], False,
                          (90, 90, 90), 2)
            if state is None:
                continue

            def to_px(metres: float) -> float:
                if not state.calibrated or not scale.global_px_per_m:
                    return metres
                lo, hi = 0.0, curb.total_px
                for _ in range(24):                     # invert px_to_m
                    mid = (lo + hi) / 2
                    if (scale.px_to_m(mid) or 0.0) < metres:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2

            for iv in state.intervals:
                a = curb.point_at(to_px(iv.s0))
                b = curb.point_at(to_px(iv.s1))
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (0, 0, 220), 7)
            min_len = float(self.cfg.min_gap_m)
            for g in state.gaps:
                a = curb.point_at(to_px(g.s0))
                b = curb.point_at(to_px(g.s1))
                usable = (not g.open_ended) and g.length >= min_len
                color = (0, 220, 0) if usable else (0, 170, 220)
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         color, 7)
                if state.calibrated and g.length >= 1.0:
                    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                    label = f"{g.length:.1f}m" + ("?" if g.open_ended else "")
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.55, 2)
                    cv2.rectangle(frame, (int(mx - tw / 2) - 3, int(my) - th - 20),
                                  (int(mx + tw / 2) + 3, int(my) - 6), (0, 0, 0), -1)
                    cv2.putText(frame, label, (int(mx - tw / 2), int(my) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                                cv2.LINE_AA)
        return frame


def autofit_curbs(dets, frame_size: tuple[int, int],
                  vehicle_classes: set[int] | None = None,
                  min_points: int = 2,
                  min_separation_px: float = 120.0) -> list[dict]:
    """Propose curb lines from where vehicles meet the road.

    Parked cars sit on the curb by definition, so their ground-contact points
    *are* the curb — no need to survey it by hand. Points are split into a near
    and a far group by the largest vertical gap between them, then a
    total-least-squares line is fitted through each and extended across the
    frame.

    Returns [] rather than a bad guess when there is too little evidence; the
    caller is expected to try again as more vehicles arrive.
    """
    vehicle_classes = vehicle_classes or {1, 2, 3, 5, 7}
    W, H = frame_size
    pts = np.array([[(d.xyxy[0] + d.xyxy[2]) / 2.0, float(d.xyxy[3])]
                    for d in dets if d.cls_id in vehicle_classes])
    if len(pts) < min_points:
        return []

    order = np.argsort(pts[:, 1])
    ys = pts[order, 1]
    groups: dict[str, np.ndarray] = {}
    if len(ys) > 2:
        gaps = np.diff(ys)
        k = int(np.argmax(gaps))
        # Only split when the two rows are genuinely distinct; otherwise a
        # single curb gets sliced into two parallel halves.
        if gaps[k] >= min_separation_px:
            groups["far"] = pts[order[:k + 1]]
            groups["near"] = pts[order[k + 1:]]
    if not groups:
        groups["far"] = pts

    out: list[dict] = []
    for name, g in groups.items():
        if len(g) < min_points:
            continue
        c = g.mean(axis=0)
        _u, _s, vt = np.linalg.svd(g - c)
        d = vt[0]
        if abs(d[0]) < 1e-6:
            continue
        t0, t1 = (0 - c[0]) / d[0], (W - c[0]) / d[0]
        p0, p1 = c + min(t0, t1) * d, c + max(t0, t1) * d
        p0 = [float(np.clip(p0[0], 0, W)), float(np.clip(p0[1], 0, H))]
        p1 = [float(np.clip(p1[0], 0, W)), float(np.clip(p1[1], 0, H))]
        residual = float(np.abs((g - c) @ vt[1]).max())
        out.append({"name": name, "points": [p0, p1],
                    "fitted_from": int(len(g)),
                    "max_residual_px": round(residual, 1)})
    return out
