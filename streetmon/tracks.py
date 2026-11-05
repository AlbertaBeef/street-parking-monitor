"""Per-track kinematics — the behavioural half of parking-police triage.

The colour filter in `triage.py` asks what a vehicle *looks* like. This module
asks what it *does*, which is the more reliable question: an enforcement vehicle
patrols at a crawl, pauses beside parked cars and sometimes reverses, while
ordinary traffic crosses the frame at road speed and never stops mid-street.
Behaviour survives rain, dusk and a livery we guessed wrong about.

Speed is measured in **body-lengths per second**, not pixels per second. A car
near the camera sweeps far more pixels than an identical car across the street,
so raw pixel speed mostly encodes distance from the lens. Dividing by the
vehicle's own box width cancels that out and leaves a quantity with physical
meaning: a 4.5 m car at 30 km/h moves ~1.9 BL/s, the same car at 10 km/h moves
~0.6 BL/s. That ratio is what separates a patrol crawl from through traffic,
and it needs no calibration of the scene.

One assumption this rests on: box width approximates vehicle length, which
holds while traffic crosses the frame side-on. That is true for this camera —
the street runs left-to-right across the view — but a camera pointed *along* a
road would see approaching vehicles box-width-first (~1.8 m, not ~4.5 m) and
read them as roughly 2.5x faster than they are.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Below this, a vehicle is treated as not moving at all.
STATIONARY_BL_S = 0.05

# A second, stricter motion test used only to open event clips.
#
# A parked car's detection box flickers in extent -- one measured example swung
# between 900 and 2829 px wide -- and when it collapses the centroid lurches
# hundreds of pixels, which the instantaneous speed test reads as motion. On
# 2026-11-02 that produced 387 spurious event clips out of 542, 4.9 hours of
# video in which nothing moved a single body-length.
#
# What separates the two is direction, not magnitude. Jitter wanders, so net
# displacement over a few samples stays far below the path walked; real motion
# is monotonic. Measured straightness (net/path over 4 steps): parked cars
# 0.30, transiting vehicles 1.00.
STRAIGHTNESS_SAMPLES = 4


@dataclass
class Sample:
    ts: float
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass
class Track:
    track_id: int
    cls_name: str
    cls_id: int
    first_ts: float
    last_ts: float
    samples: list[Sample] = field(default_factory=list)

    crops_saved: int = 0
    last_crop_ts: float = 0.0
    crop_files: list[str] = field(default_factory=list)

    best_white: float = 0.0
    best_red: float = 0.0
    colour_candidate: bool = False

    curbs_touched: set[str] = field(default_factory=set)
    edge_entry: bool = False
    edge_exit: bool = False

    # rolling kinematic state, updated as samples arrive
    speed_bl_s: float = 0.0
    straightness: float = 0.0
    # `moving` is deliberately sensitive: it drives crop capture and curb
    # occupancy, where reacting late costs real data and a false trigger is
    # cheap. `moving_confirmed` is deliberately strict: it opens event clips,
    # where a false trigger costs minutes of video.
    moving: bool = False
    moving_confirmed: bool = False
    _recent_moving: list = field(default_factory=list)

    def add(self, ts: float, cx: float, cy: float, w: float, h: float,
            conf: float) -> None:
        self.samples.append(Sample(ts, cx, cy, w, h, conf))
        self.last_ts = ts
        # Cap history so a car parked for six hours cannot grow without bound;
        # the summary statistics below only need the shape of the path.
        if len(self.samples) > 4000:
            del self.samples[1::2]

    def _straightness(self, k: int = STRAIGHTNESS_SAMPLES) -> float:
        """Net displacement divided by path walked, over the last k steps.

        ~1.0 for a vehicle going somewhere, ~0.3 for a stationary box jittering
        about its mean. This is what tells a parked car apart from a moving one
        without needing a long observation window -- which matters, because a
        vehicle can cross this frame in under two seconds.
        """
        seg = self.samples[-(k + 1):]
        if len(seg) < 3:
            return 0.0
        net = math.hypot(seg[-1].cx - seg[0].cx, seg[-1].cy - seg[0].cy)
        path = sum(math.hypot(seg[i].cx - seg[i - 1].cx, seg[i].cy - seg[i - 1].cy)
                   for i in range(1, len(seg)))
        return (net / path) if path > 0 else 0.0

    def update_speed(self, min_dt: float = 0.5, straight_min: float = 0.75,
                     confirm_samples: int = 2) -> None:
        """Instantaneous speed over the last >=min_dt seconds, in BL/s."""
        if len(self.samples) < 2:
            return
        last = self.samples[-1]
        ref = None
        for s in reversed(self.samples[:-1]):
            if last.ts - s.ts >= min_dt:
                ref = s
                break
        if ref is None:
            return
        dt = last.ts - ref.ts
        if dt <= 0:
            return
        dist = math.hypot(last.cx - ref.cx, last.cy - ref.cy)
        bl = max(last.w, 1.0)
        self.speed_bl_s = (dist / bl) / dt
        self.moving = self.speed_bl_s >= STATIONARY_BL_S

        self.straightness = self._straightness()
        decision = self.moving and self.straightness >= straight_min
        self._recent_moving.append(decision)
        if len(self._recent_moving) > 8:
            del self._recent_moving[:-8]
        n = max(1, int(confirm_samples))
        self.moving_confirmed = (len(self._recent_moving) >= n
                                 and all(self._recent_moving[-n:]))


def _segment_speeds(samples: list[Sample], min_dt: float = 0.4) -> list[float]:
    """Body-lengths/sec between samples at least min_dt apart."""
    speeds: list[float] = []
    anchor = 0
    for i in range(1, len(samples)):
        dt = samples[i].ts - samples[anchor].ts
        if dt < min_dt:
            continue
        a, b = samples[anchor], samples[i]
        dist = math.hypot(b.cx - a.cx, b.cy - a.cy)
        bl = max((a.w + b.w) / 2.0, 1.0)
        speeds.append((dist / bl) / dt)
        anchor = i
    return speeds


def summarise(track: Track, cfg) -> dict:
    """Collapse a finished track into one analysis record."""
    s = track.samples
    duration = max(track.last_ts - track.first_ts, 0.0)
    rec: dict = {
        "id": track.track_id,
        "c": track.cls_name,
        "ci": track.cls_id,
        "first_ts": round(track.first_ts, 3),
        "last_ts": round(track.last_ts, 3),
        "duration_sec": round(duration, 2),
        "n_samples": len(s),
        "crops": track.crop_files,
        "curbs": sorted(track.curbs_touched),
        "edge_entry": track.edge_entry,
        "edge_exit": track.edge_exit,
        "triage": {
            "best_white": round(track.best_white, 4),
            "best_red_band": round(track.best_red, 4),
            "colour_candidate": track.colour_candidate,
        },
    }
    if len(s) < 2:
        rec.update(motion_class="fleeting", direction="unknown",
                   kinematic_candidate=False)
        return rec

    first, last = s[0], s[-1]
    bl = max(sum(x.w for x in s) / len(s), 1.0)
    net_px = math.hypot(last.cx - first.cx, last.cy - first.cy)
    path_px = sum(math.hypot(s[i].cx - s[i - 1].cx, s[i].cy - s[i - 1].cy)
                  for i in range(1, len(s)))
    net_bl = net_px / bl

    speeds = _segment_speeds(s)
    mean_sp = sum(speeds) / len(speeds) if speeds else 0.0
    med_sp = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
    max_sp = max(speeds) if speeds else 0.0
    moving_flags = [sp >= STATIONARY_BL_S for sp in speeds]
    stationary_frac = (1.0 - sum(moving_flags) / len(moving_flags)) if moving_flags else 1.0

    # A "stop" is a move -> halt -> move cycle: the pause an enforcement
    # vehicle makes beside a parked car, which through traffic does not do.
    stops = 0
    for i in range(1, len(moving_flags) - 1):
        if not moving_flags[i] and moving_flags[i - 1]:
            if any(moving_flags[i + 1:]):
                stops += 1

    dx, dy = last.cx - first.cx, last.cy - first.cy
    bearing = math.degrees(math.atan2(-dy, dx))
    if net_bl < 0.5:
        direction = "static"
    elif abs(dx) >= abs(dy):
        direction = "L2R" if dx > 0 else "R2L"
    else:
        direction = "toward" if dy > 0 else "away"

    if stationary_frac > 0.90 and net_bl < 0.5:
        motion_class = "parked"
    elif net_bl >= 2.0 and stationary_frac < 0.5:
        motion_class = "transit"
    elif moving_flags and moving_flags[0] and not moving_flags[-1]:
        motion_class = "arrival"
    elif moving_flags and not moving_flags[0] and moving_flags[-1]:
        motion_class = "departure"
    else:
        motion_class = "complex"

    # The patrol signature: actually went somewhere, but slowly — optionally
    # pausing on the way. Parked cars are excluded by the net-displacement
    # test, through traffic by the speed ceiling.
    kin = cfg.kinematic
    # Pedestrians walk at exactly the pace this filter looks for, so a person
    # would trip it on every pass. Only vehicles can be a patrol.
    is_vehicle = track.cls_id in set(cfg.detector.vehicle_classes)
    patrol_like = (
        is_vehicle
        and motion_class in ("transit", "complex")
        and net_bl >= float(kin.min_travel_bl)
        and 0.0 < med_sp <= float(kin.patrol_max_bl_s)
        and duration >= float(kin.min_duration_sec)
    )
    stopped_midstreet = (
        is_vehicle
        and motion_class in ("transit", "complex")
        and stops >= int(kin.min_stops_for_flag)
        and net_bl >= float(kin.min_travel_bl)
    )

    rec.update(
        motion_class=motion_class,
        direction=direction,
        bearing_deg=round(bearing, 1),
        net_bl=round(net_bl, 2),
        path_bl=round(path_px / bl, 2),
        mean_speed_bl_s=round(mean_sp, 3),
        median_speed_bl_s=round(med_sp, 3),
        max_speed_bl_s=round(max_sp, 3),
        stationary_frac=round(stationary_frac, 3),
        stops=stops,
        mean_box_px=round(bl, 1),
        kinematic_candidate=bool(patrol_like or stopped_midstreet),
        kinematic_reason=("slow_patrol" if patrol_like else
                          "stopped_midstreet" if stopped_midstreet else None),
        # Downsampled path, for plotting and for phase-2 feature work.
        path=[[round(x.cx), round(x.cy), round(x.ts, 2)]
              for x in s[::max(1, len(s) // 60)]],
    )
    return rec


class TrackRegistry:
    """Owns live tracks and emits a summary when one goes away."""

    def __init__(self, cfg, expire_sec: float = 8.0):
        self.cfg = cfg
        self.expire_sec = expire_sec
        self.live: dict[int, Track] = {}
        self.completed = 0
        self.kinematic_hits = 0

    def observe(self, det, ts: float, frame_wh: tuple[int, int]) -> Track:
        tid = det.track_id
        tr = self.live.get(tid)
        if tr is None:
            tr = Track(track_id=tid, cls_name=det.name, cls_id=det.cls_id,
                       first_ts=ts, last_ts=ts)
            w, h = frame_wh
            tr.edge_entry = (det.xyxy[0] <= 2 or det.xyxy[2] >= w - 2)
            self.live[tid] = tr
        cx, cy = det.center
        bw, bh = det.wh
        tr.add(ts, cx, cy, float(bw), float(bh), float(det.conf))
        tr.update_speed(
            straight_min=float(getattr(self.cfg.kinematic,
                                       "motion_straightness_min", 0.75)),
            confirm_samples=int(getattr(self.cfg.kinematic,
                                        "motion_confirm_samples", 2)))
        tr.cls_name = det.name
        return tr

    def reap(self, now: float, frame_wh: tuple[int, int]) -> list[dict]:
        """Summarise and drop tracks not seen for `expire_sec`."""
        out: list[dict] = []
        for tid in [t for t, tr in self.live.items()
                    if now - tr.last_ts > self.expire_sec]:
            tr = self.live.pop(tid)
            if tr.samples:
                w, _ = frame_wh
                last = tr.samples[-1]
                tr.edge_exit = (last.cx - last.w / 2 <= 3
                                or last.cx + last.w / 2 >= w - 3)
            rec = summarise(tr, self.cfg)
            self.completed += 1
            if rec.get("kinematic_candidate"):
                self.kinematic_hits += 1
            out.append(rec)
        return out

    def flush(self, frame_wh: tuple[int, int]) -> list[dict]:
        out = [summarise(tr, self.cfg) for tr in self.live.values() if tr.samples]
        self.live.clear()
        return out
