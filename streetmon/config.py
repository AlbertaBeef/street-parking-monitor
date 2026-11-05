"""Configuration loading.

`config/site.yaml` is hand-edited and may be partial; anything it omits comes
from DEFAULTS below. Parking-spot polygons are kept separately in
`config/curb.yaml` because a tool generates that file and would strip comments.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "site": {"name": "street-cam", "notes": ""},
    "camera": {
        "device": 0,
        "fourcc": "MJPG",
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "warmup_frames": 10,
        "autofocus": True,
        "focus": None,
        "auto_exposure": True,
        "exposure": None,
        "zoom": None,
    },
    "detector": {
        "model": "yolo11m.pt",
        "device": 0,
        "imgsz": 960,
        "conf": 0.30,
        "iou": 0.50,
        "tracker": "bytetrack.yaml",
        # NMS uses IoU and cannot see containment, so a slice of one car
        # detected as a second car survives it. These clean that up.
        "nested_contained_min": 0.80,  # fraction of the inner box inside the outer
        "nested_conf_margin": 0.25,    # ...and it must be this much less confident
        "aspect_max": 3.5,        # a side-on car is 1.8-2.6; 4+ is a roofline
        "aspect_min_conf": 0.60,  # ...but only reject if the model is unsure too
        "classes": [0, 1, 2, 3, 5, 7],
        "vehicle_classes": [1, 2, 3, 5, 7],
    },
    "capture": {
        "detect_fps": 4.0,
        "timelapse_width": 1920,   # timelapse is downscaled from the capture
        "timelapse_sec": 10,
        "log_every_n": 1,
        "event": {
            "enabled": True,
            "fps": 5.0,
            "pre_roll_sec": 2.0,
            "post_roll_sec": 5.0,
            "max_sec": 90,
            "trigger_on_new_vehicle": True,
            "quality": 24,   # H.264 qp; lower = better/larger
            "width": 1920,   # event clip width; 0 = full capture resolution
            "trigger_on_person": True,
        },
        "crops": {
            "enabled": True,
            "max_per_track": 8,
            "min_interval_sec": 2.0,        # stationary vehicles
            "min_interval_sec_moving": 0.0,  # moving: keep every sample
            "min_box_px": 48,
            "pad": 0.08,
            "jpeg_quality": 92,
            "persons": True,
            "min_box_px_person": 32,
        },
    },
    "triage": {
        "enabled": True,
        # Gated features: white body + dashed red mid-body stripe.
        "white_min_frac": 0.18,
        "red_band_min_frac": 0.012,
        "red_min_dashes": 3,      # a stripe is dashed; a tail light is one blob
        "dash_min_area_frac": 0.0025,
        "dash_max_area_frac": 0.25,
        "dash_row_tol_frac": 0.35,
        "white_sat_max": 45,
        "white_val_min": 165,
        "red_hue_lo": 8,        # hue wraps: <=8 or >=172 is red
        "red_hue_hi": 172,
        "red_sat_min": 65,
        "red_val_min": 70,
        "band_top": 0.45,       # mid-body rows only: excludes brake lights
        "band_bottom": 0.85,
        # Recorded but not gated — the crest is too small to rely on.
        "blue_hue": [95, 135],
        "blue_sat_min": 90,
        "blue_val_min": 55,
    },
    "schedule": {
        "enabled": True,
        "active": [{"days": ["mon", "tue", "wed", "thu", "fri"],
                    "start": "07:00", "end": "19:00", "label": "enforcement"}],
        "idle": {"timelapse_sec": 300, "events": False, "crops": False},
    },
    "kinematic": {
        # Speeds are body-lengths/sec: scale-invariant, so no scene calibration.
        # A 4.5 m car at 30 km/h is ~1.9 BL/s; at 10 km/h, ~0.6 BL/s.
        "enabled": True,
        "patrol_max_bl_s": 1.0,      # ~16 km/h for a 4.5 m car
        "min_travel_bl": 3.0,        # must actually go somewhere
        "min_duration_sec": 3.0,
        "min_stops_for_flag": 1,     # a mid-street pause is suspicious alone
        "expire_sec": 8.0,           # unseen this long => track is finished
        # Event-trigger motion test (see tracks.py). A parked car's box
        # flickers in extent; when it collapses the centroid lurches and reads
        # as motion. Direction separates them: jitter wanders (straightness
        # ~0.30), real motion is monotonic (~1.00).
        "motion_straightness_min": 0.75,
        "motion_confirm_samples": 2,
    },
    "curb": {
        # This street has no painted bays, so parking is modelled as intervals
        # on a line and free space as the gaps between them.
        "enabled": True,
        "max_perp_px": 260,       # how far off the line a vehicle may sit
        "min_gap_m": 5.4,         # a 4.5 m car plus manoeuvring room
        "scale_bins": 12,
        "scale_min_samples": 8,
        "scale_max_per_bin": 400,  # sliding window; unbounded history cannot adapt
        "gap_confirm_sec": 8.0,   # a gap must hold this long to be believed
        "log_interval_sec": 30,
    },
    "storage": {
        "root": "data",
        "jpeg_quality": 85,
        "min_free_gb": 50,
        "retention_days": 0,
        "rotate_daily": True,
    },
}


class Cfg(dict):
    """dict with attribute access, so `cfg.camera.width` reads naturally."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge; scalars and lists in `override` replace those in `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Cfg:
    """Load site.yaml merged over DEFAULTS. Missing file is fine — pure defaults."""
    path = Path(path) if path else REPO_ROOT / "config" / "site.yaml"
    user: dict = {}
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
    cfg = _wrap(_deep_merge(DEFAULTS, user))
    cfg.config_path = str(path)
    return cfg


def curb_path() -> Path:
    return REPO_ROOT / "config" / "curb.yaml"


def curb_calib_path() -> Path:
    return REPO_ROOT / "config" / "curb_calibration.json"


def load_curb_doc(path: str | Path | None = None) -> dict:
    path = Path(path) if path else curb_path()
    if not path.exists():
        return {"frame_size": None, "curbs": []}
    return yaml.safe_load(path.read_text()) or {"frame_size": None, "curbs": []}


def save_curb_doc(doc: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else curb_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Curb lines for interval-based parking occupancy.\n"
        "# This street has no painted bays, so parking is measured as intervals\n"
        "# along these lines and free space as the gaps between them.\n"
        "# Auto-fitted from where vehicles actually park; adjust with\n"
        "# tools/curb_editor.py, or delete this file to have it re-fitted.\n"
        "# Coordinates are pixels at `frame_size` and are rescaled if the\n"
        "# capture resolution changes.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    return path


