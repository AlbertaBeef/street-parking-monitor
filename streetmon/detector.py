"""YOLO detection + multi-object tracking.

Tracking matters more than detection here. A parked car produces an identical
box in every frame for hours; without stable track ids we would save thousands
of near-duplicate crops of the same vehicle and learn nothing. Track ids let us
treat "one vehicle visit" as the unit of data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# BGR colours per class family, for overlays.
_COLORS = {
    "person": (0, 200, 255),
    "bicycle": (255, 200, 0),
    "car": (0, 255, 0),
    "motorcycle": (255, 160, 0),
    "bus": (255, 0, 200),
    "truck": (0, 160, 255),
}
_DEFAULT_COLOR = (200, 200, 200)


def _area(b: tuple) -> int:
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _contained_frac(inner: tuple, outer: tuple) -> float:
    """How much of `inner` lies inside `outer`, as a fraction of *inner*.

    Deliberately asymmetric. An earlier version divided by the smaller of the
    two areas, which made the test symmetric and let a small ghost "contain"
    the large real car it was sitting on — suppressing the car instead. Here
    the caller decides which box is the inner one, and only that box can lose.
    """
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inter = ix * iy
    return 0.0 if inter <= 0 else inter / _area(inner)


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def wh(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return x2 - x1, y2 - y1

    @property
    def area(self) -> int:
        w, h = self.wh
        return max(0, w) * max(0, h)

    def to_json(self) -> dict:
        d = {
            "id": self.track_id,
            "c": self.name,
            "ci": self.cls_id,
            "p": round(float(self.conf), 3),
            "b": [int(v) for v in self.xyxy],
        }
        if self.extra:
            d.update(self.extra)
        return d


class Detector:
    def __init__(self, cfg: Any):
        from ultralytics import YOLO  # imported lazily; pulls in torch

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.names: dict[int, str] = self.model.names
        self.classes = list(cfg.classes)
        self.vehicle_classes = set(cfg.vehicle_classes)
        self.device = cfg.device
        self.infer_ms_ema: float | None = None
        self.suppressed_nested = 0
        self.suppressed_aspect = 0

        log.info("detector: %s on device=%s imgsz=%d conf=%.2f",
                 cfg.model, cfg.device, cfg.imgsz, cfg.conf)

    def is_vehicle(self, det: Detection) -> bool:
        return det.cls_id in self.vehicle_classes

    def track(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.time()
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            classes=self.classes,
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            imgsz=int(self.cfg.imgsz),
            device=self.device,
            verbose=False,
        )
        dt_ms = (time.time() - t0) * 1000.0
        self.infer_ms_ema = dt_ms if self.infer_ms_ema is None else (
            0.9 * self.infer_ms_ema + 0.1 * dt_ms
        )

        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # boxes.id is None on frames where the tracker produced no confirmed tracks.
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        dets: list[Detection] = []
        for i in range(len(clss)):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
            dets.append(
                Detection(
                    cls_id=int(clss[i]),
                    name=self.names.get(int(clss[i]), str(clss[i])),
                    conf=float(confs[i]),
                    xyxy=(x1, y1, x2, y2),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return self.suppress_artifacts(dets)

    def suppress_artifacts(self, dets: list[Detection]) -> list[Detection]:
        """Drop nested duplicates and implausibly-shaped vehicle boxes.

        Shooting a street of parked cars produces two recurring artifacts that
        survive NMS:

        * **Nested duplicates** — a slice of one car (typically the roofline)
          detected as a second, weaker car. Measured on a real case: IoU 0.187
          against its parent but 94% contained in it, so NMS at IoU 0.5 kept
          both.
        * **Letterbox boxes** — that roofline slice has an aspect ratio of 4.06
          where a side-on car sits between 1.8 and 2.6.

        Containment alone is *not* enough to call two boxes the same object.
        This camera looks along a street, so a distant car frequently sits
        inside the bounding box of a near one while being an entirely different
        vehicle. A first version suppressed on containment alone and deleted 27
        real vehicles from one afternoon. So a box is only dropped when all
        three hold: it is smaller than its container, mostly inside it, and
        markedly less confident. The genuine ghost scored 0.36 against its
        parent's 0.94; a real far car scored 0.87 against 0.95.
        """
        cfg = self.cfg
        contain_min = float(getattr(cfg, "nested_contained_min", 0.80))
        conf_margin = float(getattr(cfg, "nested_conf_margin", 0.25))
        aspect_max = float(getattr(cfg, "aspect_max", 3.5))
        aspect_min_conf = float(getattr(cfg, "aspect_min_conf", 0.60))

        kept: list[Detection] = []
        for d in sorted(dets, key=lambda x: -x.conf):
            if d.cls_id in self.vehicle_classes:
                w, h = d.wh
                aspect = w / max(h, 1)
                # A generous ceiling: only reject a freakish shape when the
                # model is also unsure. A confident long box may be a bus.
                if aspect > aspect_max and d.conf < aspect_min_conf:
                    self.suppressed_aspect += 1
                    continue
                nested = any(
                    k.cls_id in self.vehicle_classes
                    and _area(d.xyxy) < _area(k.xyxy)
                    and d.conf <= k.conf - conf_margin
                    and _contained_frac(d.xyxy, k.xyxy) >= contain_min
                    for k in kept
                )
                if nested:
                    self.suppressed_nested += 1
                    continue
            kept.append(d)
        # Restore detection order so downstream output stays stable.
        order = {id(d): i for i, d in enumerate(dets)}
        kept.sort(key=lambda d: order[id(d)])
        return kept

    def reset_tracker(self) -> None:
        """Drop tracker state — call after a camera reconnect or long stall."""
        try:
            for t in getattr(self.model.predictor, "trackers", []) or []:
                t.reset()
            log.info("tracker state reset")
        except Exception as exc:
            log.debug("tracker reset skipped: %s", exc)


def color_for(name: str) -> tuple[int, int, int]:
    return _COLORS.get(name, _DEFAULT_COLOR)


def draw_detections(frame: np.ndarray, dets: list[Detection],
                    flagged_ids: set[int] | None = None) -> np.ndarray:
    """Draw boxes in place. Triage-flagged tracks get a thick magenta box."""
    import cv2

    flagged_ids = flagged_ids or set()
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        hit = d.track_id is not None and d.track_id in flagged_ids
        color = (255, 0, 255) if hit else color_for(d.name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if hit else 2)

        label = f"{d.name} {d.conf:.2f}"
        if d.track_id is not None:
            label = f"#{d.track_id} " + label
        if hit:
            label += " CANDIDATE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
