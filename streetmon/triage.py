"""Colour prefilter for the Ville de Québec parking-enforcement vehicles.

Retuned against real photographs of the actual vehicles (see
`captures-manual/parking-police/` and `reference/`). The first version of this
file looked for "white body + blue markings", which was wrong in a way that
would have made it useless: measured on the real vehicles, the blue Ville de
Québec crest is small and dark and scores **blue_frac = 0.000** at street
scale, while genuinely blue civilian cars score up to 0.58. Requiring blue
would have rejected every true positive and admitted the false ones.

What actually identifies them is a **dashed red stripe along the mid-body**,
present on both vehicle types in the fleet:

  * white Honda Civic sedan — "VILLE DE QUÉBEC" crest, "STATIONNEMENT"
  * white Nissan Rogue SUV  — roof ALPR cameras, rear scanner arms

Measured separation on 10 confirmed positives vs 38 negatives (including a
silver Toyota photographed *with its brake lights lit*, and 32 crops of
ordinary traffic from this camera):

    white_frac   police 0.24-0.47   ordinary traffic median 0.07
    red_band     police 0.017-0.042 silver Toyota 0.000, traffic median 0.000

Restricting the red test to the middle rows of the crop is what makes it
robust: brake lights, tail lights and roof-line sky all fall outside that band.
The chosen thresholds sit mid-plateau of a region that gave 100% recall at zero
false positives, so neither is on a cliff edge.

This is still a *prefilter*, not a classifier. 38 negatives from one street on
one afternoon is a small sample; expect red delivery vans, taxis and Canada
Post to trip it. Its job is to keep the review queue short enough to read.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TriageResult:
    white_frac: float
    red_frac: float
    red_band_frac: float
    red_dashes: int
    blue_frac: float
    is_white_vehicle: bool
    is_police_candidate: bool

    def to_json(self) -> dict:
        # Every feature is recorded, not just the ones currently gated on, so
        # thresholds can be re-tuned later without re-collecting anything.
        return {
            "wf": round(self.white_frac, 4),
            "rf": round(self.red_frac, 4),
            "rb": round(self.red_band_frac, 4),
            "rd": self.red_dashes,
            "bf": round(self.blue_frac, 4),
            "white": self.is_white_vehicle,
            "cand": self.is_police_candidate,
        }


def _masks(crop: np.ndarray, cfg, size: int = 96):
    """Pixel fractions at a fixed size, so cost is independent of crop scale."""
    sm = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    white = (s <= int(cfg.white_sat_max)) & (v >= int(cfg.white_val_min))

    # Red wraps the hue circle, so it needs both ends of the 0..179 range.
    lo, hi = cfg.red_hue_lo, cfg.red_hue_hi
    red = (((h <= int(lo)) | (h >= int(hi)))
           & (s >= int(cfg.red_sat_min)) & (v >= int(cfg.red_val_min)))

    lo_b, hi_b = cfg.blue_hue
    blue = ((h >= int(lo_b)) & (h <= int(hi_b))
            & (s >= int(cfg.blue_sat_min)) & (v >= int(cfg.blue_val_min)))

    top = int(size * float(cfg.band_top))
    bot = int(size * float(cfg.band_bottom))
    return white, red, blue, red[top:bot, :]


def count_dashes(band: np.ndarray, cfg) -> int:
    """How many separate red marks sit in a row across the band.

    The livery is a *dashed* stripe: several similar marks strung along the
    body. A brake or tail light is one blob. Counting components is what tells
    them apart — and it is the difference between flagging the enforcement
    fleet and flagging every white hatchback that happens to be braking.
    """
    if band is None or band.size == 0 or not band.any():
        return 0
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        band.astype(np.uint8), connectivity=8)
    h, w = band.shape
    min_area = max(2, int(float(cfg.dash_min_area_frac) * h * w))
    max_area = max(min_area + 1, int(float(cfg.dash_max_area_frac) * h * w))
    keep = []
    for i in range(1, n):                       # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep.append(float(centroids[i][1]))  # row of the centroid
    if len(keep) < 2:
        return len(keep)
    # Dashes lie along one band; scattered specks at different heights are
    # noise or trim, not a stripe.
    med = float(np.median(keep))
    tol = max(2.0, float(cfg.dash_row_tol_frac) * h)
    return sum(1 for cy in keep if abs(cy - med) <= tol)


def analyse(crop: np.ndarray, cfg) -> TriageResult:
    """Score one vehicle crop for the enforcement-fleet colour signature."""
    if crop is None or crop.size == 0:
        return TriageResult(0.0, 0.0, 0.0, 0, 0.0, False, False)

    white, red, blue, band = _masks(crop, cfg)
    white_frac = float(white.mean())
    red_frac = float(red.mean())
    band_frac = float(band.mean()) if band.size else 0.0
    blue_frac = float(blue.mean())
    dashes = count_dashes(band, cfg)

    is_white = white_frac >= float(cfg.white_min_frac)
    is_candidate = (is_white
                    and band_frac >= float(cfg.red_band_min_frac)
                    and dashes >= int(cfg.red_min_dashes))
    return TriageResult(white_frac, red_frac, band_frac, dashes, blue_frac,
                        is_white, is_candidate)


def debug_masks(crop: np.ndarray, cfg) -> np.ndarray:
    """crop | white | red | red-band, for threshold tuning by eye."""
    size = 192
    sm = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    white, red, _blue, _band = _masks(crop, cfg, size=size)

    def tint(mask, color):
        out = np.zeros((size, size, 3), np.uint8)
        out[mask] = color
        return out

    banded = np.zeros((size, size, 3), np.uint8)
    top = int(size * float(cfg.band_top))
    bot = int(size * float(cfg.band_bottom))
    banded[top:bot][red[top:bot]] = (0, 0, 255)
    cv2.rectangle(banded, (0, top), (size - 1, bot), (90, 90, 90), 1)
    return np.hstack([sm, tint(white, (255, 255, 255)),
                      tint(red, (0, 0, 255)), banded])
