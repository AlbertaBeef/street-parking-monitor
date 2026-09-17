"""Active-hours gating.

Parking enforcement here runs weekday daytime, so the collector does not need
to work equally hard at 3 a.m. on a Sunday. Outside active hours it keeps a
sparse timelapse — enough to see cars arrive and leave overnight, which still
matters for goal 1 — and stops spending disk on event clips and crops.

Windows are local time and may span midnight (`22:00`-`06:00` works). Set
`enabled: false` to collect flat-out around the clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)

_DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _parse_hhmm(text: str) -> int:
    """'07:30' -> minutes since midnight."""
    hh, _, mm = str(text).partition(":")
    return int(hh) * 60 + int(mm or 0)


@dataclass
class Window:
    days: set[int]
    start_min: int
    end_min: int
    label: str = ""

    def contains(self, when: datetime) -> bool:
        minute = when.hour * 60 + when.minute
        if self.start_min <= self.end_min:
            in_time = self.start_min <= minute < self.end_min
            day = when.weekday()
        else:
            # Spans midnight: after start today, or before end tomorrow — in
            # which case the day that must match is yesterday's.
            if minute >= self.start_min:
                in_time, day = True, when.weekday()
            elif minute < self.end_min:
                in_time, day = True, (when.weekday() - 1) % 7
            else:
                in_time, day = False, when.weekday()
        return in_time and day in self.days


class Schedule:
    def __init__(self, cfg):
        self.enabled = bool(cfg.enabled)
        self.windows: list[Window] = []
        for w in (cfg.active or []):
            days = {_DAYS[d.lower()[:3]] for d in w.get("days", list(_DAYS))}
            self.windows.append(Window(days, _parse_hhmm(w["start"]),
                                       _parse_hhmm(w["end"]),
                                       w.get("label", "")))
        self.idle = cfg.idle
        if self.enabled and self.windows:
            log.info("schedule: %d active window(s); idle timelapse every %ss",
                     len(self.windows), self.idle.timelapse_sec)
        elif not self.enabled:
            log.info("schedule disabled — collecting at full rate continuously")

    def is_active(self, when: datetime | None = None) -> bool:
        if not self.enabled or not self.windows:
            return True
        when = when or datetime.now()
        return any(w.contains(when) for w in self.windows)

    def describe(self, when: datetime | None = None) -> str:
        return "ACTIVE" if self.is_active(when) else "idle"
