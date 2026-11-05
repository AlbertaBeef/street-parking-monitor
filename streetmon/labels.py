"""Persistent review labels.

Append-only JSONL: a decision is never overwritten, only superseded, so a
change of mind stays auditable and the file can be concatenated across machines
without a merge. The last record for a (session, track) pair wins.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

POLICE = "parking_police"
NOT_POLICE = "not_police"
UNSURE = "unsure"
LABELS = (POLICE, NOT_POLICE, UNSURE)


def labels_path() -> Path:
    return REPO_ROOT / "labels" / "labels.jsonl"


def reviews_path() -> Path:
    return REPO_ROOT / "labels" / "reviews.jsonl"


@dataclass
class LabelStore:
    path: Path
    decisions: dict[tuple[str, int], dict]

    @classmethod
    def load(cls, path: Path | None = None) -> "LabelStore":
        path = Path(path or labels_path())
        decisions: dict[tuple[str, int], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decisions[(r["session"], int(r["track_id"]))] = r
        return cls(path=path, decisions=decisions)

    def get(self, session: str, track_id: int) -> str | None:
        rec = self.decisions.get((session, int(track_id)))
        return rec["label"] if rec else None

    def put(self, session: str, track_id: int, label: str,
            note: str = "", extra: dict | None = None) -> None:
        rec = {"session": session, "track_id": int(track_id), "label": label,
               "ts": round(time.time(), 3), "note": note}
        if extra:
            rec.update(extra)
        self.decisions[(session, int(track_id))] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in LABELS}
        for r in self.decisions.values():
            out[r["label"]] = out.get(r["label"], 0) + 1
        return out


@dataclass
class ReviewLog:
    """Which days have been examined in full, and what was found.

    Per-track labels cannot express the most common and most valuable outcome
    of a review session: "I looked at every track on this day and there was
    nothing." Recording that one negative per track is impractical -- a single
    weekday runs to a thousand tracks -- so without a day-level record,
    `examined and empty` is indistinguishable from `never looked at`.

    That distinction is the denominator. A sighting rate means nothing unless
    the hours it is divided by were actually inspected.
    """

    path: Path
    days: dict[str, dict]

    @classmethod
    def load(cls, path: Path | None = None) -> "ReviewLog":
        path = Path(path or reviews_path())
        days: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                days[r["day"]] = r          # last record for a day wins
        return cls(path=path, days=days)

    def get(self, day: str) -> dict | None:
        return self.days.get(day)

    def record(self, day: str, tracks_examined: int, positives: int,
               scope: str = "all", note: str = "") -> dict:
        rec = {"day": day, "ts": round(time.time(), 3),
               "reviewed": dt.datetime.now().isoformat(timespec="seconds"),
               "tracks_examined": int(tracks_examined),
               "positives": int(positives), "scope": scope, "note": note}
        self.days[day] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
        return rec

    def summary(self) -> list[dict]:
        return [self.days[d] for d in sorted(self.days)]
