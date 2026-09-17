"""Persistent review labels.

Append-only JSONL: a decision is never overwritten, only superseded, so a
change of mind stays auditable and the file can be concatenated across machines
without a merge. The last record for a (session, track) pair wins.
"""

from __future__ import annotations

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
