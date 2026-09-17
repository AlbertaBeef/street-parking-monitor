# street-parking-monitor

A fixed USB camera watching a residential street, with two goals:

1. **Parking spaces** — where along the curb there is room to park, when gaps
   open, and how long vehicles stay. There are no painted bays on this street:
   cars pack in wherever they fit, so parking is modelled as *intervals on a
   line* and free space as the *gaps between them*.
2. **The parking police** — detect the municipal enforcement vehicle (white body,
   blue city markings, sometimes roof/side cameras), and learn *when* it patrols.

**Phase 1, the current phase, is data collection.** There is no enforcement-vehicle
classifier yet, because there are no confirmed examples of one yet. Everything
here exists to capture the raw material and surface the handful of frames worth
a human's attention.

## Hardware

| | |
|---|---|
| Host | AMD Ryzen AI MAX+ 395 "Strix Halo", 124 GB RAM |
| Camera | Logitech BRIO, 3840×2160 MJPG @ 30 fps, through window glass |
| Inference | YOLO11m on the Radeon 8060S iGPU via ROCm — **20 ms/frame** (same at 1080p or 4K) |
| Video encode | `h264_vaapi` on the same iGPU — **280 fps**, 35× smaller than JPEG |
| Storage | 17 TB free; nothing is auto-deleted |

The camera's field of view is already at maximum. All 16:9 modes (720p through
4K) cover an identical scene — higher resolution buys detail, not coverage —
and the 4:3 modes are horizontally cropped, not vertically taller. `zoom_absolute`
sits at its minimum of 100. **The only way to see more street is to aim lower.**

## Quick start

```bash
# 1. Aim the camera and fix focus/exposure (important: shooting through glass)
DISPLAY=:1 ~/aiml-env/bin/python tools/aim.py --detect

# 2. Collect — the curb fits itself from the first parked cars it sees

~/aiml-env/bin/python collect.py                  # until Ctrl-C
~/aiml-env/bin/python collect.py --duration 3600  # one hour
~/aiml-env/bin/python collect.py --dry-run --duration 60   # exercise, write nothing

# 3. Only if the auto-fitted curb looks wrong
~/aiml-env/bin/python tools/curb_editor.py --latest
```

Everything runs from `~/aiml-env`. See `requirements.txt` for a fresh host.

## What gets recorded

Four streams, each answering a different question:

| Stream | Cadence | Purpose |
|---|---|---|
| `timelapse/` | 60 s active, 5 min idle | occupancy over time |
| `logs/curb.jsonl` | every 30 s | occupied intervals + gaps, in metres |
| `events/*.mp4` | H.264 burst on motion | arrivals, departures, pass-bys, people on foot |
| `crops/vehicle/` | ≤8 per track | the labelling pool for goal 2 |
| `crops/person/` | ≤8 per track | officers work on foot too |
| `logs/tracks.jsonl` | one per finished track | path, speed, direction, stops |

**The JSONL logs are the dataset; the pixels are evidence.** Occupancy, speeds,
directions and timings all live in a few tens of MB/day of text. The images
exist so a human — or a future classifier — can verify.

### Session layout

```
data/<YYYY-MM-DD>/session_<HHMMSS>/
    meta.json                 config, host, camera, final counts
    timelapse/                periodic full frames
    events/evt_000123.mp4     one clip per event (JPEG dir if ffmpeg absent)
    crops/vehicle/            trk000042_03_car_20260917_130729_360.jpg
    candidates/police/        copies of colour-flagged crops
    logs/detections.jsonl     every detection sample (4 Hz)
    logs/tracks.jsonl         per-track summaries  <- the analysis dataset
    logs/curb.jsonl           occupied intervals and gaps, in metres
    logs/curb_events.jsonl    a parkable gap opened or closed
    logs/events.jsonl         event index
    logs/crops.jsonl          crop index with triage scores
    collect.log
```

Filenames are `YYYYmmdd_HHMMSS_mmm`, so lexical order is time order.

## Measuring parking space

There are no painted bays, so "is spot 3 free?" is not a well-posed question.
The useful question is *where along the curb is there a gap, and will a car fit
in it?* So the curb is a polyline, each parked car is an interval on its arc
length, and free space is the complement.

**The curb fits itself.** Parked cars sit on the curb by definition, so their
ground-contact points *are* the curb. `autofit_curbs()` splits those points into
a near and a far group by the largest vertical gap between them and fits a
total-least-squares line through each — 18 px max residual on the first real
frame. The result is written to `config/curb.yaml`; delete that file to re-fit.

**The scale calibrates itself.** A passenger car is about 4.5 m, so every
vehicle the camera sees is a ruler laid down at a known point on the curb.
Collecting `(position, pixel-length)` pairs and taking a robust median per bin
gives a pixels-per-metre function that varies along the line exactly as
perspective requires — no tape measure, no survey marks, and it re-derives
itself if the camera is nudged. Samples accumulate across sessions in
`config/curb_calibration.json`, so accuracy improves with time.

Only *stationary* vehicles occupy curb space; a car driving past must not be
allowed to erase a real gap. Gaps that touch the edge of the frame are reported
but marked `open`, never measured — we cannot see where they end.

```
[far curb]  15.2 m visible   12.7 m occupied   2.5 m free
  gap  1.0 -> 1.4    0.4 m   too small
  gap  6.0 -> 7.4    1.4 m   too small
  gap 11.8 -> 12.5   0.7 m   too small
```

Known limits: pixel length approximates real length only while vehicles are
side-on, which holds for a street crossing the view; a clipped vehicle is
rejected as a calibration sample because it is not a whole car; and the camera
sees roughly **15 m of far curb, about three car lengths**, so the system can
report "the visible stretch is full" but cannot survey the block.

## Finding the parking police

The enforcement vehicle is a **rare class with zero examples so far**, so phase 1
is optimised for finding needles, not classifying them. Two *independent* triage
channels flag candidates; either alone yields too many false positives to review
by hand, but agreement between them is a strong signal.

**Colour** (`streetmon/triage.py`) — white body plus saturated blue markings,
measured as pixel fractions in HSV on each crop. A triage aid, not a classifier:
expect false positives from any white car near something blue. Flagged crops are
copied to `candidates/police/`.

**Kinematics** (`streetmon/tracks.py`) — what the vehicle *does*. Enforcement
vehicles patrol at a crawl, pause beside parked cars and sometimes reverse;
ordinary traffic crosses the frame at road speed and never stops mid-street.
This works regardless of livery, lighting or weather.

Speed is measured in **body-lengths per second**, not pixels per second. A car
near the lens sweeps far more pixels than an identical car across the street, so
raw pixel speed mostly encodes distance. Dividing by the vehicle's own box width
cancels that and yields a physically meaningful quantity needing no scene
calibration:

| Behaviour | BL/s | Flagged |
|---|---|---|
| parked car | 0.01 | no |
| through traffic, ~35 km/h | 2.19 | no |
| **patrol crawl, ~10 km/h** | **0.53** | **yes** |
| **patrol with mid-street stops** | **0.28** | **yes** |

This assumes box width ≈ vehicle length, which holds while traffic crosses the
frame side-on — true for this camera, but not for one pointed along a road.

### The road from here

- **Stage 1 (now)** — collect; review the candidate queue with `tools/review.py`
  and label by hand. Target 10–20 confirmed passes.

  The unit of review is a **track**, not a crop: one vehicle visit, one decision,
  with every crop of that vehicle shown side by side so a bad angle in one frame
  does not cost the sighting. `v` plays the event clip it came from. Labels
  append to `labels/labels.jsonl` — never overwritten, so relabelling is safe and
  the history stays auditable.

  On the first afternoon of collection the queue ran at **2 candidates out of
  102 vehicle tracks** (~2%), which is the point: the triage exists to make the
  review small enough that a human will actually do it.
- **Stage 2** — few-shot identity via embeddings + nearest neighbour, *not*
  fine-tuning: 20 positives will overfit any detector you train.
- **Stage 3** — per-sighting records: time, direction, speed, dwell, return pass.
- **Stage 4** — the payoff: presence alerts, sweep direction, and an enforcement
  schedule heatmap by day-of-week × hour.

## Configuration

`config/site.yaml` is hand-editable and may be partial; anything omitted falls
back to `streetmon/config.py:DEFAULTS`. Parking polygons live separately in
`config/curb.yaml` because a tool generates that file.

The knobs that matter most:

| Key | Default | Notes |
|---|---|---|
| `schedule.active` | weekdays 07:00–19:00 | **set to the parking sign's hours** |
| `capture.detect_fps` | 4.0 | detector sample rate |
| `capture.event.quality` | 24 | H.264 qp; lower = better, larger |
| `kinematic.patrol_max_bl_s` | 1.0 | ~16 km/h for a 4.5 m car |
| `triage.white_min_frac` | 0.18 | white body fraction |
| `triage.red_band_min_frac` | 0.012 | the livery stripe — the decisive feature |
| `curb.min_gap_m` | 5.4 | a 4.5 m car plus room to manoeuvre |
| `capture.event.width` | 1920 | event clip width; 0 = full capture resolution |

### Resolution

Capture is 4K; the streams are written at the resolution each one needs.
Vehicle crops keep the full 3840 px (≈1870 px per car, 3.3× the linear detail of
1080p) because distinguishing fleet units — `S3-24-04` from `S3-24-15`, two
visually identical Rogues — is a fine-detail problem. Event video and timelapse
are downscaled to 1920, where the extra pixels buy nothing.

Pixel thresholds in the config (`min_box_px`, `max_perp_px`) are written for a
1920-wide frame and scaled automatically, so changing capture resolution does
not silently retune the filters. Everything else — triage fractions, speeds in
body-lengths/sec — is scale-invariant by construction. Curb calibration files
record the frame width they were measured at and rescale on load; without that,
reusing a 1080p calibration at 4K would report every distance at double its
true length while looking perfectly healthy.
| `storage.retention_days` | 0 | 0 = never auto-delete |

## Tools

| Tool | Purpose |
|---|---|
| `tools/aim.py` | live preview; focus/exposure tuning; prints YAML to paste |
| `tools/curb_editor.py` | review/adjust the auto-fitted curb lines |
| `tools/review.py` | page through candidates and label them — the stage-1 workhorse |

## What is not in this repository

Source and documentation only. Four things live alongside it but are
deliberately untracked (see `.gitignore`):

| Path | Why |
|---|---|
| `data/` | collected imagery and logs — large, and of a public street |
| `captures-manual/` | original photos of the enforcement fleet — irreplaceable, back up separately |
| `reference/`, `labels/` | labelled crops and review decisions, derived from the above |
| `config/curb.yaml`, `config/curb_calibration.json` | site-specific, and both regenerate themselves |

Consequences for a fresh clone: the curb geometry and pixels-per-metre scale
re-derive on their own within a few hours of traffic, so the collector runs
unattended out of the box. The triage thresholds in `config/site.yaml` are
already tuned and travel with the repo, but the reference crops they were
measured against do not — re-validating a threshold change needs that local
data.

## A note on what this records

The camera points at a public street from a private window, so pedestrians,
licence plates and visitors' vehicles land in the dataset. It all stays on this
machine — nothing is uploaded — and `data/` is gitignored. Worth keeping in mind
before sharing clips or moving the archive anywhere else.
