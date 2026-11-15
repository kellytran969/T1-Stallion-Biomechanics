# Annotation Guide — T1 Checkpoint 2

How the group produces the sparse manual annotations that PosePAL's
test-time optimization (TTO) and the held-out scoring depend on.
Protocol per the Checkpoint 1 plan (which scored 100 — do not change it
without telling the whole group).

## What we annotate

- 28 landmarks per frame: 24 horse + 4 dummy (`annotation/landmarks.csv`).
  Landmark names/indices still need one confirmation pass against Sam's
  endpoint scheme / the DLC CSV columns before annotating real clips.
- One CSV per clip, copied from `annotation/annotations_template.csv`.
  Columns: `frame,keypoint,x,y,visible` — pixel coordinates in the
  original frame, `visible` = 1 or 0. This maps directly onto PosePAL's
  keypoint format (x, y, frame, id) in `main.py`.

## Which frames (per clip)

1. **Frame 0 (first frame): all 28 landmarks.** These are the query
   points PosePAL starts from. Take your time — every track depends on
   this frame.
2. **Correction frames: every ~10th frame (10, 20, 30, …), all 28
   landmarks.** These are fed to TTO as supervision.
3. **Blind scoring frames: the interleaved frames (5, 15, 25, …).**
   Annotate these in a SEPARATE file (`<clip>_scoring.csv`). They are
   the held-out ground truth for δ_avg / J_masked and must never be
   used as correction points. Mixing them into training is leakage.

## Rules

- **Occluded = invisible, never guessed.** If a landmark is hidden
  (behind the dummy, the other horse, a leg), set `visible` = 0 and
  leave x,y blank. Missing outputs are reported as missing; nothing is
  interpolated, in annotation, endpoints, or scoring.
- **Same annotator per clip** where possible; note the annotator in the
  clip's row of the split manifest.
- **Clips are named** with the Checkpoint 1 draft IDs: clip-01 … clip-05
  (plus horse ID once known). The demo must name its clip.
- Budget check (Checkpoint 1 commitment): at least 100 reference frames
  across at least 5 clips, correction and scoring frames counted
  separately; split by horse when possible, otherwise by whole clip.

## Files per clip

| File | Contents |
|---|---|
| `<clip>_annotations.csv` | Frame 0 + correction frames (TTO input) |
| `<clip>_scoring.csv` | Blind scoring frames only (evaluation input) |
| `<clip>_tracks_frozen.csv` | Frozen CoTracker3 output (Stephen) |
| `<clip>_tracks_adapted.csv` | PosePAL TTO output (Stephen) |
| `<clip>_endpoints.csv` | Per-frame endpoint measurements (Sam) |
| `<clip>_phases.csv` | Per-frame phase labels (Jim) |

Track/endpoint/phase CSV formats are the ones `code/overlay_demo.py`
reads, so the demo video renders straight from these files.
