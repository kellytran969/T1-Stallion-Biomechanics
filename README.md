# T1: Stallion Biomechanics (CPSC 4430/6430, Checkpoint 2)

PosePAL (test-time optimization inside a frozen CoTracker3 point tracker)
applied to stallion biomechanics: dense tracks for 28 custom landmarks
(24 horse + 4 dummy) → four image-plane endpoint measurements → breeding
phase predictions, with held-out evaluation.

**Group:** Jim Daou, Stephen Gancayco, Kelly Tran, Samuel Whitney
**Repository:** https://github.com/kellytran969/T1-Stallion-Biomechanics
**Submitted revision:** TODO, commit hash or release tag (must match the
code in the submission ZIP)

## Pipeline

1. Sparse manual annotations (frame 0 + every ~10th frame):
   `annotation/ANNOTATION_GUIDE.md`
2. Frozen CoTracker3 baseline tracks, TODO (Stephen): command
3. PosePAL TTO adaptation (qp_emb only; settings: TODO state steps / lr
   actually used), TODO (Stephen): command
4. Endpoint measurements from tracks, TODO (Sam): command
5. Phase prediction (rule-based + random forest), TODO (Jim): command
6. Held-out evaluation on blind scoring frames, TODO (Jim): command
7. Demo video render, `code/overlay_demo.py` (working, see below)

## Setup

```bash
git clone https://github.com/Zhuoyang-Pan/PosePAL.git   # authors' code, commit 63572cd
# CoTracker3 weights (97 MB):
#   https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
pip install -r code/requirements.txt
```

## Data

- Stallion clips: TODO, source, clip IDs (clip-01 …), horse IDs, counts.
  Split by horse (otherwise by whole clip); split manifest: TODO path.
- Paper datasets for the tracking reproduction: DAVIS-Animals,
  DeepFly3D (links in the Checkpoint 1 report).
- Annotations: one `<clip>_annotations.csv` (frame 0 + corrections) and
  one `<clip>_scoring.csv` (blind frames 5/15/25…, never used in TTO)
  per clip. Formats in `annotation/ANNOTATION_GUIDE.md`.

## Demo render (working)

```bash
python code/overlay_demo.py --video clip-01.mp4 \
    --tracks clip-01_tracks_adapted.csv \
    --endpoints clip-01_endpoints.csv \
    --phases clip-01_phases.csv \
    --landmarks annotation/landmarks.csv \
    --out demo_clip-01.mp4
```

Renderer self-test (synthetic data on the PosePAL sample video; proves
the overlay end to end, not a result):

```bash
python code/gen_synthetic_test.py --video goat.mp4 --outdir synthetic/
python code/overlay_demo.py --video goat.mp4 --tracks synthetic/tracks.csv \
    --endpoints synthetic/endpoints.csv --phases synthetic/phases.csv \
    --landmarks annotation/landmarks.csv --tag "SYNTHETIC TEST" \
    --out demo_preview.mp4
```

## Expected outputs

- Per clip: tracks (frozen + adapted), endpoints, phases CSVs
- `results/`, TODO: metric tables (δ_avg, J, J_masked; endpoint errors;
  phase macro-F1)
- `demo_<clip>.mp4`, processed video with landmarks, measurements, phases
- Execution screen recording, TODO: file name

## Hardware and runtime

- Paper reference: ~3 min per 100-frame 480p clip on an NVIDIA RTX A6000.
- This group: TODO, machine/GPU actually used, runtime per clip.

## Known limitations

- Landmark names/indices pending final confirmation against the DLC CSV
  columns (roles in `annotation/landmarks.csv` follow the Checkpoint 1
  schematic).
- Endpoints are image-plane quantities (no 3D / physical units).
- Occluded landmarks are reported missing, never interpolated; endpoint
  coverage therefore varies by phase.
- TODO, clip count / annotation budget actually achieved; targets met
  or missed, honestly (Checkpoint 2 instructions require this).

## Contributions (Checkpoint 2)

| Member | Parts |
|---|---|
| Stephen Gancayco | Implementation lead: PosePAL runs (frozen + TTO), repo |
| Samuel Whitney | Data, split, endpoint measurements |
| Jim Daou | Evaluation, comparisons, phase prediction |
| Kelly Tran | Annotations, demo video, README/report package |
