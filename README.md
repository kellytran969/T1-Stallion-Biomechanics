# T1: Stallion Biomechanics (CPSC 4430/6430, Checkpoint 2)

PosePAL (test-time optimization inside a frozen CoTracker3 point tracker)
applied to stallion biomechanics: dense tracks for 28 custom landmarks
(24 horse + 4 dummy), four image-plane endpoint measurements, and breeding
phase predictions, with held-out evaluation on blind scoring frames.

**Group:** Jim Daou, Stephen Gancayco, Kelly Tran, Samuel Whitney
**Repository:** https://github.com/kellytran969/T1-Stallion-Biomechanics
**Submitted revision:** 3a06f2a (update this line to the final packaging
commit if more commits land before the ZIP is built)

## Pipeline

1. **Sparse manual annotations.** 28 landmarks per annotated frame, one
   correction file and one blind scoring file per clip, interleaved so no
   scoring frame is ever a correction frame. Protocol:
   `annotation/ANNOTATION_GUIDE.md`.
2. **Tracking (frozen baseline + PosePAL TTO).**
   ```bash
   python code/run_posepal.py --video data/clip-07/clip-07.mp4 \
       --annotations annotation/clip-07_annotations.csv \
       --method frozen --out data/clip-07/Clip7frozen.csv \
       --endpoints-out data/clip-07/Clip7frozen_endpoints.csv
   python code/run_posepal.py --video data/clip-07/clip-07.mp4 \
       --annotations annotation/clip-07_annotations.csv \
       --method tto --out data/clip-07/Clip7tto.csv \
       --endpoints-out data/clip-07/Clip7tto_endpoints.csv
   ```
   TTO settings (from the report): Adam, 1,000 steps, learning rate
   decreasing linearly 0.001 to 0.00001, 4 tracking iterations, frozen
   baseline 6 iterations, visibility threshold 0.7. Only the per-keypoint
   appearance embeddings are optimized; CoTracker3 stays frozen.
3. **Endpoint measurements.** Computed from the tracks (see
   `data+endpoints/endpoint_definitions.md`): back angle and neck angle in
   degrees, hind position and chest position normalized by torso or dummy
   length. Output columns: `frame,back_angle,neck_angle,hind_position,
   chest_position`. A measurement is missing (blank) in any frame where a
   constituent landmark is not visible; nothing is interpolated.
4. **Phase prediction.** Random forests (pose-only and temporal variants)
   on the four endpoint features, plus a rule-based predictor
   (`code/phase_rules.json`). Labels are the per-frame phase annotations
   (`<clip>_phases.csv`: approach, mount, thrust, dismount; "other" is
   treated as unlabeled). Training uses clips 01, 02, 07 and validates on
   clip-03; clip-06 is fully held out.
   ```bash
   python code/phase_prediction.py train \
     --clip-01-endpoints data/clip-01/Clip1tto_endpoints.csv \
     --clip-02-endpoints data/clip-02/Clip2tto_endpoints.csv \
     --clip-07-endpoints data/clip-07/Clip7tto_endpoints.csv \
     --clip-03-endpoints data/clip-03/Clip3tto_endpoints.csv \
     --model-dir results/phase_models_tto --variant both
   python code/phase_prediction.py predict \
     --model results/phase_models_tto/pose.joblib \
     --endpoints data/clip-07/Clip7tto_endpoints.csv \
     --out results/clip07_pose_predictions.csv
   ```
5. **Held-out evaluation.** Tracking is scored against the blind scoring
   annotations (mean Euclidean error in pixels, delta_avg, on
   ground-truth-visible points). Phase predictions are scored against the
   ground-truth phase labels (accuracy and macro-F1).
6. **Demo video.** One renderer takes a clip's tracks, endpoints, and
   phase predictions and draws the landmarks, endpoint reference lines, a
   live measurement panel, and a phase banner:
   ```bash
   python code/overlay_demo.py --video data/clip-07/clip-07.mp4 \
       --tracks data/clip-07/Clip7tto.csv \
       --endpoints data/clip-07/Clip7tto_endpoints.csv \
       --phases demo/clip-07_phases_predicted.csv \
       --landmarks annotation/landmarks.csv \
       --out demo/clip-07_overlay.mp4
   ```

## Setup

```bash
git clone https://github.com/Zhuoyang-Pan/PosePAL.git   # authors' code, commit 63572cd
# CoTracker3 weights (97 MB):
#   https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
pip install -r code/requirements.txt
```

The saved phase models in `results/phase_models_tto/` were pickled with
scikit-learn 1.7.2; load them with that version (newer scikit-learn
releases fail to unpickle the imputer).

## Data

Five clips, five different stallions, split by horse. 234 manually
annotated reference frames in total (correction + blind scoring), plus
per-frame phase labels for every clip.

| Clip | Source | Size | Annotated frames |
|---|---|---|---|
| clip-01 | UNL MediaHub, "Stallion Collection" (segment t = 83 to 100 s) | 1280x720, 510 frames | 34 (17 + 17) |
| clip-02 | Candlewood Equine, YouTube JdhKsA-T8Lc (segment t = 54 to 70 s) | 640x480, 480 frames | 32 (16 + 16) |
| clip-03 | YouTube m15Yi0Wf52o, "Kyrymsun stallion Collection" | 640x360, 440 frames | 12 (6 + 6) |
| clip-06 | Team-provided clip, white stallion, full collection cycle | 1470x822, 1,384 frames | 93 (47 + 46) |
| clip-07 | Team-provided clip, palomino stallion, full collection cycle | 1720x1056, 941 frames | 63 (32 + 31) |

Clips 01 to 03 are side-profile segments that start already mounted.
Clips 06 and 07 are full clips containing approach, mount, thrust, and
dismount. A planned clip-05 was dropped (link only, no usable footage).
Video sources for clips 01 to 03 are also cited in the report.

## Results (verified against the blind scoring annotations)

**Tracking, delta_avg in pixels (lower is better). Pixel errors scale
with frame size, so compare frozen vs TTO within a clip, not across
clips:**

| Clip | Frozen | TTO | Notes |
|---|---|---|---|
| clip-01 (1280x720) | 36.9 (median 30.9) | 14.9 (median 10.4) | Side profile, mounted throughout; TTO cuts the mean error by 60%, coverage 90% to 95% |
| clip-02 (640x480) | 24.4 (median 22.6) | 13.6 (median 8.8) | TTO cuts the mean error by 44%; coverage 88% to 98% |
| clip-03 (640x360) | 40.2 (median 25.0) | 30.3 (median 9.9) | Smallest horse in frame of the five clips |
| clip-07 (1720x1056) | 99.2 (median 72.9) | 63.8 (median 31.7) | TTO also raises visible-point coverage from 23% to 94% |
| clip-06 (1470x822) | 165.2 | 181.5 (median 103.3) | Failure case: head, back, and dummy corners track well (11 to 65 px), but all leg keypoints are lost (290 to 450 px) after the camera cut at frame 90 and during the rear-view thrust |

On the three side-profile clips (01 to 03) and on clip-07, TTO improves
on the frozen baseline in both error and coverage. Clip-06 is the one
failure case, and it is the only clip with a mid-clip camera cut.

**Endpoints on clip-07 (from TTO tracks):** back and neck angles present
in 930 of 941 frames, chest position in 910, hind position in 836.

**Phase prediction:**
- Validation on clip-03 (the training script's validation split):
  macro-F1 0.27 for both random forest variants. Weak; the validation
  clip has only 5 dismount frames and no approach frames.
- Held-out clip-06 (never trained on), scored against ground-truth
  labels: rule-based 96.0% accuracy, macro-F1 0.59; pose random forest
  94.1%, macro-F1 0.52; temporal random forest 91.6%, macro-F1 0.43.
  Thrust is strong (F1 0.97 to 0.98); mount is the weakest class
  (F1 0 to 0.5) because it lasts only about 2 seconds per clip.
- The demo's clip-07 phase predictions are in-sample: clip-07 is one of
  the phase model's training clips, so its near-perfect agreement with
  the ground-truth labels is expected and is not held-out evidence.

**Known issue, label mapping:** the random forests' raw output labels
are rotated relative to the label names (a training-encoding
inconsistency). `code/phase_prediction.py` currently carries a hardcoded
remapping in `predict` that corrects the rotation for held-out clips;
the clip-06 prediction files in `results/` were regenerated with it and
verified above. The same remapping inverts the labels for clip-07,
which is in the training set, so the demo phases in
`demo/clip-07_phases_predicted.csv` are the model's raw output, verified
frame by frame against the ground-truth labels for this clip. The clean
fix is to retrain with a consistent label encoding; it is documented
here rather than hidden.

## Expected outputs

- Per clip: tracks (frozen + TTO), endpoints, ground-truth phases, and
  phase predictions, in `data/<clip>/` and `results/`
- `demo/clip-07_overlay.mp4`: processed clip-07 with tracked landmarks,
  live endpoint measurements, and phase banner
- Execution screen recording: TODO, file name (added at packaging)

## Hardware and runtime

- Paper reference: about 3 min per 100-frame 480p clip on an NVIDIA
  RTX A6000.
- This group: tracking runs were executed on a Windows workstation
  (Intel CPU, 64 GB RAM, NVIDIA GeForce RTX GPU). Exact GPU model and
  per-clip runtimes: TODO (Stephen).

## Known limitations

- Occlusion dominates. Across clips 01 to 03 the chest landmark is
  visible in only 2 of 78 annotated frames (the handler and the dummy
  block it in the mounted position), so the chest endpoint is reported
  only where it can actually be measured. The hind chain (pelvis, hock,
  fetlock, pastern, hoof) is the most reliable signal and is visible in
  nearly every frame of clips 06 and 07.
- Clips 06 and 07 film the thrust phase from behind the horse. Endpoint
  definitions are image-plane quantities defined for a side profile, so
  in rear views the back and neck angles describe the projected line,
  not the anatomical angle, and are near vertical. Treat rear-view angle
  values as pipeline outputs, not biomechanics.
- Clip-06 contains a camera cut (frame 90) and both clips 06 and 07 are
  screen recordings with player overlays baked into some frames.
- Five horses and 234 annotated frames is a small sample; cross-horse
  generalization is demonstrated only by the clip-06 held-out results.
- Occluded landmarks are reported missing, never interpolated, in
  annotation, endpoints, and scoring alike.

## Contributions (Checkpoint 2)

| Member | Parts |
|---|---|
| Stephen Gancayco | Implementation lead: PosePAL runs (frozen + TTO), repo |
| Samuel Whitney | Data, split, endpoint measurements |
| Jim Daou | Evaluation, comparisons, phase prediction |
| Kelly Tran | Annotations, phase labels, demo video, README/report package |
