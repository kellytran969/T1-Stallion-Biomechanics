# Endpoint Definitions

All measurements are calculated in image-plane coordinates.

## 1. Back Angle
- Start: Withers (landmark 7)
- End: Croup (landmark 10)
- Reference: Horizontal
- Positive direction: Croup above withers
- Output: Degrees

## 2. Neck Angle
- Start: Poll (landmark 4)
- End: Withers (landmark 7)
- Reference: Back-line direction
- Positive direction: Poll above back line
- Output: Degrees

## 3. Hind-Limb Position
- Reference: Pelvis (landmark 18)
- Measured point: Hind hoof (landmark 22)
- Normalization: Withers-to-croup distance
- Positive direction: Hoof forward of pelvis
- Output: Normalized horizontal offset

## 4. Chest Position
- Reference: Dummy D1
- Measured point: Chest (landmark 11)
- Normalization: Distance from Dummy D1 to Dummy D2
- Positive direction: Chest toward dummy
- Output: Normalized horizontal offset

## Missing Landmarks
If a required landmark is invisible or missing,
the corresponding endpoint must be missing.
Do not interpolate missing values.