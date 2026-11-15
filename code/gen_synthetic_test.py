#!/usr/bin/env python3
"""Generate SYNTHETIC tracks/endpoints/phases CSVs for testing overlay_demo.py.

The points follow the Checkpoint 1 schematic layout (horse facing left) with
simple sinusoidal motion. Endpoint values are computed from the synthetic
geometry with the Checkpoint 1 sign conventions. These files are test data
only - they are NOT model outputs and must never be reported as results.

Usage:
  python gen_synthetic_test.py --video goat.mp4 --outdir synthetic/
"""

import argparse
import csv
import math
import os

import cv2

# Schematic layout as (x, y) fractions of frame size, per Checkpoint 1 diagram.
LAYOUT = {
    "1": (0.160, 0.310), "2": (0.200, 0.290), "3": (0.235, 0.200),
    "4": (0.245, 0.240), "5": (0.265, 0.360), "6": (0.310, 0.420),
    "7": (0.345, 0.500), "8": (0.490, 0.480), "9": (0.600, 0.490),
    "10": (0.635, 0.480), "11": (0.335, 0.590), "12": (0.490, 0.670),
    "13": (0.360, 0.610), "14": (0.365, 0.770), "15": (0.368, 0.890),
    "16": (0.369, 0.945), "17": (0.370, 0.970), "18": (0.575, 0.585),
    "19": (0.565, 0.750), "20": (0.575, 0.860), "21": (0.578, 0.930),
    "22": (0.580, 0.965), "23": (0.395, 0.890), "24": (0.397, 0.965),
    "D1": (0.760, 0.600), "D2": (0.925, 0.600),
    "D3": (0.760, 0.760), "D4": (0.925, 0.760),
}
HEAD = {"1", "2", "3", "4", "5", "6"}


def phase_of(frame, n_frames):
    q = frame / max(n_frames, 1)
    if q < 1 / 3:
        return "approach"
    if q < 2 / 3:
        return "mount"
    if q < 0.9:
        return "thrust"
    return "dismount"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    os.makedirs(args.outdir, exist_ok=True)

    tracks_path = os.path.join(args.outdir, "tracks.csv")
    endpoints_path = os.path.join(args.outdir, "endpoints.csv")
    phases_path = os.path.join(args.outdir, "phases.csv")

    with open(tracks_path, "w", newline="") as ft, \
            open(endpoints_path, "w", newline="") as fe, \
            open(phases_path, "w", newline="") as fp:
        wt = csv.writer(ft)
        wt.writerow(["frame", "keypoint", "x", "y", "visible"])
        we = csv.writer(fe)
        we.writerow(["frame", "theta_back_deg", "theta_neck_deg",
                     "hind_norm", "chest_norm"])
        wp = csv.writer(fp)
        wp.writerow(["frame", "phase"])

        for f in range(n):
            dx = 30 * math.sin(f / 20.0)
            dy = 4 * math.sin(f / 9.0)
            pts = {}
            for kid, (fx, fy) in LAYOUT.items():
                x, y = fx * w + dx, fy * h + dy
                if kid in HEAD:
                    y += 6 * math.sin(f / 11.0 + 1)
                if kid == "22":  # hind hoof swings -> hind_norm varies
                    x += 18 * math.sin(f / 6.0)
                pts[kid] = (x, y)
                # point 15 is "occluded" for a stretch, to exercise the
                # visibility rule (missing, never interpolated)
                visible = 0 if (kid == "15" and 40 <= f < 50) else 1
                wt.writerow([f, kid, f"{x:.1f}", f"{y:.1f}", visible])

            wx, wy = pts["7"]
            cx, cy = pts["10"]
            px, py = pts["4"]
            pelvis_x = pts["18"][0]
            hoof_x = pts["22"][0]
            chest_x = pts["11"][0]
            d1x = pts["D1"][0]
            d2x = pts["D2"][0]
            l_torso = math.hypot(cx - wx, cy - wy)
            l_dummy = abs(d2x - d1x)
            theta_back = math.degrees(math.atan2(-(cy - wy), cx - wx))
            a = math.degrees(math.atan2(-(wy - py), wx - px))
            b = math.degrees(math.atan2(-(cy - wy), cx - wx))
            theta_neck = (a - b + 180) % 360 - 180
            hind_norm = (pelvis_x - hoof_x) / l_torso   # horse faces left
            chest_norm = (chest_x - d1x) / l_dummy
            we.writerow([f, f"{theta_back:.2f}", f"{theta_neck:.2f}",
                         f"{hind_norm:.3f}", f"{chest_norm:.3f}"])
            wp.writerow([f, phase_of(f, n)])

    print(f"synthetic test data for {n} frames -> {args.outdir}")


if __name__ == "__main__":
    main()
