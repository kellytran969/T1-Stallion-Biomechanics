#!/usr/bin/env python3
"""T1 Checkpoint 2 demo renderer.

Overlays tracked landmarks, biomechanics measurements, and phase predictions
on a video clip and writes an MP4. This is the processed-video deliverable for
the T1 demonstration: "processed video with tracked landmarks, biomechanics
measurements, and phase predictions".

Inputs (CSV, formats agreed by the group):
  tracks.csv     frame,keypoint,x,y,visible
                 keypoint = landmark id from landmarks.csv (1..24, D1..D4);
                 visible = 1/0. Invisible points are NOT drawn and are never
                 interpolated (Checkpoint 1 visibility rule).
  endpoints.csv  frame,theta_back_deg,theta_neck_deg,hind_norm,chest_norm
                 Blank cell = measurement missing that frame -> panel shows
                 "missing". Produced by the endpoints module (Sam).
  phases.csv     frame,phase     e.g. approach / mount / thrust / dismount.
                 Produced by the phase module (Jim). Optional.
  landmarks.csv  id,group,role   Role tags (poll, withers, croup, pelvis,
                 chest) drive the endpoint reference lines. Optional but
                 recommended.

Usage:
  python overlay_demo.py --video clip-01.mp4 --tracks tracks.csv \
      --endpoints endpoints.csv --phases phases.csv \
      --landmarks ../annotation/landmarks.csv --out demo_clip-01.mp4

  --tag "SYNTHETIC TEST" burns a label into the top-right corner (use it for
  any test render so test footage is never confused with real results).
"""

import argparse
import csv
import sys

import cv2

# BGR colors
HORSE_COLOR = (0, 165, 255)      # orange
DUMMY_COLOR = (255, 255, 0)      # cyan
BACK_LINE = (255, 0, 0)          # blue
NECK_LINE = (0, 200, 0)          # green
PELVIS_LINE = (0, 0, 255)        # red
DUMMY_LINE = (255, 0, 255)       # magenta
PANEL_BG = (0, 0, 0)
TEXT = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_tracks(path):
    tracks = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            visible = str(row.get("visible", "1")).strip() in ("1", "true", "True")
            tracks.setdefault(frame, {})[row["keypoint"].strip()] = (
                float(row["x"]), float(row["y"]), visible)
    return tracks


def load_endpoints(path):
    out = {}
    if not path:
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            def val(key):
                v = (row.get(key) or "").strip()
                return float(v) if v else None
            out[int(row["frame"])] = {
                "theta_back_deg": val("theta_back_deg"),
                "theta_neck_deg": val("theta_neck_deg"),
                "hind_norm": val("hind_norm"),
                "chest_norm": val("chest_norm"),
            }
    return out


def load_phases(path):
    out = {}
    if not path:
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["frame"])] = (row.get("phase") or "").strip()
    return out


def load_roles(path):
    """Return {role: keypoint_id} and {keypoint_id: group} from landmarks.csv."""
    roles, groups = {}, {}
    if not path:
        return roles, groups
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kid = (row.get("id") or "").strip()
            if not kid:
                continue
            groups[kid] = (row.get("group") or "horse").strip()
            role = (row.get("role") or "").strip()
            if role:
                roles[role] = kid
    return roles, groups


def fmt(value, digits):
    return "missing" if value is None else f"{value:.{digits}f}"


def draw_frame(frame_img, points, roles, groups, endpoint, phase, frame_idx, tag):
    h, w = frame_img.shape[:2]

    def pt(role):
        kid = roles.get(role)
        if kid and kid in points and points[kid][2]:
            return (int(points[kid][0]), int(points[kid][1]))
        return None

    # --- endpoint reference lines (only when both ends are visible) ---
    withers, croup, poll = pt("withers"), pt("croup"), pt("poll")
    pelvis, chest = pt("pelvis"), pt("chest")
    if withers and croup:
        cv2.line(frame_img, withers, croup, BACK_LINE, 2)
    if poll and withers:
        cv2.line(frame_img, poll, withers, NECK_LINE, 2)
    if pelvis:
        cv2.line(frame_img, pelvis, (pelvis[0], h - 5), PELVIS_LINE, 1)
    d1, d2 = pt("dummy_D1"), pt("dummy_D2")
    if d1 and d2:
        cv2.line(frame_img, d1, d2, DUMMY_LINE, 2)

    # --- landmark dots (invisible points are skipped, never interpolated) ---
    for kid, (x, y, visible) in points.items():
        if not visible:
            continue
        cx, cy = int(x), int(y)
        if groups.get(kid) == "dummy":
            cv2.rectangle(frame_img, (cx - 5, cy - 5), (cx + 5, cy + 5),
                          DUMMY_COLOR, -1)
        else:
            cv2.circle(frame_img, (cx, cy), 5, HORSE_COLOR, -1)
            cv2.circle(frame_img, (cx, cy), 5, TEXT, 1)

    # --- measurement panel (top-left) ---
    lines = [f"frame {frame_idx}"]
    if endpoint is not None:
        lines.append(f"back angle:  {fmt(endpoint['theta_back_deg'], 1)} deg")
        lines.append(f"neck angle:  {fmt(endpoint['theta_neck_deg'], 1)} deg")
        lines.append(f"hind/pelvis: {fmt(endpoint['hind_norm'], 2)}")
        lines.append(f"chest/dummy: {fmt(endpoint['chest_norm'], 2)}")
    panel_h = 28 + 22 * len(lines)
    overlay = frame_img.copy()
    cv2.rectangle(overlay, (8, 8), (300, panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)
    for i, line in enumerate(lines):
        cv2.putText(frame_img, line, (16, 34 + 22 * i), FONT, 0.55, TEXT, 1,
                    cv2.LINE_AA)

    # --- phase banner (bottom center) ---
    if phase:
        label = f"PHASE: {phase.upper()}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        x0 = (w - tw) // 2
        cv2.rectangle(frame_img, (x0 - 12, h - th - 26), (x0 + tw + 12, h - 8),
                      PANEL_BG, -1)
        cv2.putText(frame_img, label, (x0, h - 16), FONT, 0.8, TEXT, 2,
                    cv2.LINE_AA)

    if tag:
        cv2.putText(frame_img, tag, (w - 260, 34), FONT, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return frame_img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--endpoints")
    ap.add_argument("--phases")
    ap.add_argument("--landmarks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tracks = load_tracks(args.tracks)
    endpoints = load_endpoints(args.endpoints)
    phases = load_phases(args.phases)
    roles, groups = load_roles(args.landmarks)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"cannot write output: {args.out}")

    idx = 0
    while True:
        ok, frame_img = cap.read()
        if not ok:
            break
        draw_frame(frame_img, tracks.get(idx, {}), roles, groups,
                   endpoints.get(idx), phases.get(idx, ""), idx, args.tag)
        writer.write(frame_img)
        idx += 1
    cap.release()
    writer.release()
    print(f"wrote {args.out}: {idx} frames, {width}x{height} @ {fps:g} fps; "
          f"tracks for {len(tracks)} frames")


if __name__ == "__main__":
    main()
