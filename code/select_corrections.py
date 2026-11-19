#!/usr/bin/env python3
"""Select correction frames from a frozen track CSV, before annotation.

Scores are motion/visibility heuristics, not calibrated model uncertainty.
The output lists frames to label; no ground-truth coordinates are generated.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path


def read_tracks(path):
    data = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'frame', 'keypoint', 'x', 'y', 'visible'} <= set(reader.fieldnames):
            raise ValueError('Tracks need frame,keypoint,x,y,visible columns')
        for line, row in enumerate(reader, 2):
            try:
                frame, key = int(row['frame']), row['keypoint'].strip()
                x, y, visible = float(row['x']), float(row['y']), int(row['visible'])
                if frame < 0 or not key or visible not in (0, 1) or not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError('invalid value')
                if (frame, key) in data:
                    raise ValueError('duplicate frame/keypoint')
                data[frame, key] = (x, y, visible)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Invalid track row {line}: {exc}') from exc
    if not data:
        raise ValueError('Tracks file is empty')
    frames = sorted({f for f, _ in data})
    keys = sorted({k for _, k in data})
    if frames != list(range(frames[-1] + 1)) or len(data) != len(frames) * len(keys):
        raise ValueError('Tracks must contain every keypoint in every frame, starting at frame 0')
    return data, frames, keys


def excluded_frames(paths):
    excluded = {0}
    for path in paths:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'frame' not in reader.fieldnames:
                raise ValueError(f'{path}: expected a frame column')
            for row in reader:
                try:
                    f = int(row['frame'])
                    if f < 0:
                        raise ValueError('negative frame')
                    excluded.add(f)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}: invalid frame') from exc
    return excluded


def motion_scores(data, frames, keys):
    # Only compare coordinates across adjacent visible observations. A lost
    # visible point is flagged once, not repeatedly while it remains invisible.
    raw = {}
    speeds, accelerations, stops = [], [], []
    for f in frames:
        per_frame = []
        for key in keys:
            cur = data[f, key]
            prev = data.get((f - 1, key))
            prev2 = data.get((f - 2, key))
            lost = int(bool(prev and prev[2] and not cur[2]))
            speed = accel = stop = 0.0
            if prev and cur[2] and prev[2]:
                vx, vy = cur[0] - prev[0], cur[1] - prev[1]
                speed = math.hypot(vx, vy)
                if prev2 and prev2[2]:
                    pvx, pvy = prev[0] - prev2[0], prev[1] - prev2[1]
                    old_speed = math.hypot(pvx, pvy)
                    accel = math.hypot(vx - pvx, vy - pvy)
                    stop = max(0.0, old_speed - speed)
            per_frame.append((speed, accel, stop, lost))
            speeds.append(speed)
            accelerations.append(accel)
            stops.append(stop)
        raw[f] = per_frame
    def scale(values):
        nonzero = [v for v in values if v > 1e-9]
        return statistics.median(nonzero) if nonzero else 1.0
    scales = [scale(speeds), scale(accelerations), scale(stops)]
    result = {}
    for f in frames:
        # Cap extreme values so one signal cannot overwhelm every other one.
        components = [max(min(row[i] / scales[i], 10.0) for row in raw[f]) for i in range(3)]
        lost = sum(row[3] for row in raw[f]) / len(keys)
        result[f] = (*components, lost, sum(components) + 2.0 * lost)
    return result


def select(frames, scores, excluded, budget, min_gap, method):
    candidates = [f for f in frames if f not in excluded]
    if budget < 1 or min_gap < 1:
        raise ValueError('budget and min-gap must be positive')
    if method == 'motion':
        order = sorted(candidates, key=lambda f: (-scores[f][-1], f))
    else:
        # Match targets spaced across the clip; tie-break toward earlier frames.
        order = []
        remaining = set(candidates)
        for i in range(1, budget + 1):
            target = i * (frames[-1] + 1) / (budget + 1)
            if remaining:
                f = min(remaining, key=lambda n: (abs(n - target), n))
                order.append(f)
                remaining.remove(f)
        order.extend(sorted(remaining))
    chosen = []
    for frame in order:
        if all(abs(frame - other) >= min_gap for other in chosen):
            chosen.append(frame)
            if len(chosen) == budget:
                return chosen
    raise ValueError(f'Could select only {len(chosen)} of {budget} frames; reduce budget or min-gap, or adjust exclusions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', required=True, type=Path, help='frozen tracks CSV')
    parser.add_argument('--out', required=True, type=Path, help='ranked frame list CSV')
    parser.add_argument('--budget', required=True, type=int, help='number of correction frames, excluding initial query frames')
    parser.add_argument('--method', choices=('motion', 'interval'), default='motion')
    parser.add_argument('--exclude-csv', action='append', default=[], type=Path, help='CSV with frame column; repeat for query and held-out frames')
    parser.add_argument('--min-gap', type=int, default=1, help='minimum frame distance between selections')
    args = parser.parse_args()
    data, frames, keys = read_tracks(args.tracks)
    excluded = excluded_frames(args.exclude_csv)
    scores = motion_scores(data, frames, keys)
    selected = select(frames, scores, excluded, args.budget, args.min_gap, args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'frame', 'method', 'speed_score', 'acceleration_score', 'stop_score', 'lost_visibility_fraction', 'difficulty_score'])
        for rank, frame in enumerate(selected, 1):
            writer.writerow([rank, frame, args.method, *scores[frame]])
    print(f'Selected {len(selected)} frames in {args.out}. Label these frames manually; scores are not ground truth.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'Error: {exc}')
