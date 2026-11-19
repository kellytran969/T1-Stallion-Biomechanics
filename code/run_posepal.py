#!/usr/bin/env python3
"""Run frozen CoTracker3 or PosePAL-style feature optimization on one clip.

Input annotations: frame,keypoint,x,y,visible (original video pixel coordinates).
Use only frame 0 / correction annotations here, never blind scoring annotations.
Output tracks: frame,keypoint,x,y,visible, as used by overlay_demo.py.

Requires the PosePAL co-tracker fork and scaled_offline.pth; see --help.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--method', choices=('frozen', 'tto'), required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cotracker-dir', type=Path, default=ROOT / 'external/PosePAL/dependencies/cotracker3')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--iters', type=int, default=1000, help='TTO optimizer steps (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='initial TTO learning rate')
    parser.add_argument('--final-lr', type=float, default=0.00001, help='final TTO learning rate')
    parser.add_argument('--feature-chunk', type=int, default=50, help='frames per feature extraction batch')
    return parser.parse_args()


def load_annotations(path, nframes, width, height):
    """Check the team's CSV; retain first appearance and visible corrections."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'frame', 'keypoint', 'x', 'y', 'visible'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Annotations must have columns: frame,keypoint,x,y,visible')
        points, seen = {}, set()
        for line, row in enumerate(reader, 2):
            try:
                f = int(row['frame'])
                key = row['keypoint'].strip()
                visible = int(row['visible'])
                if not key or visible not in (0, 1) or not 0 <= f < nframes:
                    raise ValueError('invalid keypoint, visibility, or frame number')
                if (f, key) in seen:
                    raise ValueError('duplicate frame/keypoint')
                seen.add((f, key))
                points.setdefault(key, [])
                if visible:
                    x, y = float(row['x']), float(row['y'])
                    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                        raise ValueError('visible coordinates must be finite and within the frame')
                    points[key].append((f, x, y))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Invalid annotation at CSV line {line}: {exc}') from exc
    if not points:
        raise ValueError('Annotations file contains no points')
    empty = [key for key, values in points.items() if not values]
    if empty:
        raise ValueError('No visible annotation for keypoint(s): ' + ', '.join(empty))
    for values in points.values():
        values.sort()
    return points


def read_video(path, cv2):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'No readable frames in video: {path}')
    return frames


def feature_pyramid(model, video, chunk, torch, F):
    # Mirrors PosePAL main.py extract_features. Input is RGB in [0, 255].
    b, t, c, h, w = video.shape
    with torch.no_grad():
        batches = []
        for start in range(0, t, chunk):
            part = video[:, start:start + chunk]
            features = model.fnet(part.reshape(-1, c, h, w))
            batches.append(features.reshape(b, part.shape[1], *features.shape[1:]))
        features = torch.cat(batches, dim=1)
        features = F.normalize(features, p=2, dim=2, eps=1e-6)
        pyramid = [features]
        for _ in range(model.corr_levels - 1):
            prev = pyramid[-1]
            pooled = F.avg_pool2d(prev.reshape(b * t, *prev.shape[2:]), 2, stride=2)
            pyramid.append(pooled.reshape(b, t, *pooled.shape[1:]))
    return pyramid


def main():
    args = arguments()
    if args.iters < 1 or args.lr <= 0 or args.final_lr <= 0 or args.feature_chunk < 1:
        raise ValueError('iters, lr, final-lr, and feature-chunk must be positive')
    if not args.cotracker_dir.is_dir():
        raise FileNotFoundError(f'CoTracker directory not found: {args.cotracker_dir}. Clone PosePAL recursively or use --cotracker-dir.')
    checkpoint = args.checkpoint or args.cotracker_dir / 'checkpoints/scaled_offline.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Download scaled_offline.pth or use --checkpoint.')
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError('Install PyTorch, numpy and opencv-python in your environment.') from exc
    sys.path.insert(0, str(args.cotracker_dir.resolve()))
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    from cotracker.models.core.cotracker.losses import sequence_loss

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available')
    print(f'Device: {device}; method: {args.method}', flush=True)
    raw = read_video(args.video, cv2)
    height, width = raw[0].shape[:2]
    if width < 2 or height < 2 or any(frame.shape != raw[0].shape for frame in raw):
        raise ValueError('Video frames must all have the same dimensions, at least 2 x 2')
    labels = load_annotations(args.annotations, len(raw), width, height)
    ids = list(labels)
    print(f'Loaded {len(raw)} frames and {len(ids)} keypoints', flush=True)
    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60,
                                  model_resolution=(384, 512), linear_layer_for_vis_conf=True)
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('model', state))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mh, mw = model.model_resolution
    # Resize on CPU before transfer to reduce GPU memory required by the clip.
    resized = np.stack([cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_LINEAR) for frame in raw])
    del raw
    video = torch.from_numpy(resized).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    sx, sy = (mw - 1) / (width - 1), (mh - 1) / (height - 1)
    query_values = [[labels[key][0][0], labels[key][0][1] * sx, labels[key][0][2] * sy] for key in ids]
    queries = torch.tensor([query_values], dtype=torch.float32, device=device)
    start_frames = [int(q[0]) for q in query_values]
    if any(start_frames):
        print('Some points start after frame 0; earlier frames will be marked invisible.', flush=True)

    if args.method == 'frozen':
        with torch.no_grad():
            tracks, visibility, _, _ = model(video, queries, iters=6)
    else:
        correction_count = sum(len(v) - 1 for v in labels.values())
        if not correction_count:
            raise ValueError('TTO needs visible correction annotations in addition to each first query.')
        print('Extracting video features; this may require substantial memory.', flush=True)
        pyramid = feature_pyramid(model, video, args.feature_chunk, torch, F)
        # PosePAL initializes each point's embedding from the mean support
        # features sampled at all its visible manual annotations.
        with torch.no_grad():
            per_point = []
            for key in ids:
                q = torch.tensor([[f, x * sx, y * sy] for f, x, y in labels[key]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
                _, support = model.get_track_feat(pyramid[0], q[:, :, 0].long(),
                                                  q[:, :, 1:] / model.stride,
                                                  support_radius=model.corr_radius)
                per_point.append(support.mean(dim=-2))
            features = torch.cat(per_point, dim=0).permute(1, 0, 2)[None, None, None].detach()
        original = features.clone()
        features.requires_grad_(True)
        truth = torch.zeros((1, len(resized), len(ids), 2), device=device)
        valid = torch.zeros((1, len(resized), len(ids)), device=device)
        for idx, key in enumerate(ids):
            for f, x, y in labels[key]:
                truth[0, f, idx] = torch.tensor((x * sx, y * sy), device=device)
                valid[0, f, idx] = 1
        optimizer = torch.optim.Adam([features], lr=args.lr)
        for step in range(args.iters):
            # Linear schedule, with the requested ending rate at the last step.
            frac = step / max(args.iters - 1, 1)
            optimizer.param_groups[0]['lr'] = args.lr + frac * (args.final_lr - args.lr)
            optimizer.zero_grad(set_to_none=True)
            _, _, _, train = model(video, queries, iters=4, is_train=True,
                                   fmaps_pyramid=pyramid,
                                   track_feat_support_pyramid=features.repeat(model.corr_levels, 1, 1, 1, 1, 1))
            tracking_loss = sequence_loss(train[0], [truth], [valid], gamma=0.8, add_huber_loss=True)
            regularizer = (features - original).abs().mean()
            loss = tracking_loss + 0.01 * regularizer
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite TTO loss at step {step + 1}')
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.iters:
                print(f'Step {step+1}/{args.iters}: loss={loss.item():.5f}, tracking={tracking_loss.item():.5f}, regularizer={regularizer.item():.5f}', flush=True)
        with torch.no_grad():
            tracks, visibility, _, _ = model(video, queries, iters=4, fmaps_pyramid=pyramid,
                track_feat_support_pyramid=features.repeat(model.corr_levels, 1, 1, 1, 1, 1))

    tracks = tracks[0].detach().cpu().numpy()
    visible = (visibility[0] > 0.9).detach().cpu().numpy()
    tracks[..., 0] /= sx
    tracks[..., 1] /= sy
    # Ground-truth query coordinates are known; apply the same query correction
    # used by CoTrackerPredictor for both methods.
    for i, key in enumerate(ids):
        f, x, y = labels[key][0]
        tracks[f, i] = (x, y)
        visible[f, i] = True
        visible[:f, i] = False
    if not np.isfinite(tracks).all():
        raise RuntimeError('Tracker produced non-finite coordinates; no CSV was written')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'keypoint', 'x', 'y', 'visible'])
        for f in range(len(tracks)):
            for i, key in enumerate(ids):
                writer.writerow([f, key, float(tracks[f, i, 0]), float(tracks[f, i, 1]), int(visible[f, i])])
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
