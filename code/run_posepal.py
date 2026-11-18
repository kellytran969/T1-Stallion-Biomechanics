import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import imageio.v3 as iio

# Add external/PosePAL/co-tracker to Python path
COTRACKER_DIR = Path(__file__).resolve().parents[1] / "external" / "PosePAL" / "co-tracker"
sys.path.insert(0, str(COTRACKER_DIR))

from cotracker.predictor import CoTrackerPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Run PosePAL baseline or TTO adaptation")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--method", type=str, choices=["frozen", "tto"], required=True)
    parser.add_argument("--iters", type=int, default=50, help="Number of TTO iterations")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for TTO")
    parser.add_argument("--out", type=str, required=True, help="Output path for tracks CSV")
    return parser.parse_args()

def load_video_tensor(video_path, device):
    frames = iio.imread(video_path, plugin="FFMPEG")  # [T, H, W, 3]
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # [1, T, 3, H, W]
    return video

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading video from {args.video}...")
    video = load_video_tensor(args.video, device)

    ckpt_path = Path(__file__).resolve().parents[1] / "external" / "PosePAL" / "checkpoints" / "scaled_offline.pth"
    print(f"Loading CoTracker3 offline model from {ckpt_path}...")
    predictor = CoTrackerPredictor(checkpoint=str(ckpt_path), offline=True, window_len=60).to(device)
    model = predictor.model
    model.eval()

    # Load sparse annotations
    df = pd.read_csv(args.annotations)
    frames = sorted(df['frame'].unique())
    landmark_ids = sorted(df['landmark_id'].unique())
    num_points = len(landmark_ids)

    # Build initial queries from frame 0: shape [1, N, 3] -> (t, x, y)
    frame_0_df = df[df['frame'] == 0]
    queries = torch.zeros((1, num_points, 3), device=device)
    for idx, l_id in enumerate(landmark_ids):
        row = frame_0_df[frame_0_df['landmark_id'] == l_id]
        if not row.empty:
            queries[0, idx, 0] = 0.0
            queries[0, idx, 1] = float(row['x'].values[0])
            queries[0, idx, 2] = float(row['y'].values[0])

    if args.method == "frozen":
        print("Running frozen CoTracker3 baseline...")
        with torch.no_grad():
            pred_tracks, pred_visibility = predictor(video, queries=queries)

    elif args.method == "tto":
        print(f"Running PosePAL TTO adaptation (iters={args.iters}, lr={args.lr})...")
        # Freeze all model weights except query point embeddings (qp_emb)
        for name, param in model.named_parameters():
            param.requires_grad = ("qp_emb" in name) or ("query" in name)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            # Fallback if model uses buffer/feature-level adaptation
            print("Warning: No explicit qp_emb parameter found; optimizing input query offsets/features.")
            query_delta = torch.zeros_like(queries, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([query_delta], lr=args.lr)
        else:
            query_delta = None
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        # Build sparse ground truth tensor for supervision
        gt_frames = torch.tensor(frames, dtype=torch.long, device=device)
        gt_coords = torch.zeros((len(frames), num_points, 2), device=device)
        gt_valid = torch.zeros((len(frames), num_points), device=device)

        for f_idx, f in enumerate(frames):
            f_data = df[df['frame'] == f]
            for idx, l_id in enumerate(landmark_ids):
                row = f_data[f_data['landmark_id'] == l_id]
                if not row.empty and row['visible'].values[0] == 1:
                    gt_coords[f_idx, idx, 0] = float(row['x'].values[0])
                    gt_coords[f_idx, idx, 1] = float(row['y'].values[0])
                    gt_valid[f_idx, idx] = 1.0

        for step in range(args.iters):
            optimizer.zero_grad()
            active_queries = queries + query_delta if query_delta is not None else queries
            pred_tracks, pred_visibility = predictor(video, queries=active_queries)
            
            # Extract predictions at the annotated correction frames: [len(frames), N, 2]
            pred_at_gt = pred_tracks[0, gt_frames]
            loss = (F.smooth_l1_loss(pred_at_gt, gt_coords, reduction='none').sum(dim=-1) * gt_valid).sum() / (gt_valid.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step [{step+1}/{args.iters}] - Sparse L1 Loss: {loss.item():.4f}")

        with torch.no_grad():
            active_queries = queries + query_delta if query_delta is not None else queries
            pred_tracks, pred_visibility = predictor(video, queries=active_queries)

    # Save tracks to CSV
    print(f"Saving output tracks to {args.out}...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tracks_np = pred_tracks[0].detach().cpu().numpy()
    vis_np = pred_visibility[0].detach().cpu().numpy()

    out_rows = []
    for t in range(tracks_np.shape[0]):
        for idx, l_id in enumerate(landmark_ids):
            out_rows.append({
                "frame": t,
                "landmark_id": l_id,
                "x": float(tracks_np[t, idx, 0]),
                "y": float(tracks_np[t, idx, 1]),
                "visible": int(vis_np[t, idx, 0] > 0.5)
            })

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print("Done!")

if __name__ == "__main__":
    main()