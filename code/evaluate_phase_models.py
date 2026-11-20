
#!/usr/bin/env python3
"""
Compare three phase-prediction methods without ground-truth labels.

Inputs:
    - Rule-based prediction CSV
    - Pose-only Random Forest prediction CSV
    - Temporal Random Forest prediction CSV

Each CSV must contain:
    frame,phase

Allowed phase values:
    approach, mount, thrust, dismount, unknown

Outputs:
    predictions_aligned.csv
    model_summary.csv
    pairwise_agreement.csv
    transition_counts.csv

This script does NOT train models and does NOT calculate accuracy,
precision, recall, or F1 because no ground-truth labels are supplied.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = ["approach", "mount", "thrust", "dismount"]
UNKNOWN = "unknown"

MODEL_COLUMNS = {
    "rule": "rule_based",
    "pose": "pose_only_rf",
    "temporal": "temporal_rf",
}


def read_predictions(path):
    """Read and validate one model's frame-by-frame predictions."""
    df = pd.read_csv(path)

    if not {"frame", "phase"}.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns named 'frame' and 'phase'."
        )

    df = df[["frame", "phase"]].copy()

    df["frame"] = pd.to_numeric(df["frame"], errors="raise")

    if (
        df["frame"].isna().any()
        or (df["frame"] < 0).any()
        or (df["frame"] % 1 != 0).any()
    ):
        raise ValueError(f"{path}: frame values must be nonnegative integers.")

    df["frame"] = df["frame"].astype(int)
    df["phase"] = df["phase"].astype("string").str.strip().str.lower()

    if df["phase"].isna().any():
        raise ValueError(f"{path}: phase values cannot be blank.")

    valid_values = set(PHASES + [UNKNOWN])
    invalid = sorted(set(df["phase"]) - valid_values)

    if invalid:
        raise ValueError(
            f"{path}: invalid phase values {invalid}. "
            f"Allowed values: {sorted(valid_values)}"
        )

    if df["frame"].duplicated().any():
        duplicates = df.loc[
            df["frame"].duplicated(keep=False), "frame"
        ].tolist()
        raise ValueError(f"{path}: duplicate frame numbers: {duplicates[:10]}")

    return df.sort_values("frame").reset_index(drop=True)


def load_and_align(args):
    """Load all three predictions and align them by frame number."""
    paths = {
        "rule": args.rule,
        "pose": args.pose,
        "temporal": args.temporal,
    }

    aligned = None

    for model, path in paths.items():
        df = read_predictions(path)
        column = MODEL_COLUMNS[model]

        df = df.rename(columns={"phase": column})

        if aligned is None:
            aligned = df
        else:
            aligned = aligned.merge(
                df,
                on="frame",
                how="outer",
                validate="one_to_one",
            )

    aligned = aligned.sort_values("frame").reset_index(drop=True)

    # Missing frames in a model's file count as unknown predictions.
    for column in MODEL_COLUMNS.values():
        aligned[column] = aligned[column].fillna(UNKNOWN)

    return aligned


def calculate_model_summary(aligned, fps):
    """Summarize each model's prediction distribution and stability."""
    rows = []

    # Estimate the typical spacing between sampled frames.
    frame_gaps = np.diff(aligned["frame"].to_numpy())
    frame_gaps = frame_gaps[frame_gaps > 0]

    if len(frame_gaps) == 0:
        typical_gap = 1.0
    else:
        typical_gap = float(np.median(frame_gaps))

    for column in MODEL_COLUMNS.values():
        predictions = aligned[column].tolist()

        known = [p for p in predictions if p in PHASES]
        unknown_count = predictions.count(UNKNOWN)
        known_count = len(known)
        total_count = len(predictions)

        transitions = 0
        runs = 0
        run_lengths = []

        current_phase = None
        current_length = 0

        # Unknown predictions break a phase run.
        for prediction in predictions:
            if prediction == UNKNOWN:
                if current_phase is not None:
                    runs += 1
                    run_lengths.append(current_length)
                current_phase = None
                current_length = 0
                continue

            if prediction == current_phase:
                current_length += 1
            else:
                if current_phase is not None:
                    transitions += 1
                    runs += 1
                    run_lengths.append(current_length)

                current_phase = prediction
                current_length = 1

        if current_phase is not None:
            runs += 1
            run_lengths.append(current_length)

        phase_counts = {
            phase: predictions.count(phase)
            for phase in PHASES
        }

        row = {
            "model": column,
            "total_frames": total_count,
            "known_predictions": known_count,
            "unknown_predictions": unknown_count,
            "prediction_coverage_pct": (
                100 * known_count / total_count if total_count else 0
            ),
            "phase_transitions": transitions,
            "transitions_per_100_known_predictions": (
                100 * transitions / known_count if known_count else 0
            ),
            "phase_runs": runs,
            "average_run_length_samples": (
                float(np.mean(run_lengths)) if run_lengths else 0
            ),
            "median_run_length_samples": (
                float(np.median(run_lengths)) if run_lengths else 0
            ),
            "estimated_median_run_duration_seconds": (
                float(np.median(run_lengths) * typical_gap / fps)
                if run_lengths else 0
            ),
        }

        for phase in PHASES:
            row[f"{phase}_count"] = phase_counts[phase]
            row[f"{phase}_percentage"] = (
                100 * phase_counts[phase] / known_count
                if known_count else 0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_pairwise_agreement(aligned):
    """Compare models on frames where both models made known predictions."""
    rows = []

    for model_a, model_b in combinations(MODEL_COLUMNS.values(), 2):
        both_known = (
            aligned[model_a].isin(PHASES)
            & aligned[model_b].isin(PHASES)
        )

        compared = aligned.loc[both_known]

        if len(compared):
            matches = (
                compared[model_a] == compared[model_b]
            ).sum()
            agreement = 100 * matches / len(compared)
        else:
            matches = 0
            agreement = np.nan

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "frames_compared": len(compared),
            "matching_frames": int(matches),
            "agreement_percentage": agreement,
        })

    return pd.DataFrame(rows)


def calculate_transition_counts(aligned):
    """Count each model's observed phase-to-phase transitions."""
    rows = []

    for column in MODEL_COLUMNS.values():
        previous = None

        for prediction in aligned[column]:
            if prediction == UNKNOWN:
                previous = None
                continue

            if previous is not None and prediction != previous:
                rows.append({
                    "model": column,
                    "from_phase": previous,
                    "to_phase": prediction,
                    "count": 1,
                })

            previous = prediction

    if not rows:
        return pd.DataFrame(
            columns=["model", "from_phase", "to_phase", "count"]
        )

    return (
        pd.DataFrame(rows)
        .groupby(
            ["model", "from_phase", "to_phase"],
            as_index=False
        )["count"]
        .sum()
        .sort_values(["model", "from_phase", "to_phase"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare three phase-prediction outputs without labels."
    )

    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frames per second. Used for estimated durations.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")

    for path in [args.rule, args.pose, args.temporal]:
        if not path.is_file():
            parser.error(f"Prediction file not found: {path}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    aligned = load_and_align(args)

    summary = calculate_model_summary(aligned, args.fps)
    agreement = calculate_pairwise_agreement(aligned)
    transitions = calculate_transition_counts(aligned)

    aligned.to_csv(
        args.outdir / "predictions_aligned.csv", index=False
    )
    summary.to_csv(
        args.outdir / "model_summary.csv", index=False
    )
    agreement.to_csv(
        args.outdir / "pairwise_agreement.csv", index=False
    )
    transitions.to_csv(
        args.outdir / "transition_counts.csv", index=False
    )

    print("\nMODEL SUMMARY")
    print(summary.to_string(index=False))

    print("\nPAIRWISE AGREEMENT")
    print(agreement.to_string(index=False))

    print("\nTRANSITION COUNTS")
    if transitions.empty:
        print("No phase transitions were detected.")
    else:
        print(transitions.to_string(index=False))

    print(f"\nResults saved to: {args.outdir.resolve()}")
    print(
        "\nReminder: agreement and stability are not measures of "
        "accuracy. No ground-truth labels were used."
    )


if __name__ == "__main__":
    main()
