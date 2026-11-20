#!/usr/bin/env python3
"""Train and run stallion phase random forests on endpoint CSVs.

Inputs: manifest CSV columns clip_id,group_id,split,endpoints,labels.
Splits: train or val. Groups (horse IDs if known; otherwise clip IDs) must
not overlap. Paths resolve relative to the manifest's parent directory.
Endpoint CSV: frame,back_angle,neck_angle,hind_position,chest_position, as
produced by data+endpoints/endpoints.py. Alternate overlay column names are
accepted. Labels CSV: frame,phase OR start_frame,end_frame,phase (inclusive),
with reviewed, zero-based frames and approach/mount/thrust/dismount labels.

Training tunes a pose-only and/or temporal random forest on validation clips.
It saves the best estimator for each requested variant, fitted on TRAIN ONLY.
No held-out test set is read. Validation F1 is model-selection evidence only.
Prediction writes frame,phase for overlay_demo.py; unknown means no endpoints
were available in that frame, and is not a fifth trained class.

Examples (run from repository root):
 python code/phase_prediction.py train --manifest phase_manifest.csv --model-dir results/phase_models --variant both
 python code/phase_prediction.py predict --model results/phase_models/temporal.joblib --endpoints results/clip-01_endpoints.csv --out results/clip-01_phases.csv

joblib model files may execute code when loaded: load only trusted models.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

CLASSES = ['approach', 'mount', 'thrust', 'dismount']
FEATURES = ['back_angle', 'neck_angle', 'hind_position', 'chest_position']
ALIASES = {'theta_back_deg': 'back_angle', 'theta_neck_deg': 'neck_angle',
           'hind_norm': 'hind_position', 'chest_norm': 'chest_position'}


def read_endpoints(path):
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in ALIASES.items() if v not in df.columns})
    missing = {'frame', *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing endpoint columns: {sorted(missing)}')
    df = df[['frame', *FEATURES]].copy()
    df['frame'] = pd.to_numeric(df['frame'], errors='raise')
    if df['frame'].isna().any() or (df['frame'] < 0).any() or (df['frame'] % 1 != 0).any():
        raise ValueError(f'{path}: frames must be nonnegative integers')
    df['frame'] = df['frame'].astype(int)
    if df['frame'].duplicated().any():
        raise ValueError(f'{path}: duplicate frame numbers')
    df = df.sort_values('frame').reset_index(drop=True)
    if df.empty:
        raise ValueError(f'{path}: no frames')
    for name in FEATURES:
        # Invalid nonblank values must not silently become missing.
        df[name] = pd.to_numeric(df[name], errors='raise').replace([np.inf, -np.inf], np.nan)
    return df


def make_features(df, variant, lag):
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c + '_missing'] = X[c].isna().astype(float)
    if variant == 'temporal':
        # Only compare exactly lag frames apart; never bridge dropped frames.
        previous = df.set_index('frame')[FEATURES].reindex(df['frame'].to_numpy() - lag)
        previous.index = df.index
        for c in FEATURES:
            delta = df[c] - previous[c]
            if c in ('back_angle', 'neck_angle'):
                delta = (delta + 180) % 360 - 180
            X[c + '_delta'] = delta
            X[c + '_delta_missing'] = delta.isna().astype(float)
    return X


def read_labels(path, frames):
    labels = pd.read_csv(path)
    if 'phase' not in labels:
        raise ValueError(f'{path}: expected phase column')
    labels['phase'] = labels['phase'].astype('string').str.strip().str.lower()
    if labels['phase'].isna().any() or not labels['phase'].isin(CLASSES).all():
        raise ValueError(f'{path}: labels must be one of {CLASSES}')
    allowed = set(frames)
    output = {}
    def integer(value):
        v = float(value)
        if not np.isfinite(v) or v < 0 or not v.is_integer():
            raise ValueError(f'{path}: frame numbers must be nonnegative integers')
        return int(v)
    if 'frame' in labels:
        for row in labels.itertuples(index=False):
            f = integer(getattr(row, 'frame'))
            if f in output:
                raise ValueError(f'{path}: duplicate frame {f}')
            if f not in allowed:
                raise ValueError(f'{path}: labeled frame {f} not in endpoints')
            output[f] = row.phase
    elif {'start_frame', 'end_frame'} <= set(labels.columns):
        for row in labels.itertuples(index=False):
            start, end = integer(row.start_frame), integer(row.end_frame)
            if end < start:
                raise ValueError(f'{path}: interval ends before it starts')
            # Every frame in a reviewed interval must exist in endpoints.
            for f in range(start, end + 1):
                if f not in allowed:
                    raise ValueError(f'{path}: labeled frame {f} not in endpoints')
                if f in output:
                    raise ValueError(f'{path}: overlapping intervals at frame {f}')
                output[f] = row.phase
    else:
        raise ValueError(f'{path}: use frame,phase or start_frame,end_frame,phase')
    return pd.Series(df_map(frames, output), dtype='object')


def df_map(frames, mapping):
    return [mapping.get(int(f)) for f in frames]


def read_manifest(path):
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'clip_id', 'group_id', 'split', 'endpoints', 'labels'}
    if not required <= set(m.columns):
        raise ValueError(f'{path}: required columns: {sorted(required)}')
    for col in required:
        m[col] = m[col].str.strip()
        if (m[col] == '').any():
            raise ValueError(f'{path}: empty {col}')
    m['split'] = m['split'].str.lower()
    if not set(m['split']) <= {'train', 'val'}:
        raise ValueError('Manifest split must be train or val; keep test clips out of training')
    if m['clip_id'].duplicated().any():
        raise ValueError('Manifest clip_id must be unique')
    if set(m.loc[m.split == 'train', 'group_id']) & set(m.loc[m.split == 'val', 'group_id']):
        raise ValueError('Group leakage: a horse or clip appears in both train and val')
    if set(m['split']) != {'train', 'val'}:
        raise ValueError('Manifest needs at least one train and one val clip')
    return m


def load_split(manifest, base, variant, lag):
    result = {}
    seen_paths = set()
    for split in ['train', 'val']:
        xs, ys = [], []
        for row in manifest[manifest['split'] == split].itertuples(index=False):
            ep = (base / row.endpoints).resolve()
            lp = (base / row.labels).resolve()
            if ep in seen_paths:
                raise ValueError(f'Endpoint file reused in manifest: {ep}')
            seen_paths.add(ep)
            df = read_endpoints(ep)
            y = read_labels(lp, df.frame)
            X = make_features(df, variant, lag)
            usable = y.notna() & df[FEATURES].notna().any(axis=1)
            dropped = int((y.notna() & ~usable).sum())
            print(f'{row.clip_id} ({split}): {int(usable.sum())} labeled usable frames; {dropped} labeled frames have no endpoint data')
            if not usable.any():
                raise ValueError(f'{row.clip_id}: no labeled frames with at least one endpoint')
            xs.append(X.loc[usable]); ys.append(y.loc[usable])
        result[split] = (pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True))
    return result


def train_variant(manifest, base, variant, lag, model_dir, seed):
    splits = load_split(manifest, base, variant, lag)
    Xtr, ytr = splits['train']; Xval, yval = splits['val']
    missing_classes = sorted(set(CLASSES) - set(ytr))
    if missing_classes:
        raise ValueError(f'Training set is missing phase labels: {missing_classes}. Collect reviewed training labels for all four phases.')
    missing_val = sorted(set(CLASSES) - set(yval))
    if missing_val:
        warnings.warn(f'Validation set has no examples of {missing_val}; macro F1 counts missing classes as zero and cannot establish their performance.')
    grid = {'n_estimators': [100, 200], 'max_depth': [None, 12],
            'min_samples_leaf': [1, 5], 'max_features': ['sqrt', 1.0]}
    best, rows = None, []
    for params in ParameterGrid(grid):
        pipeline = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
                             ('forest', RandomForestClassifier(**params, class_weight='balanced', random_state=seed, n_jobs=1))])
        pipeline.fit(Xtr, ytr)
        pred = pipeline.predict(Xval)
        score = f1_score(yval, pred, labels=CLASSES, average='macro', zero_division=0)
        rows.append({'validation_macro_f1_all_four': float(score), **params})
        if best is None or score > best[0]:
            best = (score, params, pipeline, pred)
    score, params, model, pred = best
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (variant + '.joblib')
    joblib.dump({'model': model, 'variant': variant, 'lag': lag,
                 'columns': list(Xtr.columns), 'classes': CLASSES}, model_path)
    results = {'variant': variant, 'lag_frames': lag, 'training_frames': len(ytr),
               'validation_frames': len(yval), 'training_classes': ytr.value_counts().to_dict(),
               'validation_classes': yval.value_counts().to_dict(),
               'best_validation_macro_f1_all_four': float(score), 'best_params': params,
               'classification_report_validation': classification_report(yval, pred, labels=CLASSES, output_dict=True, zero_division=0),
               'note': 'Validation was used for tuning. This is not held-out test performance. Model fitted on training data only.',
               'grid_results': rows}
    (model_dir / (variant + '_validation.json')).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'{variant}: best validation macro F1 (all four classes) = {score:.4f}; saved {model_path}')


def predict(args):
    bundle = joblib.load(args.model)
    df = read_endpoints(args.endpoints)
    X = make_features(df, bundle['variant'], bundle['lag'])
    if list(X.columns) != bundle['columns']:
        raise ValueError('Model feature columns do not match script features')
    valid = df[FEATURES].notna().any(axis=1)
    phase = np.full(len(df), 'unknown', dtype=object)
    if valid.any():
        phase[valid] = bundle['model'].predict(X.loc[valid])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'frame': df['frame'], 'phase': phase}).to_csv(args.out, index=False)
    print(f'Wrote {args.out}; {int((~valid).sum())} frames marked unknown because all endpoints were missing')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train', help='Tune models with separate train and validation groups')
    tr.add_argument('--manifest', type=Path, required=True)
    tr.add_argument('--model-dir', type=Path, required=True)
    tr.add_argument('--variant', choices=['pose', 'temporal', 'both'], default='both')
    tr.add_argument('--lag', type=int, default=1, help='Temporal difference lag in frames')
    tr.add_argument('--seed', type=int, default=42)
    pr = sub.add_parser('predict', help='Predict phases from one endpoint CSV')
    pr.add_argument('--model', type=Path, required=True)
    pr.add_argument('--endpoints', type=Path, required=True)
    pr.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'train':
        if args.lag < 1: raise ValueError('lag must be positive')
        manifest = read_manifest(args.manifest)
        variants = ['pose', 'temporal'] if args.variant == 'both' else [args.variant]
        for variant in variants:
            train_variant(manifest, args.manifest.resolve().parent, variant, args.lag, args.model_dir, args.seed)
    else:
        predict(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
