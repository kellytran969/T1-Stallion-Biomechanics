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
Training can use any nonempty subset of the four phases. A model can predict
only phases present in training; its validation score averages over the union
of phases present in train or validation, using zero for undefined F1 scores.
Warnings and the JSON report identify absent phases and validation support.
Prediction writes frame,phase for overlay_demo.py; unknown means no endpoints
were available in that frame, and is not a fifth trained class.

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
    trained_classes = [name for name in CLASSES if name in set(ytr)]
    validation_classes = [name for name in CLASSES if name in set(yval)]
    missing_classes = [name for name in CLASSES if name not in trained_classes]
    missing_val = [name for name in trained_classes if name not in validation_classes]
    unseen_val = [name for name in validation_classes if name not in trained_classes]
    score_classes = [name for name in CLASSES if name in set(ytr) | set(yval)]
    if missing_classes:
        warnings.warn(f'Training lacks {missing_classes}; the model cannot predict those phases.')
    if len(trained_classes) == 1:
        warnings.warn('Training has only one phase; predictions are constant when endpoint features are available.')
    if unseen_val:
        warnings.warn(f'Validation includes {unseen_val}, which the model cannot predict because training lacks them.')
    if missing_val:
        warnings.warn(f'Validation lacks {missing_val}; their F1 is counted as zero and their performance is not established.')
    grid = {'n_estimators': [100, 200], 'max_depth': [None, 12],
            'min_samples_leaf': [1, 5], 'max_features': ['sqrt', 1.0]}
    best, rows = None, []
    for params in ParameterGrid(grid):
        pipeline = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
                             ('forest', RandomForestClassifier(**params, class_weight='balanced', random_state=seed, n_jobs=1))])
        pipeline.fit(Xtr, ytr)
        pred = pipeline.predict(Xval)
        score = f1_score(yval, pred, labels=score_classes, average='macro', zero_division=0)
        rows.append({'validation_macro_f1_scored_classes': float(score), **params})
        if best is None or score > best[0]:
            best = (score, params, pipeline, pred)
    score, params, model, pred = best
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (variant + '.joblib')
    joblib.dump({'model': model, 'variant': variant, 'lag': lag,
                 'columns': list(Xtr.columns), 'classes': trained_classes}, model_path)
    results = {'variant': variant, 'lag_frames': lag, 'training_frames': len(ytr),
               'validation_frames': len(yval), 'training_classes': ytr.value_counts().to_dict(),
               'validation_classes': yval.value_counts().to_dict(),
               'trained_classes': trained_classes, 'untrained_classes': missing_classes,
               'validation_present_classes': validation_classes, 'validation_unseen_classes': unseen_val,
               'validation_missing_trained_classes': missing_val, 'macro_f1_classes': score_classes,
               'best_validation_macro_f1_scored_classes': float(score), 'best_params': params,
               'classification_report_validation': classification_report(yval, pred, labels=score_classes, output_dict=True, zero_division=0),
               'note': 'Macro F1 averages classes present in training or validation (undefined F1=0). Validation was used for tuning; it is not held-out test performance. Model fitted on training data only.',
               'grid_results': rows}
    (model_dir / (variant + '_validation.json')).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'{variant}: scored classes {score_classes}; best validation macro F1 = {score:.4f}; saved {model_path}')


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


def write_phase_rules(path: Path) -> None:
    config = {
        "example_only": False,
        "lag_frames": 1,
        "note": (
            "Endpoints-only baseline rules fitted on training data "
            "(other excluded; approach prioritized) and checked on "
            "validation data before held-out comparison."
        ),
        "rules": [
            {
                "phase": "approach",
                "all": [{"feature": "back_angle", "op": ">", "value": -12.0}],
            },
            {
                "phase": "mount",
                "all": [{"feature": "hind_position", "op": ">", "value": -0.45}],
            },
            {
                "phase": "dismount",
                "all": [
                    {"feature": "neck_angle", "op": "<", "value": -33.0},
                    {"feature": "hind_position", "op": ">", "value": -0.7},
                ],
            },
            {
                "phase": "thrust",
                "all": [{"feature": "back_angle", "op": "<=", "value": -12.0}],
            },
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Wrote endpoints-based phase rules to {path}")


def rule_predictions(df, config):
    if not isinstance(config, dict) or not isinstance(config.get('rules'), list) or not config['rules']:
        raise ValueError('Rule config needs a nonempty rules list')
    lag = config.get('lag_frames', 1)
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError('Rule lag_frames must be a positive integer')
    X = make_features(df, 'temporal', lag)
    # Make the possible inputs explicit; visibility indicators are not phase labels.
    allowed = set(FEATURES + [c + '_delta' for c in FEATURES])
    ops = {'<': lambda x, v: x < v, '<=': lambda x, v: x <= v,
           '>': lambda x, v: x > v, '>=': lambda x, v: x >= v}
    for rule in config['rules']:
        if not isinstance(rule, dict) or rule.get('phase') not in CLASSES:
            raise ValueError('Each rule needs a valid phase')
        conditions = rule.get('all')
        if not isinstance(conditions, list) or not conditions:
            raise ValueError('Each rule needs at least one condition in all')
        for cond in conditions:
            if not isinstance(cond, dict) or cond.get('feature') not in allowed or cond.get('op') not in ops:
                raise ValueError('Invalid rule feature or operator')
            value = cond.get('value')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Rule thresholds must be finite numbers')
    outputs = []
    for _, row in X.iterrows():
        chosen = 'unknown'
        for rule in config['rules']:
            states = []
            for cond in rule['all']:
                value = row[cond['feature']]
                states.append(None if pd.isna(value) else ops[cond['op']](value, cond['value']))
            if False in states:
                continue
            if None in states:
                # Do not use a lower-priority rule when a higher-priority rule
                # could have matched if its missing feature were observed.
                break
            chosen = rule['phase']
            break
        outputs.append(chosen)
    return outputs


def run_rules(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('example_only') is not False:
        raise ValueError('Rules are example-only or unreviewed. Review and edit the config, then explicitly set example_only to false.')
    df = read_endpoints(args.endpoints)
    phase = rule_predictions(df, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'frame': df['frame'], 'phase': phase}).to_csv(args.out, index=False)
    print(f'Wrote {args.out}; {phase.count("unknown")} frames unknown (missing data or no matching rule)')


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
    rt = sub.add_parser('rules-template', help='Create an illustrative JSON rules file (review before use)')
    rt.add_argument('--out', type=Path, required=True)
    rule = sub.add_parser('rules', help='Apply reviewed human-defined rules to one endpoint CSV')
    rule.add_argument('--endpoints', type=Path, required=True)
    rule.add_argument('--config', type=Path, required=True)
    rule.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'train':
        if args.lag < 1: raise ValueError('lag must be positive')
        manifest = read_manifest(args.manifest)
        variants = ['pose', 'temporal'] if args.variant == 'both' else [args.variant]
        for variant in variants:
            train_variant(manifest, args.manifest.resolve().parent, variant, args.lag, args.model_dir, args.seed)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'rules-template':
        write_phase_rules(args.out)
    else:
        run_rules(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
