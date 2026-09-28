"""R1: declared subset retraining; official DT/Cat fit functions, saved predictions."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'official_code' / 'src'))
from classifiers import CLASSIFIER_HPS, _fit_catboost, _fit_decision_tree

DATASETS = ['ecoli', 'yeast_me2', 'abalone_19', 'mammography']
SEEDS = tuple(range(7))
MODELS = ('dt', 'cat')
METHODS = ('none', 'random', 'smote')
MODES = [('none', None, None)] + [('random', r, None) for r in [.2, .5]] + [('smote', r, k) for r in [.2, .5] for k in [3, 5]]
ENDPOINTS = ('auc', 'f1_fixed', 'f1_tuned')
TIE_RULE = 'validation endpoint descending, then hp_id, ratio, k descending; missing=-1; never test scores'


class ProvenanceError(RuntimeError):
    """A saved run cannot safely be resumed or included in an analysis."""


def refuse(message):
    raise ProvenanceError(f'{message}. Refusing reuse; choose a new output directory.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def package_versions():
    return {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'catboost', 'xgboost', 'lightgbm', 'scipy']}


def job_specs(dataset, model, seed):
    for method, ratio, k in MODES:
        for hp_id, hp in enumerate(CLASSIFIER_HPS[model]):
            yield dict(run_id=f'{dataset}__{model}__s{seed}__{method}__r{ratio}__k{k}__hp{hp_id}',
                       dataset=dataset, model=model, seed=seed, method=method,
                       ratio=ratio, k=k, hp_id=hp_id, hp=deepcopy(hp))


def all_specs():
    return {s['run_id']: s for d in DATASETS for seed in SEEDS for m in MODELS for s in job_specs(d, m, seed)}


def validate_manifest(out, *, training=False):
    path = out / 'manifest.json'
    if not path.exists():
        refuse('Missing training manifest')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        refuse(f'Unreadable manifest: {error}')
    if manifest['script_sha256'] != sha(__file__):
        snapshot = out / 'run_reproduction_training_snapshot.py'
        if training or not snapshot.exists() or sha(snapshot) != manifest['script_sha256']:
            refuse('Training code hash differs and no verified analysis-only snapshot applies')
    if manifest['protocol_sha256'] != sha(ROOT / 'reproduction_protocol.md'):
        refuse('Protocol hash differs')
    sources = {p.name: sha(p) for p in (ROOT / 'official_code' / 'src').glob('*.py')}
    if sources != manifest['official_sources']:
        refuse('Official source hashes differ')
    if training and manifest.get('packages') != package_versions():
        refuse('Training dependency versions differ')
    if {d['dataset'] for d in manifest['datasets']} != set(DATASETS):
        refuse('Dataset grid differs from the training manifest')
    if manifest.get('expected_fits') != len(all_specs()):
        refuse('Expected fit grid differs from the training manifest')
    return manifest


def load_verified_split(out, manifest, dataset, seed):
    entry = next(d for d in manifest['datasets'] if d['dataset'] == dataset)
    data_path = out / 'data' / f'{dataset}.npz'
    if not data_path.exists() or sha(data_path) != entry['sha256']:
        refuse(f'Dataset hash differs: {dataset}')
    with np.load(data_path) as data:
        x, y = data['x'], data['y']
    if list(x.shape) != list(entry['shape']) or int(y.sum()) != entry['minority']:
        refuse(f'Dataset metadata differs: {dataset}')
    split_path = out / 'splits' / f'{dataset}_{seed}.npz'
    if not split_path.exists():
        refuse(f'Missing split: {dataset}/{seed}')
    expected_hash = manifest.get('splits_sha256', {}).get(f'{dataset}_{seed}')
    if expected_hash is not None and sha(split_path) != expected_hash:
        refuse(f'Split hash differs: {dataset}/{seed}')
    with np.load(split_path) as saved:
        split = tuple(saved[key] for key in ['train', 'validation', 'test'])
    if not all(np.array_equal(a, b) for a, b in zip(split, split_indices(y, seed))):
        refuse(f'Saved split indices differ from the declared protocol: {dataset}/{seed}')
    return x, y, split


def record_provenance(out, manifest, dataset, seed):
    data_path = out / 'data' / f'{dataset}.npz'
    split_path = out / 'splits' / f'{dataset}_{seed}.npz'
    return dict(manifest_sha256=sha(out / 'manifest.json'), script_sha256=manifest['script_sha256'],
                protocol_sha256=manifest['protocol_sha256'],
                dataset_sha256=sha(data_path) if data_path.exists() else None,
                split_sha256=sha(split_path) if split_path.exists() else None)


def validate_record(record, out, manifest, spec, verified_data=None):
    if any(record.get(k) != v for k, v in spec.items()):
        refuse(f'Run identity/HP mismatch: {spec["run_id"]}')
    provenance = record_provenance(out, manifest, spec['dataset'], spec['seed'])
    keys = ['dataset_sha256', 'split_sha256']
    if manifest.get('schema_version', 1) >= 2:
        keys += ['manifest_sha256', 'script_sha256', 'protocol_sha256']
    if any(record.get(k) != provenance[k] for k in keys):
        refuse(f'Run provenance hash differs: {spec["run_id"]}')
    if record.get('status') not in ['success', 'failed']:
        refuse(f'Unknown run status: {spec["run_id"]}')
    if record['status'] == 'success':
        path = (out / record.get('predictions', '')).resolve()
        if not path.is_relative_to((out / 'predictions').resolve()) or not path.is_file() or sha(path) != record.get('prediction_sha256'):
            refuse(f'Prediction hash/path differs: {spec["run_id"]}')
        x, y, (_, va, te) = verified_data or load_verified_split(out, manifest, spec['dataset'], spec['seed'])
        with np.load(path) as predictions:
            for prefix, idx in [('validation', va), ('test', te)]:
                p = predictions[f'{prefix}_prob']
                if not np.array_equal(predictions[f'{prefix}_idx'], idx) or not np.array_equal(predictions[f'{prefix}_y'], y[idx]):
                    refuse(f'Prediction indices/labels differ: {spec["run_id"]}')
                if p.shape != (len(idx),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    refuse(f'Invalid saved probabilities: {spec["run_id"]}')
        if not all(np.isfinite(value) for block in ['validation', 'test'] for value in record[block].values()):
            refuse(f'Nonfinite saved metric: {spec["run_id"]}')
    return record


def split_indices(y, seed):
    trval, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    train, val = train_test_split(trval, test_size=.25, stratify=y[trval], random_state=seed)
    return train, val, test


def threshold_from_validation(y, p):
    ts = np.arange(0, 1, .01)
    pred = np.asarray(p)[:, None] > ts[None, :]
    tp = (pred & (np.asarray(y)[:, None] == 1)).sum(axis=0)
    den = pred.sum(axis=0) + np.sum(y)
    scores = np.divide(2 * tp, den, out=np.zeros(100), where=den > 0)
    return float(ts[np.argmax(scores)]) if scores.max() > 0 else float(ts[-1])


def f1(y, p, t):
    pred = p > t
    den = pred.sum() + y.sum()
    return float(2 * np.sum(pred & (y == 1)) / den) if den else 0.0


def metrics(y, p, threshold):
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    losses = -(y * np.log(pc) + (1-y) * np.log1p(-pc))
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                logloss=float(losses.mean()), loss_minority=float(losses[y == 1].mean()),
                loss_majority=float(losses[y == 0].mean()), brier=float(np.mean((p-y)**2)),
                f1_fixed=f1(y, p, .5), f1_tuned=f1(y, p, threshold))


def target_count(y, ratio):
    return int(np.round(np.sum(y == 0) * ratio / (1 - ratio)))


def record_task_failure(job, error, manifest=None):
    dataset, model_name, seed, out_str = job
    out = Path(out_str)
    manifest = manifest or validate_manifest(out, training=True)
    specs = list(job_specs(dataset, model_name, seed))
    saved_records = {}
    # Validate every reused record before writing any new failure placeholders.
    for spec in specs:
        path = out / 'runs' / f'{spec["run_id"]}.json'
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                saved_records[spec['run_id']] = validate_record(saved, out, manifest, spec)
            except Exception as integrity_error:
                refuse(f'Cannot reuse a record after task failure: {integrity_error}')
    records = []
    for spec in specs:
        path = out / 'runs' / f'{spec["run_id"]}.json'
        if spec['run_id'] in saved_records:
            records.append(saved_records[spec['run_id']])
            continue
        record = dict(**spec, **record_provenance(out, manifest, dataset, seed), status='failed',
                      failure_stage='task', error=repr(error), wall_seconds=0.0, cpu_seconds=0.0,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, record)
        records.append(record)
    return dict(dataset=dataset, model=model_name, seed=seed, count=len(records),
                success=sum(r['status'] == 'success' for r in records), task_error=repr(error))


def execute_job(job):
    manifest = validate_manifest(Path(job[3]), training=True)
    try:
        return _execute_job(job, manifest)
    except ProvenanceError:
        raise  # Never turn an integrity mismatch into reusable result files.
    except Exception as error:
        return record_task_failure(job, error, manifest)


def _execute_job(job, manifest):
    dataset, model_name, seed, out_str = job
    out = Path(out_str)
    verified = load_verified_split(out, manifest, dataset, seed)
    x, y, (tr, va, te) = verified
    provenance = record_provenance(out, manifest, dataset, seed)
    records = []
    job_started = time.perf_counter()
    for mode, ratio, k in MODES:
        xr, yr = x[tr], y[tr]
        sampling_error = None
        try:
            if mode != 'none':
                target = target_count(yr, ratio)
                sampler = RandomOverSampler(sampling_strategy={1: target}, random_state=seed) if mode == 'random' else SMOTE(sampling_strategy={1: target}, k_neighbors=k, random_state=seed)
                xr, yr = sampler.fit_resample(xr, yr)
        except Exception as error:
            sampling_error = (repr(error), traceback.format_exc())
        for hp_id, hps in enumerate(CLASSIFIER_HPS[model_name]):
            run_id = f'{dataset}__{model_name}__s{seed}__{mode}__r{ratio}__k{k}__hp{hp_id}'
            result_path = out / 'runs' / f'{run_id}.json'
            cfg = dict(run_id=run_id, dataset=dataset, model=model_name, seed=seed, method=mode, ratio=ratio, k=k, hp_id=hp_id, hp=deepcopy(hps))
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding='utf-8'))
                records.append(validate_record(saved, out, manifest, cfg, verified))
                continue
            record = dict(**cfg, **provenance, n_train=len(tr), n_resampled=None if sampling_error else len(yr), n_val=len(va), n_test=len(te))
            wall, cpu = time.perf_counter(), time.process_time()
            if sampling_error:
                record.update(status='failed', failure_stage='resampling', error=sampling_error[0], traceback=sampling_error[1],
                              wall_seconds=0.0, cpu_seconds=0.0, finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(result_path, record)
                records.append(record)
                continue
            try:
                params = deepcopy(hps)
                if model_name == 'cat':
                    params['allow_writing_files'] = False
                with threadpool_limits(limits=1):
                    model = {'cat': _fit_catboost, 'dt': _fit_decision_tree}[model_name](xr, yr, x[va], y[va], seed, params)
                    prediction_kwargs = {'thread_count': 1} if model_name == 'cat' else {}
                    pv = model.predict_proba(x[va], **prediction_kwargs)[:, 1]
                    pt = model.predict_proba(x[te], **prediction_kwargs)[:, 1]
                assert np.isfinite(pv).all() and np.isfinite(pt).all()
                threshold = threshold_from_validation(y[va], pv)
                record.update(status='success', threshold=threshold, validation=metrics(y[va], pv, threshold), test=metrics(y[te], pt, threshold))
                pred_path = out / 'predictions' / f'{run_id}.npz'
                np.savez_compressed(pred_path, validation_prob=pv, test_prob=pt, validation_y=y[va], test_y=y[te], validation_idx=va, test_idx=te)
                record.update(predictions=str(pred_path.relative_to(out)), prediction_sha256=sha(pred_path), effective_params=model.get_all_params() if model_name == 'cat' else model.get_params(), fitted_trees=int(model.tree_count_) if model_name == 'cat' else None)
            except Exception as e:
                record.update(status='failed', failure_stage='fit_or_evaluation', error=repr(e), traceback=traceback.format_exc())
            record.update(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(result_path, record)
            records.append(record)
    return dict(dataset=dataset, model=model_name, seed=seed, count=len(records), success=sum(r['status']=='success' for r in records), wall_seconds=time.perf_counter()-job_started)


def prepare(out):
    if (out / 'manifest.json').exists():
        metadata = validate_manifest(out, training=True)
        data = {(d, seed): load_verified_split(out, metadata, d, seed) for d in DATASETS for seed in SEEDS}
        specs = all_specs()
        for path in sorted((out / 'runs').glob('*.json')):
            if path.stem not in specs:
                refuse(f'Unexpected saved run: {path.name}')
            record = json.loads(path.read_text(encoding='utf-8'))
            validate_record(record, out, metadata, specs[path.stem], data[(record['dataset'], record['seed'])])
        return metadata  # The original manifest is immutable, including its timestamp.
    if out.exists() and any(out.iterdir()):
        refuse('Nonempty output directory has no manifest')
    for name in ['runs', 'predictions', 'splits', 'data']:
        (out / name).mkdir(exist_ok=True, parents=True)
    datasets = fetch_datasets(filter_data=tuple(DATASETS), download_if_missing=False)
    manifest, split_hashes = [], {}
    for name in DATASETS:
        d = datasets[name]
        x, y = np.asarray(d['data'], dtype=float), (d['target'] == 1).astype(int)
        assert np.isfinite(x).all()
        path = out / 'data' / f'{name}.npz'
        if not path.exists():
            np.savez_compressed(path, x=x, y=y)
        for seed in SEEDS:
            train, val, test = split_indices(y, seed)
            sp = out / 'splits' / f'{name}_{seed}.npz'
            if not sp.exists():
                np.savez_compressed(sp, train=train, validation=val, test=test)
            split_hashes[f'{name}_{seed}'] = sha(sp)
        manifest.append(dict(dataset=name, shape=x.shape, minority=int(y.sum()), sha256=sha(path), source='https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz', cache='<local path omitted>'))
    metadata = dict(schema_version=2, started_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'reproduction_protocol.md'),
                    official_sources={p.name:sha(p) for p in (ROOT / 'official_code' / 'src').glob('*.py')}, python=sys.version,
                    packages=package_versions(), datasets=manifest, splits_sha256=split_hashes,
                    max_workers=2, expected_fits=len(all_specs()), mode='R1_SUBSET_RETRAINING', note='No innovation Gate results are included.')
    write_json(out / 'manifest.json', metadata)
    return metadata


def select_configurations(frame, endpoint):
    """Select on validation only; tie policy was fixed before reading R1 outcomes."""
    tie_columns = ['_tie_hp', '_tie_ratio', '_tie_k']
    working = frame.assign(**{dest: frame[src].fillna(-1) for dest, src in zip(tie_columns, ['hp_id', 'ratio', 'k'])})
    selected = (working.sort_values(['validation.' + endpoint] + tie_columns, ascending=False, kind='stable')
                .groupby(['dataset', 'seed', 'model', 'method'], sort=True).head(1)
                .drop(columns=tie_columns).copy())
    selected['selection_endpoint'] = endpoint
    return selected


def summarize(out):
    specs = all_specs()
    expected_cells = {(d, seed, m, method) for d in DATASETS for seed in SEEDS for m in MODELS for method in METHODS}
    paths = sorted((out / 'runs').glob('*.json'))
    state = dict(records=len(paths), success=0, failed=0, expected=len(specs), complete=False,
                 full_summary_available=False, expected_cells=len(expected_cells), matched_cells=0,
                 tie_rule=TIE_RULE, aggregation_script_sha256=sha(__file__), audit_errors=[])
    records = []
    try:
        manifest = validate_manifest(out)
        state.update(training_script_sha256=manifest['script_sha256'], training_manifest_sha256=sha(out / 'manifest.json'))
        data = {(d, seed): load_verified_split(out, manifest, d, seed) for d in DATASETS for seed in SEEDS}
        for path in paths:
            if path.stem not in specs:
                refuse(f'Unexpected saved run: {path.name}')
            record = json.loads(path.read_text(encoding='utf-8'))
            records.append(validate_record(record, out, manifest, specs[path.stem], data[(record['dataset'], record['seed'])]))
    except Exception as error:
        state['audit_errors'].append(repr(error))
        state['status'] = 'PROVENANCE_REJECTED'
        write_json(out / 'status.json', state)
        # An older full summary must not masquerade as valid after failed verification.
        if (out / 'summary.csv').exists():
            (out / 'summary.csv').replace(out / 'summary.invalidated.csv')
        refuse(f'Artifact audit failed: {error}')

    successful = [r for r in records if r['status'] == 'success']
    actual_cells = {(r['dataset'], r['seed'], r['model'], r['method']) for r in successful}
    observed_ids = {r['run_id'] for r in records}
    state.update(success=len(successful), failed=len(records) - len(successful), matched_cells=len(actual_cells & expected_cells),
                 missing_cells=sorted(expected_cells - actual_cells), unexpected_cells=sorted(actual_cells - expected_cells),
                 missing_run_ids=sorted(set(specs) - observed_ids),
                 fit_cpu_seconds=sum(r.get('cpu_seconds', 0) for r in records),
                 fit_wall_seconds=sum(r.get('wall_seconds', 0) for r in records))
    complete = (observed_ids == set(specs) and len(records) == len(specs) and len(successful) == len(specs)
                and actual_cells == expected_cells)
    state.update(complete=complete, full_summary_available=complete, status='COMPLETE' if complete else 'INCOMPLETE')
    frame = pd.json_normalize(successful)
    frame.to_csv(out / 'all_fits.csv', index=False)
    if successful:
        all_selected = pd.concat([select_configurations(frame, endpoint) for endpoint in ENDPOINTS])
        all_selected.to_csv(out / 'selected.csv', index=False)
    else:
        all_selected = pd.DataFrame()
        all_selected.to_csv(out / 'selected.csv', index=False)
    if complete:
        # Every endpoint must contain precisely the same declared cells before means are taken.
        for _, selected in all_selected.groupby('selection_endpoint'):
            cells = set(selected[['dataset', 'seed', 'model', 'method']].itertuples(index=False, name=None))
            if cells != expected_cells or len(selected) != len(expected_cells):
                refuse('Selected endpoint does not cover the full matched grid')
        metrics_keys = ['test.' + x for x in ['auc', 'logloss', 'loss_minority', 'loss_majority', 'brier', 'ap', 'f1_fixed', 'f1_tuned']]
        by_seed = all_selected.groupby(['selection_endpoint', 'model', 'method', 'seed'])[metrics_keys].mean().reset_index()
        summary = by_seed.groupby(['selection_endpoint', 'model', 'method'])[metrics_keys].agg(['mean', 'std'])
        summary.to_csv(out / 'summary.csv')
    elif (out / 'summary.csv').exists():
        (out / 'summary.csv').replace(out / 'summary.invalidated.csv')
    write_json(out / 'status.json', state)
    write_json(out / 'aggregation_manifest.json', dict(training_manifest_sha256=sha(out / 'manifest.json'),
               training_script_sha256=manifest['script_sha256'], aggregation_script_sha256=sha(__file__),
               tie_rule=TIE_RULE, packages=package_versions(), full_summary_available=complete,
               legacy_snapshot=manifest['script_sha256'] != sha(__file__),
               note='Reproduction correction only; validation-only selection; original training manifest and run JSON files were not rewritten.'))
    print(json.dumps({k: v for k, v in state.items() if k not in ['missing_run_ids', 'missing_cells']}), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'r1', help='Use a new directory after any training-code or protocol change.')
    args = parser.parse_args()
    out = args.output
    if not args.summarize_only:
        prepare(out)
        jobs = [(d,m,s,str(out)) for d in DATASETS for s in SEEDS for m in MODELS]
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(execute_job, job): job for job in jobs}
            for fut in as_completed(futures):
                try:
                    outcome = fut.result()
                except ProvenanceError:
                    raise
                except Exception as error:
                    outcome = record_task_failure(futures[fut], error)
                print(json.dumps(outcome), flush=True)
                if time.perf_counter()-started > 4*3600:
                    for pending in futures:
                        pending.cancel()
                    # This cooperative check is secondary. The independent process-tree
                    # watchdog enforces hard wall/CPU/RAM/disk limits, including active fits.
                    write_json(out / 'termination.json', dict(status='TIME_BUDGET_EXCEEDED', completed_records=len(list((out / 'runs').glob('*.json')))))
                    summarize(out)
                    raise RuntimeError('TIME_BUDGET_EXCEEDED: stop incomplete run; preserve completed records')
    summarize(out)
