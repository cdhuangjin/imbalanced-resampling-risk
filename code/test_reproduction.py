import numpy as np
import json
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score
import run_reproduction as runner
from run_reproduction import threshold_from_validation, split_indices, metrics, target_count


def test_threshold_matches_reference_and_does_not_accept_test_labels():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.03, 0.12, 0.19, 0.20, 0.21])
    scores = [f1_score(y, p > t, zero_division=0) for t in np.arange(0, 1, .01)]
    expected = np.arange(0, 1, .01)[np.argmax(scores)]
    assert threshold_from_validation(y, p) == expected
    assert threshold_from_validation(np.array([1, 0]), np.array([0., 1.])) == .99


def test_split_disjoint_stratified_and_repeatable():
    y = np.array([0] * 180 + [1] * 20)
    a = split_indices(y, 4)
    b = split_indices(y, 4)
    assert [len(x) for x in a] == [120, 40, 40]
    assert len(set(a[0]) & set(a[1])) == len(set(a[0]) & set(a[2])) == len(set(a[1]) & set(a[2])) == 0
    assert set(np.concatenate(a)) == set(range(200))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_auc_uses_probabilities_and_class_losses_reconstruct_total():
    y = np.array([0, 0, 0, 1])
    p = np.array([.01, .05, .1, .2])
    m = metrics(y, p, .1)
    assert m['auc'] == 1.0
    assert roc_auc_score(y, p > .5) == .5
    assert np.isclose(m['logloss'], .75 * m['loss_majority'] + .25 * m['loss_minority'])
    assert target_count(np.array([0] * 80 + [1] * 20), .5) == 80


@pytest.fixture
def toy_output(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    out = root / 'outputs' / 'trial'
    (root / 'official_code' / 'src').mkdir(parents=True)
    (root / 'reproduction_protocol.md').write_text('fixed test protocol', encoding='utf-8')
    for sub in ['data', 'splits', 'runs', 'predictions']:
        (out / sub).mkdir(parents=True)
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'DATASETS', ['toy'])
    monkeypatch.setattr(runner, 'SEEDS', (0,), raising=False)
    x = np.arange(240, dtype=float).reshape(120, 2)
    y = np.array([0] * 100 + [1] * 20)
    np.savez_compressed(out / 'data' / 'toy.npz', x=x, y=y)
    tr, va, te = split_indices(y, 0)
    np.savez_compressed(out / 'splits' / 'toy_0.npz', train=tr, validation=va, test=te)
    meta = dict(schema_version=2, script_sha256=runner.sha(runner.__file__),
                protocol_sha256=runner.sha(root / 'reproduction_protocol.md'),
                official_sources={}, packages=runner.package_versions() if hasattr(runner, 'package_versions') else {},
                datasets=[dict(dataset='toy', shape=list(x.shape), minority=int(y.sum()), sha256=runner.sha(out / 'data' / 'toy.npz'))],
                splits_sha256={'toy_0': runner.sha(out / 'splits' / 'toy_0.npz')}, expected_fits=56)
    (out / 'manifest.json').write_text(json.dumps(meta), encoding='utf-8')
    return out


def test_selection_ties_use_full_config_and_ignore_input_order():
    rows = [dict(dataset='toy', seed=0, model='dt', method='smote', hp_id=h, ratio=r, k=k,
                 **{'validation.auc': .8, 'test.auc': test})
            for h, r, k, test in [(3, .2, 5, .99), (2, .5, 5, .98), (3, .5, 3, .97), (3, .5, 5, .1)]]
    for values in [rows, list(reversed(rows))]:
        chosen = runner.select_configurations(pd.DataFrame(values), 'auc').iloc[0]
        assert (chosen.hp_id, chosen.ratio, chosen.k) == (3, .5, 5)
        assert chosen['test.auc'] == .1  # Deliberately does not maximize the test metric.


def test_prepare_refuses_changed_code_without_overwriting_manifest(toy_output):
    path = toy_output / 'manifest.json'
    meta = json.loads(path.read_text())
    meta['script_sha256'] = 'different-code'
    path.write_text(json.dumps(meta), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='new output|新输出'):
        runner.prepare(toy_output)
    assert path.read_bytes() == before


def test_sampler_failure_is_recorded_for_each_hp_and_job_continues(toy_output, monkeypatch):
    class FailingSampler:
        def __init__(self, **kwargs):
            pass
        def fit_resample(self, x, y):
            raise ValueError('deliberate sampler failure')
    monkeypatch.setattr(runner, 'SMOTE', FailingSampler)
    result = runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    records = [json.loads(p.read_text()) for p in (toy_output / 'runs').glob('*.json')]
    assert result['count'] == len(records) == 28
    assert sum(r['status'] == 'failed' for r in records) == 16
    assert all(r['failure_stage'] == 'resampling' for r in records if r['status'] == 'failed')
    assert result['success'] == 12


@pytest.fixture
def fake_cat(monkeypatch):
    calls = []
    class FakeCat:
        tree_count_ = 1
        def predict_proba(self, x, **kwargs):
            calls.append(kwargs.get('thread_count'))
            p = np.clip(x[:, 0] / 240, .01, .99)
            return np.column_stack([1 - p, p])
        def get_all_params(self):
            return {'thread_count': 1}
    monkeypatch.setattr(runner, '_fit_catboost', lambda *args: FakeCat())
    return calls


def test_cat_predictions_explicitly_use_one_thread(toy_output, fake_cat):
    result = runner.execute_job(('toy', 'cat', 0, str(toy_output)))
    assert result['success'] == 28
    assert fake_cat == [1] * 56


@pytest.mark.parametrize('artifact', ['data', 'split', 'prediction', 'protocol'])
def test_resume_rejects_changed_artifacts(toy_output, artifact):
    runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    manifest_before = (toy_output / 'manifest.json').read_bytes()
    targets = {'data': toy_output / 'data' / 'toy.npz',
               'split': toy_output / 'splits' / 'toy_0.npz',
               'prediction': next((toy_output / 'predictions').glob('*.npz')),
               'protocol': runner.ROOT / 'reproduction_protocol.md'}
    path = targets[artifact]
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='new output|新输出'):
        runner.prepare(toy_output)
    assert (toy_output / 'manifest.json').read_bytes() == manifest_before


def test_incomplete_and_empty_outputs_keep_status_without_full_summary(toy_output):
    empty = runner.summarize(toy_output)
    assert empty['complete'] is False
    assert not (toy_output / 'summary.csv').exists()
    runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    partial = runner.summarize(toy_output)
    assert partial['complete'] is False
    assert partial['matched_cells'] == 3
    assert partial['expected_cells'] == 6
    assert not (toy_output / 'summary.csv').exists()
    assert (toy_output / 'status.json').exists()


def test_job_level_failure_preserves_expected_failed_records(toy_output, monkeypatch):
    monkeypatch.setattr(runner, 'split_indices', lambda *args: (_ for _ in ()).throw(ValueError('job error')))
    result = runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    records = [json.loads(p.read_text()) for p in (toy_output / 'runs').glob('*.json')]
    assert result['success'] == 0
    assert len(records) == 28
    assert all(r['failure_stage'] == 'task' for r in records)


def test_full_matched_grid_summarizes_without_mutating_training_artifacts(toy_output, fake_cat):
    runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    runner.execute_job(('toy', 'cat', 0, str(toy_output)))
    manifest_before = (toy_output / 'manifest.json').read_bytes()
    run_hashes = {p.name: runner.sha(p) for p in (toy_output / 'runs').glob('*.json')}
    state = runner.summarize(toy_output)
    assert state['complete'] is True
    assert state['success'] == state['expected'] == 56
    assert state['matched_cells'] == state['expected_cells'] == 6
    assert (toy_output / 'summary.csv').exists()
    selected = pd.read_csv(toy_output / 'selected.csv')
    assert len(selected) == 3 * 6
    assert (toy_output / 'manifest.json').read_bytes() == manifest_before
    assert {p.name: runner.sha(p) for p in (toy_output / 'runs').glob('*.json')} == run_hashes


def test_verified_legacy_snapshot_allows_analysis_but_never_training_resume(toy_output):
    runner.execute_job(('toy', 'dt', 0, str(toy_output)))
    snapshot = toy_output / 'run_reproduction_training_snapshot.py'
    snapshot.write_bytes(b'previously verified legacy training source')
    manifest_path = toy_output / 'manifest.json'
    meta = json.loads(manifest_path.read_text())
    meta.pop('schema_version')
    meta.pop('splits_sha256')
    meta['script_sha256'] = runner.sha(snapshot)
    manifest_path.write_text(json.dumps(meta), encoding='utf-8')
    # Original v1 run records had no per-record code/protocol/manifest hashes.
    for path in (toy_output / 'runs').glob('*.json'):
        record = json.loads(path.read_text())
        for key in ['manifest_sha256', 'script_sha256', 'protocol_sha256']:
            record.pop(key)
        path.write_text(json.dumps(record), encoding='utf-8')
    original = manifest_path.read_bytes()
    state = runner.summarize(toy_output)
    assert state['success'] == 28
    assert state['training_script_sha256'] == runner.sha(snapshot)
    with pytest.raises(RuntimeError, match='new output'):
        runner.prepare(toy_output)
    assert manifest_path.read_bytes() == original
    snapshot.write_bytes(b'tampered snapshot')
    with pytest.raises(RuntimeError, match='new output'):
        runner.summarize(toy_output)
    assert json.loads((toy_output / 'status.json').read_text())['status'] == 'PROVENANCE_REJECTED'


def test_tie_sort_accepts_missing_sampling_parameters():
    rows = [dict(dataset='toy', seed=0, model='dt', method='random', hp_id=3, ratio=r, k=None,
                 **{'validation.f1_fixed': .5}) for r in [.2, .5]]
    row = runner.select_configurations(pd.DataFrame(rows), 'f1_fixed').iloc[0]
    assert row.ratio == .5
    assert pd.isna(row.k)
