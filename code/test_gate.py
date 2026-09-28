import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_gate


def test_gate_grid_is_locked():
    assert run_gate.DATASETS == ('ecoli', 'yeast_me2', 'abalone_19', 'mammography', 'oil', 'spectrometer')
    assert len(run_gate.CONFIGS) == 4
    assert run_gate.METHODS == ('none', 'random', 'smote')
    assert len(run_gate.all_run_ids()) == 360


def test_bootstrap_is_deterministic_and_clustered():
    values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    first = run_gate.cluster_bootstrap_ci(values, seed=20260913, n_boot=1000)
    second = run_gate.cluster_bootstrap_ci(values, seed=20260913, n_boot=1000)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_decide_gate_requires_all_conditions():
    good = dict(contrast_mean=0.02, contrast_ci_low=0.005, positive_datasets=4,
                minority_relative_improvements=[0.06, 0.07, 0.08, 0.09, 0.01, 0.02],
                complete=True, all_hashes_valid=True, leakage_free=True, resource_ok=True,
                expected_records=360, successful_records=360)
    result = run_gate.decide_gate(good)
    assert result['decision'] == 'GO'
    bad = dict(good, contrast_ci_low=-0.001)
    assert run_gate.decide_gate(bad)['decision'] == 'NO-GO'
