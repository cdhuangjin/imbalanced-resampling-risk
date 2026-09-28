from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from brf_experiment import METHODS, fit_brf, tune_cost_threshold  # noqa: E402
from analyze_brf_gate import validate_frames  # noqa: E402


def test_brf_fit_has_finite_probabilities():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(60, 5))
    y = np.array([0, 1] * 30)
    model = fit_brf(x[:40], y[:40], seed=0, n_estimators=8)
    p = np.asarray(model.predict_proba(x[40:]))[:, 1]
    assert p.shape == (20,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_brf_protocol_has_three_frozen_treatments():
    assert METHODS == ("none", "random_q05", "smote_q05_k5")
    assert tune_cost_threshold(np.array([0, 1]), np.array([0.2, 0.8])) in np.arange(0, 1, 0.01)


def test_analyzer_requires_brf_and_xgb_coverage():
    empty = pd.DataFrame(columns=["run_id", "dataset", "learner", "method", "seed", "metric", "value", "config_hash", "data_hash", "split_hash"])
    try:
        validate_frames(empty, empty)
    except ValueError as exc:
        assert "BRF" in str(exc) or "XGBoost" in str(exc)
    else:
        raise AssertionError("incomplete comparator coverage was accepted")
