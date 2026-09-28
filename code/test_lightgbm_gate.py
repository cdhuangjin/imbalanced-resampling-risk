from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from lgbm_experiment import METHODS, fit_lgbm  # noqa: E402
from analyze_lightgbm_gate import validate_frames  # noqa: E402


def test_lightgbm_fit_has_finite_probabilities():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(60, 5))
    y = np.array([0, 1] * 30)
    model = fit_lgbm(x[:40], y[:40], x[40:50], y[40:50], seed=0, n_estimators=8)
    p = np.asarray(model.predict_proba(x[50:]))[:, 1]
    assert p.shape == (10,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_lightgbm_protocol_reuses_frozen_treatments():
    assert METHODS == ("none", "random_q05", "smote_q05_k5")


def test_analyzer_rejects_empty_comparator():
    empty = pd.DataFrame()
    try:
        validate_frames(empty, empty)
    except ValueError as exc:
        assert "LightGBM" in str(exc) or "XGBoost" in str(exc)
    else:
        raise AssertionError("empty learner comparator was accepted")
