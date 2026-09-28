from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from cost_sensitive_xgb_experiment import (  # noqa: E402
    DECISION_COSTS,
    METHOD_PARAMS,
    build_cost_weights,
    tune_cost_threshold,
)


def test_cost_weights_match_declared_target_prior_using_training_labels_only():
    y = np.array([0] * 8 + [1] * 2)
    weights = build_cost_weights(y, target_ratio=0.5)
    effective_prior = weights[y == 1].sum() / weights.sum()
    assert np.isclose(effective_prior, 0.5)
    assert np.all(weights[y == 0] == 1.0)
    assert np.all(np.isfinite(weights))


def test_cost_threshold_is_deterministic_and_uses_validation_only():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.9])
    first = tune_cost_threshold(y, p, fn_cost=5.0, fp_cost=1.0)
    second = tune_cost_threshold(y, p, fn_cost=5.0, fp_cost=1.0)
    assert first == second
    assert 0.0 <= first <= 0.99


def test_protocol_has_four_frozen_treatments_and_explicit_costs():
    assert tuple(METHOD_PARAMS) == (
        "none",
        "random_q05",
        "smote_q05_k5",
        "cost_weight_q05",
    )
    assert DECISION_COSTS == {"fn": 5.0, "fp": 1.0}


def test_analyzer_rejects_missing_decision_utility_metric():
    from analyze_cost_sensitive_xgb import validate_raw  # noqa: E402

    frame = pd.DataFrame(
        [
            {
                "run_id": "a",
                "dataset": "ecoli",
                "learner": "xgb_gate",
                "method": "none",
                "seed": 0,
                "metric": "auc",
                "value": 0.5,
                "config_hash": "c",
                "data_hash": "d",
                "split_hash": "s",
            }
        ]
    )
    try:
        validate_raw(frame)
    except ValueError as exc:
        assert "decision_utility_c5" in str(exc)
    else:
        raise AssertionError("validation accepted incomplete utility coverage")
