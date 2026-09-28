import numpy as np
import pytest

from adaptive_resampling import validation_policy_metrics
from confidence_guarded import attach_raw_metrics
from utility_guarded import tune_cost_threshold, utility_guarded_select


def test_cost_threshold_is_deterministic_and_uses_declared_grid():
    y = np.array([1, 1, 0, 0], dtype=int)
    probability = np.array([0.8, 0.7, 0.2, 0.1], dtype=float)
    first = tune_cost_threshold(y, probability)
    second = tune_cost_threshold(y, probability)
    assert first == second
    assert first in np.arange(0.01, 1.0, 0.01)


def test_utility_guard_rejects_metrics_without_raw_validation_inputs():
    y = np.array([1, 1, 0, 0], dtype=int)
    metrics = validation_policy_metrics(y, np.array([0.8, 0.7, 0.2, 0.1]), target_priors=(0.5,))
    with pytest.raises(ValueError, match="raw validation metrics"):
        utility_guarded_select({"none": metrics, "candidate": metrics}, candidate_order=("none", "candidate"))


def test_utility_guard_is_deterministic():
    y = np.array([1] * 20 + [0] * 80, dtype=int)
    baseline = np.where(y == 1, 0.20, 0.10)
    candidate = np.where(y == 1, 0.70, 0.10)
    metrics = {
        "none": attach_raw_metrics(validation_policy_metrics(y, baseline, target_priors=(0.05, 0.5)), y=y, probability=baseline),
        "candidate": attach_raw_metrics(validation_policy_metrics(y, candidate, target_priors=(0.05, 0.5)), y=y, probability=candidate),
    }
    first = utility_guarded_select(metrics, candidate_order=("none", "candidate"), bootstrap_replicates=200)
    second = utility_guarded_select(metrics, candidate_order=("none", "candidate"), bootstrap_replicates=200)
    assert first == second
    assert first["selection_rule"] == "utility_first_risk_guard"
