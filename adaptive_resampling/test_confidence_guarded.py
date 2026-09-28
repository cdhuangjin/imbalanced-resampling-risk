import numpy as np
import pytest

from confidence_guarded import attach_raw_metrics, confidence_guarded_select
from adaptive_resampling import validation_policy_metrics


def _bundle(y, probability):
    return attach_raw_metrics(
        validation_policy_metrics(y, probability, target_priors=(0.05, 0.20, 0.50)),
        y=y,
        probability=probability,
    )


def test_confidence_guard_is_deterministic_and_records_rule():
    y = np.array([1] * 20 + [0] * 80, dtype=int)
    baseline = np.where(y == 1, 0.20, 0.10)
    safer = np.where(y == 1, 0.70, 0.10)
    metrics = {"none": _bundle(y, baseline), "safer": _bundle(y, safer)}
    first = confidence_guarded_select(
        metrics,
        candidate_order=("none", "safer"),
        bootstrap_replicates=200,
    )
    second = confidence_guarded_select(
        metrics,
        candidate_order=("none", "safer"),
        bootstrap_replicates=200,
    )
    assert first == second
    assert first["selection_rule"] == "row_bootstrap_upper_confidence_guard"


def test_confidence_guard_rejects_missing_raw_validation_inputs():
    y = np.array([1, 1, 0, 0], dtype=int)
    metrics = validation_policy_metrics(y, np.array([0.7, 0.6, 0.2, 0.1]), target_priors=(0.5,))
    with pytest.raises(ValueError, match="raw validation metrics"):
        confidence_guarded_select(candidate_metrics={"none": metrics, "candidate": metrics}, candidate_order=("none", "candidate"))
