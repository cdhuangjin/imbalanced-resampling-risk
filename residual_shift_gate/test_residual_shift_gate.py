from __future__ import annotations

import inspect

import numpy as np
import pytest

from residual_shift_gate import (
    cross_fitted_domain_auc,
    gate_consensus_probability,
    shrink_consensus_probability,
    uncertainty_aware_shrinkage,
    regime_adaptive_shrinkage,
)


def test_domain_auc_is_deterministic_and_accepts_matching_distributions():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 4))
    target = rng.normal(size=(80, 4))
    first = cross_fitted_domain_auc(source, target, seed=3)
    second = cross_fitted_domain_auc(source, target, seed=3)
    assert first == second
    assert first["valid"]
    assert 0.5 <= first["residual_auc"] <= 1.0


def test_domain_auc_rejects_a_clear_residual_shift():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(100, 3))
    target = rng.normal(loc=3.0, size=(100, 3))
    result = cross_fitted_domain_auc(source, target, seed=3)
    assert result["valid"]
    assert result["residual_auc"] > 0.60
    assert not result["accepted"]
    assert result["reason"] == "residual_shift_detected"


def test_invalid_domain_input_is_a_conservative_rejection():
    result = cross_fitted_domain_auc(
        np.ones((2, 2)), np.ones((2, 2)), seed=1, n_splits=3
    )
    assert result["valid"] is False
    assert result["accepted"] is False
    assert result["residual_auc"] is None


def test_gate_selects_consensus_only_when_diagnostic_accepts():
    source = np.array([0.1, 0.2, 0.8])
    consensus = np.array([0.2, 0.3, 0.9])
    accepted = gate_consensus_probability(
        source, consensus, {"valid": True, "residual_auc": 0.60, "reason": "accepted"}
    )
    rejected = gate_consensus_probability(
        source,
        consensus,
        {"valid": True, "residual_auc": 0.61, "reason": "residual_shift_detected"},
    )
    np.testing.assert_allclose(accepted["probability"], consensus)
    np.testing.assert_allclose(rejected["probability"], source)
    assert accepted["used_fallback"] is False
    assert rejected["used_fallback"] is True


def test_public_detector_signature_has_no_target_label_argument():
    parameters = inspect.signature(cross_fitted_domain_auc).parameters
    assert "target_label" not in parameters
    assert "target_labels" not in parameters


def test_probability_validation_is_explicit():
    with pytest.raises(ValueError, match="finite one-dimensional"):
        gate_consensus_probability(
            np.array([[0.2]]), np.array([0.3]), {"valid": True, "residual_auc": 0.5}
        )


def test_shrinkage_interpolates_at_frozen_residual_knots():
    source = np.array([0.1, 0.2])
    consensus = np.array([0.5, 0.8])
    low = shrink_consensus_probability(
        source, consensus, {"valid": True, "residual_auc": 0.55}
    )
    middle = shrink_consensus_probability(
        source, consensus, {"valid": True, "residual_auc": 0.65}
    )
    high = shrink_consensus_probability(
        source, consensus, {"valid": True, "residual_auc": 0.75}
    )
    np.testing.assert_allclose(low["probability"], consensus)
    np.testing.assert_allclose(middle["probability"], (source + consensus) / 2.0)
    np.testing.assert_allclose(high["probability"], source)
    assert low["adaptation_weight"] == 1.0
    assert middle["adaptation_weight"] == 0.5
    assert high["adaptation_weight"] == 0.0


def test_uncertainty_aware_shrinkage_multiplies_residual_and_stability_weights():
    source = np.array([0.1, 0.2])
    consensus = np.array([0.5, 0.8])
    result = uncertainty_aware_shrinkage(
        source,
        consensus,
        {"valid": True, "residual_auc": 0.55},
        {"stability_weight": 0.5},
    )
    np.testing.assert_allclose(result["probability"], source + 0.5 * (consensus - source))
    assert result["residual_adaptation_weight"] == 1.0
    assert result["stability_weight"] == 0.5
    assert result["adaptation_weight"] == 0.5


def test_uncertainty_aware_shrinkage_falls_back_on_invalid_stability():
    source = np.array([0.1, 0.2])
    consensus = np.array([0.5, 0.8])
    result = uncertainty_aware_shrinkage(
        source,
        consensus,
        {"valid": True, "residual_auc": 0.55},
        {"stability_weight": None},
    )
    np.testing.assert_allclose(result["probability"], source)
    assert result["adaptation_weight"] == 0.0


def test_regime_adaptive_policy_keeps_v51_in_accepted_regime():
    source = np.array([0.1, 0.2])
    v51 = np.array([0.3, 0.4])
    rsc = np.array([0.2, 0.3])
    result = regime_adaptive_shrinkage(
        v51, rsc, {"valid": True, "residual_auc": 0.60}
    )
    np.testing.assert_allclose(result["probability"], v51)
    assert result["selected_policy"] == "v51_shrinkage"


def test_regime_adaptive_policy_uses_rsc_after_frozen_shift_threshold():
    v51 = np.array([0.3, 0.4])
    rsc = np.array([0.2, 0.3])
    result = regime_adaptive_shrinkage(
        v51, rsc, {"valid": True, "residual_auc": 0.61}
    )
    np.testing.assert_allclose(result["probability"], rsc)
    assert result["selected_policy"] == "rsc"


def test_regime_adaptive_policy_falls_back_to_v51_on_invalid_detector():
    v51 = np.array([0.3, 0.4])
    rsc = np.array([0.2, 0.3])
    result = regime_adaptive_shrinkage(
        v51, rsc, {"valid": False, "residual_auc": None}
    )
    np.testing.assert_allclose(result["probability"], v51)
    assert result["selected_policy"] == "v51_shrinkage"
