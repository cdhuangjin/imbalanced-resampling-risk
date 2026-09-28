from __future__ import annotations

import numpy as np

from run_residual_shift_gate import (
    GATE_TASKS,
    METRICS,
    POSTPROCESSORS,
    STABILITY_POSTPROCESSORS,
    REGIME_POSTPROCESSORS,
    build_fit_specs,
    build_scenarios,
    prepare_scenario,
)


def test_scenarios_are_frozen_and_cover_all_shift_cells():
    scenarios = build_scenarios()
    assert len(scenarios) == 10
    assert [item["name"] for item in scenarios] == [
        "iid",
        "label_shift",
        "label_shift",
        "label_shift",
        "covariate_shift",
        "covariate_shift",
        "joint_shift",
        "joint_shift",
        "joint_shift",
        "joint_shift",
    ]
    assert [item["target_prior"] for item in scenarios[1:4]] == [0.05, 0.20, 0.50]
    assert {(item["target_prior"], item["covariate_severity"]) for item in scenarios[-4:]} == {
        (0.20, 1), (0.20, 2), (0.50, 1), (0.50, 2)
    }


def test_gate_fit_matrix_has_108_fits_and_expected_rows():
    specs = build_fit_specs(GATE_TASKS, seeds=(0, 1, 2))
    assert len(specs) == 108
    assert len(specs) * len(build_scenarios()) * len(POSTPROCESSORS) * len(METRICS) == 25920


def test_stability_variant_adds_one_auditable_postprocessor():
    assert STABILITY_POSTPROCESSORS[-2] == "platt_target_rsc"
    assert len(STABILITY_POSTPROCESSORS) == len(POSTPROCESSORS) + 2


def test_regime_variant_adds_auditable_policy_switch():
    assert REGIME_POSTPROCESSORS[-2] == "platt_target_rass"
    assert len(REGIME_POSTPROCESSORS) == len(STABILITY_POSTPROCESSORS) + 1


def test_prepare_scenario_returns_unlabeled_target_features_and_correct_shift():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(60, 3))
    y = np.array([0] * 45 + [1] * 15)
    scenario = build_scenarios()[-1]
    prepared = prepare_scenario(
        x,
        y,
        scenario,
        train_mean=np.zeros(3),
        train_scale=np.ones(3),
        seed=10,
    )
    assert prepared["target_features"].shape[1] == 3
    assert prepared["target_features"].shape[0] == 2000
    assert np.isclose(prepared["target_prior"], 0.50)
    assert not np.allclose(prepared["x_eval"], x)
    assert prepared["evaluation_weights"].shape == y.shape


def test_prepare_scenario_is_deterministic():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(60, 3))
    y = np.array([0] * 45 + [1] * 15)
    scenario = build_scenarios()[2]
    kwargs = dict(train_mean=np.zeros(3), train_scale=np.ones(3), seed=11)
    first = prepare_scenario(x, y, scenario, **kwargs)
    second = prepare_scenario(x, y, scenario, **kwargs)
    np.testing.assert_array_equal(first["target_features"], second["target_features"])
    np.testing.assert_array_equal(first["evaluation_weights"], second["evaluation_weights"])
