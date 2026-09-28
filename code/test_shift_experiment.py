import numpy as np

import shift_experiment


def test_shift_specs_include_iid_and_three_severity_levels():
    specs = shift_experiment.build_shift_specs(
        datasets=("d1",),
        learners=("dt_leaf01",),
        methods=("none",),
        seeds=(0,),
    )

    assert len(specs) == 7
    assert specs[0]["run_id"] == "d1__dt_leaf01__s0__none__iid__sev0"
    assert [s["scenario"] for s in specs] == [
        "iid",
        "label_shift",
        "label_shift",
        "label_shift",
        "covariate_shift",
        "covariate_shift",
        "covariate_shift",
    ]
    assert [s["severity"] for s in specs] == [0, 0, 1, 2, 0, 1, 2]


def test_target_prior_and_covariate_shift_are_frozen_and_local():
    assert shift_experiment.target_prior(0) == 0.05
    assert shift_experiment.target_prior(1) == 0.20
    assert shift_experiment.target_prior(2) == 0.50

    x = np.arange(12, dtype=float).reshape(4, 3)
    shifted, feature_index = shift_experiment.apply_covariate_shift(
        x, mean=np.zeros(3), scale=np.ones(3), severity=2
    )
    assert feature_index == 0
    assert np.allclose(shifted[:, 0], x[:, 0] + 2)
    assert np.allclose(shifted[:, 1:], x[:, 1:])


def test_label_shift_weights_recover_declared_target_prior():
    y = np.array([0, 0, 0, 1, 1])
    weights = shift_experiment.source_prior_weights(y, target_pi=0.20)

    assert np.isclose(weights[y == 1].sum() / weights.sum(), 0.20)
    metrics = shift_experiment.compute_shift_metrics(
        y,
        np.array([0.1, 0.2, 0.3, 0.7, 0.8]),
        threshold=0.5,
        sample_weight=weights,
    )
    assert set(metrics) == {
        "auc",
        "ap",
        "logloss",
        "loss_minority",
        "loss_majority",
        "brier",
        "f1_fixed",
        "f1_tuned",
    }
