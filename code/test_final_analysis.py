import numpy as np

import analysis_final


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = analysis_final.holm_adjust([0.01, 0.04, 0.20])

    assert np.allclose(adjusted, [0.03, 0.08, 0.20])
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_cluster_bootstrap_is_deterministic():
    values = np.array([0.1, 0.2, 0.3, 0.4])

    first = analysis_final.cluster_bootstrap_mean(values, n_boot=2000, seed=17)
    second = analysis_final.cluster_bootstrap_mean(values, n_boot=2000, seed=17)

    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_transfer_classification_follows_preregistered_rules():
    assert analysis_final.classify_transfer(
        auc_ci=(0.01, 0.04), minority_relative_ci=(0.08, 0.20), majority_delta=0.1
    ) == "ranking-transfer+tradeoff"
    assert analysis_final.classify_transfer(
        auc_ci=(-0.01, 0.01), minority_relative_ci=(0.08, 0.20), majority_delta=0.0
    ) == "calibration/decision-transfer"
