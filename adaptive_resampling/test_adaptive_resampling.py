import numpy as np
import pytest

from adaptive_resampling import (
    minority_support_scores,
    select_risk_constrained_candidate,
    support_gated_smote,
    validation_policy_metrics,
)


def _toy_train():
    x = np.array(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5],
            [10.0],
            [10.1],
            [10.2],
            [10.3],
            [10.4],
            [10.5],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int)
    return x, y


def test_minority_support_scores_identify_an_isolated_minority_seed():
    x, y = _toy_train()

    scores = minority_support_scores(x, y, k=5)

    assert scores.shape == (len(y),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scores[0] == pytest.approx(0.0)
    assert np.all(scores[[6, 7, 8, 9]] >= 0.5)


def test_support_gated_smote_preserves_original_rows_and_is_deterministic():
    x, y = _toy_train()

    first_x, first_y, first_diag = support_gated_smote(
        x, y, ratio=1.0, k=5, min_support=0.5, seed=13
    )
    second_x, second_y, second_diag = support_gated_smote(
        x, y, ratio=1.0, k=5, min_support=0.5, seed=13
    )

    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert first_diag == second_diag
    assert first_diag["generated_count"] > 0
    assert first_diag["eligible_seed_count"] == 4
    assert np.array_equal(first_x[: len(x)], x)
    assert np.array_equal(first_y[: len(y)], y)
    assert set(np.unique(first_y)) == {0, 1}


def test_validation_policy_metrics_are_finite_and_prior_weighted():
    y = np.array([1, 1, 0, 0], dtype=int)
    probability = np.array([0.8, 0.7, 0.2, 0.1], dtype=float)

    metrics = validation_policy_metrics(y, probability, target_priors=(0.05, 0.5))

    assert set(metrics) == {0.05, 0.5}
    for prior_metrics in metrics.values():
        assert set(prior_metrics) == {
            "minority_logloss",
            "majority_logloss",
            "brier",
        }
        assert all(np.isfinite(value) for value in prior_metrics.values())
    assert metrics[0.05]["minority_logloss"] == metrics[0.5]["minority_logloss"]
    assert metrics[0.05]["majority_logloss"] == metrics[0.5]["majority_logloss"]
    assert metrics[0.05]["brier"] != metrics[0.5]["brier"]


def test_selector_rejects_majority_or_brier_constraint_violations():
    y = np.array([1, 1, 0, 0], dtype=int)
    none = np.array([0.7, 0.6, 0.2, 0.1], dtype=float)
    risky = np.array([0.95, 0.9, 0.45, 0.4], dtype=float)
    metrics = {
        "none": validation_policy_metrics(y, none, target_priors=(0.05, 0.5)),
        "risky": validation_policy_metrics(y, risky, target_priors=(0.05, 0.5)),
    }

    result = select_risk_constrained_candidate(
        metrics,
        candidate_order=("none", "risky"),
        majority_logloss_tolerance=0.02,
        brier_tolerance=0.005,
    )

    assert result["selected"] == "none"
    assert result["feasibility"]["risky"] is False


def test_selector_returns_none_when_no_candidate_is_feasible():
    y = np.array([1, 1, 0, 0], dtype=int)
    none = np.array([0.7, 0.6, 0.2, 0.1], dtype=float)
    metrics = {
        "none": validation_policy_metrics(y, none, target_priors=(0.05,)),
        "candidate": validation_policy_metrics(y, none, target_priors=(0.05,)),
    }
    metrics["none"][0.05]["majority_logloss"] += 1.0
    metrics["candidate"][0.05]["majority_logloss"] += 1.0

    result = select_risk_constrained_candidate(
        metrics,
        candidate_order=("none", "candidate"),
        majority_logloss_tolerance=0.02,
        brier_tolerance=0.005,
    )

    assert result["selected"] == "none"


def test_selector_tie_breaks_by_candidate_order():
    y = np.array([1, 1, 0, 0], dtype=int)
    probability = np.array([0.7, 0.6, 0.2, 0.1], dtype=float)
    base = validation_policy_metrics(y, probability, target_priors=(0.05,))
    candidate_b = validation_policy_metrics(y, probability, target_priors=(0.05,))
    candidate_a = validation_policy_metrics(y, probability, target_priors=(0.05,))
    candidate_b[0.05]["minority_logloss"] -= 0.1
    candidate_a[0.05]["minority_logloss"] -= 0.1
    metrics = {"none": base, "candidate_b": candidate_b, "candidate_a": candidate_a}

    result = select_risk_constrained_candidate(
        metrics,
        candidate_order=("none", "candidate_b", "candidate_a"),
        majority_logloss_tolerance=0.02,
        brier_tolerance=0.005,
    )

    assert result["selected"] == "candidate_b"
