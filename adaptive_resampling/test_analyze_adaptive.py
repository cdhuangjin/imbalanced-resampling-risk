import numpy as np
import pandas as pd

from analyze_adaptive import (
    cluster_bootstrap_mean,
    gate_decision,
    task_selection_rates,
)
from run_adaptive import DEFAULT_TASKS, FULL_SEEDS, resolve_matrix


def test_cluster_bootstrap_is_deterministic_and_finite():
    values = np.array([-0.2, -0.1, 0.0, 0.1], dtype=float)

    first = cluster_bootstrap_mean(values, n_boot=1000, seed=7)
    second = cluster_bootstrap_mean(values, n_boot=1000, seed=7)

    assert first == second
    assert first[1] <= first[0] <= first[2]
    assert all(np.isfinite(value) for value in first)


def test_task_selection_rates_require_exact_task_coverage():
    decisions = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "selected_candidate": ["none", "smote_q05_k5", "none", "none"],
        }
    )

    rates = task_selection_rates(decisions, expected_tasks=("a", "b"))

    assert rates == {"a": 0.5, "b": 0.0}


def test_task_selection_rates_reject_missing_or_extra_tasks():
    decisions = pd.DataFrame(
        {"dataset": ["a"], "selected_candidate": ["none"]}
    )

    assert task_selection_rates(decisions, expected_tasks=("a", "b")) is None
    assert task_selection_rates(
        pd.DataFrame({"dataset": ["a", "b", "c"], "selected_candidate": ["none"] * 3}),
        expected_tasks=("a", "b"),
    ) is None


def test_gate_decision_requires_all_conditions_and_complete_artifacts():
    assert gate_decision(
        complete=True,
        selection_task_count=4,
        adaptive_smote_majority_logloss_pass=True,
        adaptive_smote_brier_pass=True,
        minority_task_count=4,
        minority_ci_pass=True,
    ) == "GO"
    assert gate_decision(
        complete=True,
        selection_task_count=4,
        adaptive_smote_majority_logloss_pass=True,
        adaptive_smote_brier_pass=False,
        minority_task_count=4,
        minority_ci_pass=True,
    ) == "NO-GO"
    assert gate_decision(
        complete=False,
        selection_task_count=4,
        adaptive_smote_majority_logloss_pass=True,
        adaptive_smote_brier_pass=True,
        minority_task_count=4,
        minority_ci_pass=True,
    ) == "INCONCLUSIVE"


def test_full_matrix_rejects_partial_overrides():
    try:
        resolve_matrix(
            full=True,
            datasets_arg="ecoli",
            seeds_arg=None,
            gate_decision="GO",
        )
    except ValueError as error:
        assert "full" in str(error).lower()
    else:
        raise AssertionError("partial --full override should be rejected")

    datasets, seeds = resolve_matrix(
        full=True,
        datasets_arg=None,
        seeds_arg=None,
        gate_decision="GO",
    )
    assert datasets == DEFAULT_TASKS
    assert seeds == FULL_SEEDS
