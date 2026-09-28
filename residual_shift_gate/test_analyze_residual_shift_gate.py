from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_residual_shift_gate import (
    GATED,
    REFERENCE,
    build_behavior,
    build_effects,
    bootstrap_task_ci,
    gate_decision,
    threshold_sensitivity,
)


def _synthetic_frame():
    rows = []
    for dataset in ("a", "b", "c"):
        for method in ("none", "random_q05", "smote_q05_k5"):
            for seed in (0, 1):
                for scenario, prior, accepted in (
                    ("label_shift", 0.20, True),
                    ("label_shift", 0.50, True),
                    ("covariate_shift", 0.10, False),
                    ("joint_shift", 0.50, False),
                ):
                    for postprocessor in ("platt_source", "platt_target_consensus", "platt_target_gated", "platt_target_shrinkage", "platt_target_oracle"):
                        for metric in ("brier", "logloss", "auc", "loss_minority", "loss_majority", "ece_10"):
                            base = 0.2 if metric == "brier" else 1.0
                            delta = -0.1 if accepted and postprocessor in ("platt_target_consensus", "platt_target_gated", "platt_target_shrinkage") else 0.0
                            if scenario in ("covariate_shift", "joint_shift") and postprocessor == "platt_target_consensus":
                                delta = 2.0 if metric == "logloss" else 0.2
                            if scenario in ("covariate_shift", "joint_shift") and postprocessor == "platt_target_gated":
                                delta = 0.0
                            value = base + delta
                            rows.append(
                                {
                                    "fit_id": f"{dataset}__{method}__s{seed}",
                                    "dataset": dataset,
                                    "learner": "dt_leaf01",
                                    "method": method,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "severity": 0,
                                    "target_prior": prior,
                                    "covariate_severity": 1,
                                    "postprocessor": postprocessor,
                                    "metric": metric,
                                    "value": value,
                                    "residual_auc": 0.55 if accepted else 0.80,
                                    "residual_valid": True,
                                }
                            )
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_and_clustered():
    values = pd.Series([1.0, 2.0, 4.0])
    assert bootstrap_task_ci(values, n_bootstrap=100, seed=4) == bootstrap_task_ci(values, n_bootstrap=100, seed=4)


def test_effects_compare_gated_and_consensus_at_task_level():
    frame = _synthetic_frame()
    effects = build_effects(frame, n_bootstrap=100, seed=9)
    selected = effects[
        (effects["candidate"] == GATED)
        & (effects["reference"] == REFERENCE)
        & (effects["scenario_group"] == "label_shift")
        & (effects["metric"] == "brier")
        & np.isclose(effects["target_prior"].astype(float), 0.20)
    ]
    assert not selected.empty
    assert np.all(selected["mean_delta"] < 0)


def test_behavior_and_threshold_sensitivity_report_gate_rates():
    frame = _synthetic_frame()
    behavior = build_behavior(frame)
    assert set(behavior["scenario_family"]) == {"pure_label_shift", "mixed_shift"}
    sensitivity = threshold_sensitivity(frame)
    assert set(sensitivity["threshold"]) == {0.55, 0.60, 0.65}


def test_gate_decision_is_not_go_without_complete_input():
    frame = _synthetic_frame()
    effects = build_effects(frame, n_bootstrap=50)
    behavior = build_behavior(frame)
    decision = gate_decision(effects, behavior, complete=False)
    assert decision["decision"] == "INCONCLUSIVE"


def test_v51_gate_decision_uses_shrinkage_criteria():
    frame = _synthetic_frame()
    effects = build_effects(frame, n_bootstrap=50)
    behavior = build_behavior(frame)
    decision = gate_decision(effects, behavior, complete=True)
    assert decision["decision"] == "GO"
    assert decision["criteria"]["utility_pass"]
    assert decision["criteria"]["mixed_safety_pass"]
    assert decision["criteria"]["shrinkage_vs_binary_pass"]
