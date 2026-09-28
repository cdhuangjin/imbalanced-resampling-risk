from __future__ import annotations

import pandas as pd

from analyze_residual_stability import RSC, SHRINKAGE, build_effects, gate_decision


def _frame():
    rows = []
    for dataset in ("a", "b", "c"):
        for seed in (0, 1):
            for scenario, prior in (
                ("label_shift", 0.20),
                ("label_shift", 0.50),
                ("covariate_shift", 0.20),
                ("joint_shift", 0.50),
            ):
                for postprocessor, delta in (
                    ("platt_source", 0.0),
                    ("platt_target_shrinkage", -0.02),
                    ("platt_target_rsc", -0.01),
                ):
                    for metric in ("brier", "logloss"):
                        base = 0.2 if metric == "brier" else 1.0
                        value = base + delta
                        if scenario in ("covariate_shift", "joint_shift"):
                            value = base
                        rows.append(
                            {
                                "fit_id": f"{dataset}__dt__s{seed}__none",
                                "dataset": dataset,
                                "learner": "dt_leaf01",
                                "method": "none",
                                "seed": seed,
                                "scenario": scenario,
                                "severity": 1,
                                "target_prior": prior,
                                "covariate_severity": 1,
                                "postprocessor": postprocessor,
                                "metric": metric,
                                "value": value,
                                "residual_auc": 0.55 if scenario == "label_shift" else 0.80,
                                "residual_valid": True,
                                "rsc_stability_weight": 0.7,
                            }
                        )
    return pd.DataFrame(rows)


def test_effects_include_rsc_and_v51_comparison():
    effects = build_effects(_frame(), n_bootstrap=100, seed=3)
    assert ((effects["candidate"] == RSC) & (effects["reference"] == "platt_source")).any()
    assert ((effects["candidate"] == RSC) & (effects["reference"] == SHRINKAGE)).any()


def test_gate_decision_is_inconclusive_for_incomplete_input():
    frame = _frame()
    effects = build_effects(frame, n_bootstrap=50)
    decision = gate_decision(effects, complete=False)
    assert decision["decision"] == "INCONCLUSIVE"
