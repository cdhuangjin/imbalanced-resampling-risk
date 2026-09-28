from pathlib import Path
import sys

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from factor_experiment import (  # noqa: E402
    fit_factor_model,
    package_names_for_learners,
    parse_learner_spec,
)
from analyze_heldout_xgb_gate import paired_bootstrap_ci, summarize_gate  # noqa: E402


def test_xgb_gate_spec_is_explicit_and_reproducible():
    family, params = parse_learner_spec("xgb_gate")
    assert family == "xgb"
    assert params == {"n_estimators": 1000, "learning_rate": 0.1}
    assert parse_learner_spec("xgb_gate") == (family, params)
    assert "xgboost" in package_names_for_learners(("xgb_gate",))


def test_xgb_factor_fit_returns_finite_probabilities():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(48, 4))
    y = np.array([0, 1] * 24)
    model = fit_factor_model(
        "xgb",
        {"n_estimators": 8, "learning_rate": 0.1},
        x[:32],
        y[:32],
        x[32:40],
        y[32:40],
        0,
    )
    p = np.asarray(model.predict_proba(x[40:]))[:, 1]
    assert p.shape == (8,)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_task_level_paired_bootstrap_is_deterministic():
    differences = np.array([-0.2, 0.0, 0.1, 0.3], dtype=float)
    first = paired_bootstrap_ci(differences, replicates=2000, seed=91)
    second = paired_bootstrap_ci(differences, replicates=2000, seed=91)
    assert first == second
    assert first["mean"] == np.mean(differences)
    assert first["low"] <= first["mean"] <= first["high"]


def test_gate_summary_can_be_reused_for_a_frozen_confirmation_task_set():
    effects = __import__("pandas").DataFrame(
        [
            {"method": method, "metric": metric, "ci_low": -0.01, "ci_high": 0.01,
             "max_task_effect": 0.0}
            for method in ("random_q05", "smote_q05_k5")
            for metric in ("auc", "loss_minority", "loss_majority", "brier", "ap", "logloss", "f1_fixed", "f1_tuned")
        ]
    )
    rows = []
    for dataset in ("a", "b"):
        for method in ("none", "random_q05", "smote_q05_k5"):
            for seed in (0, 1):
                for metric in ("auc", "loss_minority", "loss_majority", "brier", "ap", "logloss", "f1_fixed", "f1_tuned"):
                    rows.append({
                        "run_id": f"{dataset}-{method}-{seed}", "dataset": dataset,
                        "learner": "xgb_gate", "method": method, "seed": seed,
                        "metric": metric, "value": 0.0, "config_hash": "c",
                        "data_hash": "d", "split_hash": "s",
                    })
    frame = __import__("pandas").DataFrame(rows)
    summary = summarize_gate(frame, effects, tasks=("a", "b"), seeds=(0, 1))
    assert summary["expected_fits"] == 12
    assert summary["observed_fits"] == 12
    assert summary["integrity_pass"] is True
