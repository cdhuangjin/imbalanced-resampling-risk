from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from analyze_cross_learner_mechanism import paired_bootstrap_ci, summarize_mechanism  # noqa: E402


def test_bootstrap_is_deterministic():
    values = np.array([-0.2, 0.0, 0.1, 0.3])
    assert paired_bootstrap_ci(values, replicates=1000, seed=7) == paired_bootstrap_ci(values, replicates=1000, seed=7)


def test_mechanism_rule_requires_minority_signal_and_auc_overlap():
    rows = []
    for method in ("random_q05", "smote_q05_k5"):
        for metric in ("auc", "loss_minority", "loss_majority", "brier"):
            rows.append({"method": method, "metric": metric, "mean": -0.1 if metric == "loss_minority" else 0.0, "ci_low": -0.2 if metric == "loss_minority" else -0.1, "ci_high": -0.05 if metric == "loss_minority" else 0.1, "max_task_effect": 0.2})
    effects = pd.DataFrame(rows)
    summary = summarize_mechanism(effects)
    assert summary["decision"] == "SUPPORTS_CONDITIONAL_MECHANISM"
