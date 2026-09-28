from pathlib import Path
import sys

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from support_smote_xgb_experiment import (  # noqa: E402
    METHODS,
    support_filtered_smote,
)


def test_support_filtered_smote_is_deterministic_and_preserves_original_rows():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(30, 3))
    y = np.array([0] * 24 + [1] * 6)
    first_x, first_y, first_info = support_filtered_smote(x, y, seed=3)
    second_x, second_y, second_info = support_filtered_smote(x, y, seed=3)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert first_info == second_info
    assert first_x.shape[0] >= x.shape[0]
    assert np.array_equal(first_x[: len(x)], x)
    assert np.array_equal(first_y[: len(y)], y)


def test_support_filtered_smote_protocol_is_explicit():
    assert METHODS == ("none", "random_q05", "smote_q05_k5", "support_smote_q05")
