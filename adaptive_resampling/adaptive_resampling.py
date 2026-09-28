"""Support-gated synthesis and validation-only risk-constrained selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


EPS = 1e-15


def _validate_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("x must be a 2D array with one label per row")
    if x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("x must contain at least two finite rows")
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("y must contain both binary classes")
    return x, y


def _validate_probability(probability, size):
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or probability.size != size:
        raise ValueError("probability must be a one-dimensional vector")
    if not np.isfinite(probability).all():
        raise ValueError("probability must be finite")
    return np.clip(probability, EPS, 1.0 - EPS)


def _validate_priors(target_priors: Sequence[float]) -> tuple[float, ...]:
    priors = tuple(float(value) for value in target_priors)
    if not priors or not all(np.isfinite(value) and 0.0 < value < 1.0 for value in priors):
        raise ValueError("target_priors must contain finite values strictly between zero and one")
    return priors


def minority_support_scores(x, y, k: int = 5) -> np.ndarray:
    """Return the minority fraction among each row's nearest non-self neighbors."""
    x, y = _validate_xy(x, y)
    if int(k) < 1:
        raise ValueError("k must be positive")
    neighbor_count = min(int(k) + 1, x.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(x)
    indices = neighbors.kneighbors(return_distance=False)
    scores = np.empty(x.shape[0], dtype=float)
    for row_index, row_neighbors in enumerate(indices):
        selected = [index for index in row_neighbors if int(index) != row_index]
        selected = selected[: int(k)]
        if not selected:
            scores[row_index] = 0.0
        else:
            scores[row_index] = float(np.mean(y[np.asarray(selected, dtype=int)] == 1))
    return scores


def support_gated_smote(
    x,
    y,
    *,
    ratio: float = 0.5,
    k: int = 5,
    min_support: float = 0.5,
    seed: int = 0,
):
    """Generate minority points only between locally supported minority seeds.

    The function returns the original rows first, followed by synthetic rows and
    a diagnostic dictionary. Validation and test data are intentionally not part
    of this interface.
    """
    x, y = _validate_xy(x, y)
    ratio = float(ratio)
    min_support = float(min_support)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be positive and finite")
    if not np.isfinite(min_support) or not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must be between zero and one")
    if int(k) < 1:
        raise ValueError("k must be positive")

    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    scores = minority_support_scores(x, y, k=int(k))
    eligible = positive[scores[positive] >= min_support]
    target_count = int(np.ceil(ratio * negative.size))
    generated_count = max(0, target_count - positive.size)
    diagnostics = {
        "ratio": ratio,
        "k": int(k),
        "min_support": min_support,
        "minority_count": int(positive.size),
        "majority_count": int(negative.size),
        "eligible_seed_count": int(eligible.size),
        "generated_count": int(generated_count),
    }
    if generated_count == 0 or eligible.size < 2:
        return x.copy(), y.copy(), diagnostics

    rng = np.random.default_rng(int(seed))
    eligible_x = x[eligible]
    neighbor_count = min(int(k) + 1, eligible_x.shape[0])
    neighbor_indices = NearestNeighbors(n_neighbors=neighbor_count).fit(eligible_x).kneighbors(
        eligible_x, return_distance=False
    )
    synthetic = []
    for _ in range(generated_count):
        anchor_position = int(rng.integers(eligible_x.shape[0]))
        candidates = [
            int(index)
            for index in neighbor_indices[anchor_position]
            if int(index) != anchor_position
        ]
        if not candidates:
            raise RuntimeError("support-gated interpolation found no neighbor")
        neighbor_position = int(rng.choice(candidates[: int(k)]))
        interpolation = float(rng.random())
        synthetic.append(
            eligible_x[anchor_position]
            + interpolation * (eligible_x[neighbor_position] - eligible_x[anchor_position])
        )
    synthetic_x = np.asarray(synthetic, dtype=float)
    synthetic_y = np.ones(generated_count, dtype=int)
    return (
        np.vstack([x, synthetic_x]),
        np.concatenate([y, synthetic_y]),
        diagnostics,
    )


def validation_policy_metrics(
    y,
    probability,
    *,
    target_priors: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Calculate validation metrics at source and declared target priors."""
    y = np.asarray(y, dtype=int)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("y must contain both binary classes")
    probability = _validate_probability(probability, y.size)
    priors = _validate_priors(target_priors)
    source_prior = float(np.mean(y))
    positive = y == 1
    negative = ~positive
    losses = -(y * np.log(probability) + (1 - y) * np.log1p(-probability))
    conditional = {
        "minority_logloss": float(losses[positive].mean()),
        "majority_logloss": float(losses[negative].mean()),
    }
    result = {}
    for prior in priors:
        weights = np.where(
            positive,
            prior / source_prior,
            (1.0 - prior) / (1.0 - source_prior),
        )
        result[prior] = {
            **conditional,
            "brier": float(np.average((probability - y) ** 2, weights=weights)),
        }
    return result


def select_risk_constrained_candidate(
    candidate_metrics: Mapping[str, Mapping[float, Mapping[str, float]]],
    *,
    candidate_order: Sequence[str],
    majority_logloss_tolerance: float = 0.02,
    brier_tolerance: float = 0.005,
):
    """Select the best feasible candidate using validation-only summaries."""
    order = tuple(str(name) for name in candidate_order)
    if not order or order[0] != "none":
        raise ValueError("candidate_order must begin with the none baseline")
    if len(set(order)) != len(order) or set(order) != set(candidate_metrics):
        raise ValueError("candidate_order and candidate_metrics must contain the same names")
    majority_logloss_tolerance = float(majority_logloss_tolerance)
    brier_tolerance = float(brier_tolerance)
    if majority_logloss_tolerance < 0.0 or brier_tolerance < 0.0:
        raise ValueError("risk tolerances must be non-negative")
    priors = tuple(candidate_metrics["none"])
    if not priors:
        raise ValueError("candidate metrics must contain at least one target prior")
    for name in order:
        if tuple(candidate_metrics[name]) != priors:
            raise ValueError("all candidates must contain the same target priors")

    baseline = candidate_metrics["none"]
    feasibility = {}
    scores = {}
    diagnostics = {}
    for name in order:
        prior_diagnostics = {}
        feasible = True
        for prior in priors:
            current = candidate_metrics[name][prior]
            base = baseline[prior]
            majority_delta = float(current["majority_logloss"] - base["majority_logloss"])
            brier_delta = float(current["brier"] - base["brier"])
            prior_feasible = (
                majority_delta <= majority_logloss_tolerance
                and brier_delta <= brier_tolerance
            )
            feasible = feasible and prior_feasible
            prior_diagnostics[str(prior)] = {
                "majority_logloss_delta": majority_delta,
                "brier_delta": brier_delta,
                "feasible": bool(prior_feasible),
            }
        feasibility[name] = bool(feasible)
        diagnostics[name] = prior_diagnostics
        scores[name] = float(
            np.mean([candidate_metrics[name][prior]["minority_logloss"] for prior in priors])
        )

    feasible_names = [name for name in order if feasibility[name]]
    if not feasible_names:
        selected = "none"
    else:
        selected = min(feasible_names, key=lambda name: (scores[name], order.index(name)))
    return {
        "selected": selected,
        "feasibility": feasibility,
        "minority_score": scores,
        "constraint_diagnostics": diagnostics,
        "candidate_order": list(order),
        "majority_logloss_tolerance": majority_logloss_tolerance,
        "brier_tolerance": brier_tolerance,
    }
