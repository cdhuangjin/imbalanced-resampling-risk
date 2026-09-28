"""Validation-row bootstrap confidence guard for adaptive resampling selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class MetricBundle(dict):
    """Dict-compatible validation metrics carrying hidden row-level inputs."""

    def __init__(self, metrics, *, y, probability):
        super().__init__(metrics)
        self.y = np.asarray(y, dtype=int).copy()
        self.probability = np.asarray(probability, dtype=float).copy()


def attach_raw_metrics(metrics, *, y, probability):
    return MetricBundle(metrics, y=y, probability=probability)


def _bootstrap_upper(values, *, seed, n_boot):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and one-dimensional")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.975))


def _row_components(y, probability, prior):
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    if y.ndim != 1 or probability.ndim != 1 or y.size != probability.size:
        raise ValueError("y and probability must be aligned vectors")
    positive = y == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("validation data must contain both classes")
    losses = -(y * np.log(probability) + (1 - y) * np.log1p(-probability))
    weights = np.where(positive, float(prior) / positive.mean(), (1 - float(prior)) / negative.mean())
    return {
        "minority": losses[positive],
        "majority": losses[negative],
        "brier": weights * (probability - y) ** 2,
        "brier_weights": weights,
    }


def _paired_upper(candidate, baseline, *, prior, seed, n_boot):
    current = _row_components(candidate.y, candidate.probability, prior)
    reference = _row_components(baseline.y, baseline.probability, prior)
    if not np.array_equal(candidate.y, baseline.y):
        raise ValueError("all candidates must use the same validation labels")
    rng = np.random.default_rng(int(seed))
    n = candidate.y.size
    draws = rng.integers(0, n, size=(int(n_boot), n))
    positive = candidate.y == 1
    negative = ~positive
    positive_positions = np.flatnonzero(positive)
    negative_positions = np.flatnonzero(negative)
    positive_local = np.full(n, -1, dtype=int)
    negative_local = np.full(n, -1, dtype=int)
    positive_local[positive_positions] = np.arange(positive_positions.size)
    negative_local[negative_positions] = np.arange(negative_positions.size)
    minority_delta = (
        current["minority"][positive_local[draws[:, positive]]]
        - reference["minority"][positive_local[draws[:, positive]]]
    ).mean(axis=1)
    majority_delta = (
        current["majority"][negative_local[draws[:, negative]]]
        - reference["majority"][negative_local[draws[:, negative]]]
    ).mean(axis=1)
    brier_delta = (
        current["brier"][draws].sum(axis=1) / current["brier_weights"][draws].sum(axis=1)
        - reference["brier"][draws].sum(axis=1) / reference["brier_weights"][draws].sum(axis=1)
    )
    return {
        "minority_upper": float(np.quantile(minority_delta, 0.975)),
        "majority_upper": float(np.quantile(majority_delta, 0.975)),
        "brier_upper": float(np.quantile(brier_delta, 0.975)),
    }


def confidence_guarded_select(
    candidate_metrics: Mapping[str, Mapping[float, Mapping[str, float]]],
    *,
    candidate_order: Sequence[str],
    majority_logloss_tolerance: float = 0.02,
    brier_tolerance: float = 0.005,
    minority_required_improvement: float = 0.0,
    bootstrap_replicates: int = 400,
    bootstrap_seed: int = 20260917,
):
    """Select a candidate only when validation-row CIs satisfy all risk guards.

    This is a policy selector, not a new sampler. Candidate models are still
    fitted exactly as in the existing adaptive runner; only the validation
    decision rule is replaced in the dedicated confidence-guarded branch.
    """
    order = tuple(str(name) for name in candidate_order)
    if not order or order[0] != "none" or set(order) != set(candidate_metrics):
        raise ValueError("candidate_order must begin with none and match candidate_metrics")
    if int(bootstrap_replicates) < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    baseline = candidate_metrics["none"]
    baseline_bundle = candidate_metrics["none"]
    if not isinstance(baseline_bundle, MetricBundle):
        raise ValueError("confidence guarding requires raw validation metrics")
    priors = tuple(baseline)
    feasibility = {}
    scores = {}
    diagnostics = {}
    for order_index, name in enumerate(order):
        bundle = candidate_metrics[name]
        if not isinstance(bundle, MetricBundle):
            raise ValueError("all candidates must carry raw validation metrics")
        feasible = True
        prior_diagnostics = {}
        for prior_index, prior in enumerate(priors):
            ci = _paired_upper(
                bundle,
                baseline_bundle,
                prior=float(prior),
                seed=int(bootstrap_seed) + 1009 * order_index + prior_index,
                n_boot=int(bootstrap_replicates),
            )
            prior_feasible = (
                ci["majority_upper"] <= float(majority_logloss_tolerance)
                and ci["brier_upper"] <= float(brier_tolerance)
                and ci["minority_upper"] <= -float(minority_required_improvement)
            )
            feasible = feasible and prior_feasible
            prior_diagnostics[str(prior)] = {
                **ci,
                "feasible": bool(prior_feasible),
                "minority_required_improvement": float(minority_required_improvement),
            }
        feasibility[name] = bool(feasible)
        diagnostics[name] = prior_diagnostics
        scores[name] = float(
            np.mean([candidate_metrics[name][prior]["minority_logloss"] for prior in priors])
        )
    feasible_names = [name for name in order if feasibility[name]]
    selected = min(feasible_names, key=lambda name: (scores[name], order.index(name))) if feasible_names else "none"
    return {
        "selected": selected,
        "feasibility": feasibility,
        "minority_score": scores,
        "constraint_diagnostics": diagnostics,
        "candidate_order": list(order),
        "majority_logloss_tolerance": float(majority_logloss_tolerance),
        "brier_tolerance": float(brier_tolerance),
        "minority_required_improvement": float(minority_required_improvement),
        "validation_bootstrap_replicates": int(bootstrap_replicates),
        "validation_bootstrap_seed": int(bootstrap_seed),
        "selection_rule": "row_bootstrap_upper_confidence_guard",
    }
