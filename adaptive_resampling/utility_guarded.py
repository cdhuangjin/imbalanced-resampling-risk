"""Validation-only 5:1 utility selector for a separate PaperForge Gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from confidence_guarded import MetricBundle, _paired_upper


def tune_cost_threshold(y, probability, *, fn_cost=5.0, fp_cost=1.0):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.arange(0.01, 1.0, 0.01)
    costs = []
    for threshold in candidates:
        prediction = probability > float(threshold)
        false_negative = int(np.sum((y == 1) & (prediction == 0)))
        false_positive = int(np.sum((y == 0) & (prediction == 1)))
        costs.append(float(fn_cost * false_negative + fp_cost * false_positive))
    return float(candidates[int(np.argmin(costs))])


def utility_rows(y, probability, threshold, *, fn_cost=5.0, fp_cost=1.0):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability > float(threshold)
    return -(
        float(fn_cost) * ((y == 1) & (prediction == 0)).astype(float)
        + float(fp_cost) * ((y == 0) & (prediction == 1)).astype(float)
    )


def _utility_ci(candidate, baseline, *, seed, n_boot):
    if not np.array_equal(candidate.y, baseline.y):
        raise ValueError("all candidates must use the same validation labels")
    candidate_threshold = tune_cost_threshold(candidate.y, candidate.probability)
    baseline_threshold = tune_cost_threshold(baseline.y, baseline.probability)
    candidate_values = utility_rows(candidate.y, candidate.probability, candidate_threshold)
    baseline_values = utility_rows(baseline.y, baseline.probability, baseline_threshold)
    delta = candidate_values - baseline_values
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, delta.size, size=(int(n_boot), delta.size))
    means = delta[draws].mean(axis=1)
    return {
        "threshold": float(candidate_threshold),
        "baseline_threshold": float(baseline_threshold),
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def utility_guarded_select(
    candidate_metrics: Mapping[str, Mapping[float, Mapping[str, float]]],
    *,
    candidate_order: Sequence[str],
    majority_logloss_tolerance: float = 0.02,
    brier_tolerance: float = 0.005,
    bootstrap_replicates: int = 400,
    bootstrap_seed: int = 20260917,
):
    """Select the highest validation utility among risk-safe candidates."""
    order = tuple(str(name) for name in candidate_order)
    if not order or order[0] != "none" or set(order) != set(candidate_metrics):
        raise ValueError("candidate_order must begin with none and match candidate_metrics")
    if int(bootstrap_replicates) < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    baseline = candidate_metrics["none"]
    if not isinstance(baseline, MetricBundle):
        raise ValueError("utility guarding requires raw validation metrics")
    priors = tuple(baseline)
    feasibility = {"none": True}
    scores = {"none": float(np.mean(utility_rows(baseline.y, baseline.probability, tune_cost_threshold(baseline.y, baseline.probability))))}
    diagnostics = {"none": {"utility": {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}, "feasible": True}}
    thresholds = {"none": tune_cost_threshold(baseline.y, baseline.probability)}
    for order_index, name in enumerate(order[1:], start=1):
        bundle = candidate_metrics[name]
        if not isinstance(bundle, MetricBundle):
            raise ValueError("all candidates must carry raw validation metrics")
        utility = _utility_ci(
            bundle,
            baseline,
            seed=int(bootstrap_seed) + 101 * order_index,
            n_boot=int(bootstrap_replicates),
        )
        risk_by_prior = {}
        feasible = utility["ci_low"] > 0.0
        for prior_index, prior in enumerate(priors):
            risk = _paired_upper(
                bundle,
                baseline,
                prior=float(prior),
                seed=int(bootstrap_seed) + 1009 * order_index + prior_index,
                n_boot=int(bootstrap_replicates),
            )
            risk_by_prior[str(prior)] = risk
            feasible = feasible and (
                risk["majority_upper"] <= float(majority_logloss_tolerance)
                and risk["brier_upper"] <= float(brier_tolerance)
            )
        feasibility[name] = bool(feasible)
        scores[name] = float(utility["mean_delta"])
        thresholds[name] = float(utility["threshold"])
        diagnostics[name] = {
            "utility": utility,
            "risk_by_prior": risk_by_prior,
            "feasible": bool(feasible),
        }
    feasible_names = [name for name in order if feasibility.get(name, False)]
    selected = max(feasible_names, key=lambda name: (scores[name], -order.index(name)))
    return {
        "selected": selected,
        "feasibility": feasibility,
        "utility_score": scores,
        "minority_score": scores,
        "utility_thresholds": thresholds,
        "constraint_diagnostics": diagnostics,
        "candidate_order": list(order),
        "majority_logloss_tolerance": float(majority_logloss_tolerance),
        "brier_tolerance": float(brier_tolerance),
        "validation_bootstrap_replicates": int(bootstrap_replicates),
        "validation_bootstrap_seed": int(bootstrap_seed),
        "selection_rule": "utility_first_risk_guard",
    }
