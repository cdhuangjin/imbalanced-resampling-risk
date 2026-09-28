"""Unlabeled residual-shift diagnostic and conservative policy gate."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def _features(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional array with at least 3 rows")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _weights(values: np.ndarray, size: int, *, name: str) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != size:
        raise ValueError(f"{name} must have one value per source row")
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"{name} must be finite, non-negative, and non-zero")
    return values


def _rejection(reason: str) -> dict:
    return {
        "valid": False,
        "residual_auc": None,
        "accepted": False,
        "reason": str(reason),
    }


def cross_fitted_domain_auc(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_weights: np.ndarray | None = None,
    *,
    seed: int = 0,
    n_splits: int = 3,
) -> dict:
    """Measure residual source/target separability without target labels.

    Domain labels are internal only: source is 0 and target is 1. Source rows
    may be importance-weighted to the estimated target prior. The returned AUC
    is oriented so that 0.5 means indistinguishable and larger values mean
    stronger residual shift.
    """
    try:
        source = _features(source_features, name="source_features")
        target = _features(target_features, name="target_features")
        if source.shape[1] != target.shape[1]:
            return _rejection("feature_dimension_mismatch")
        if n_splits < 2 or min(source.shape[0], target.shape[0]) < n_splits:
            return _rejection("insufficient_domain_support")
        weights = _weights(source_weights, source.shape[0], name="source_weights")
    except ValueError as exc:
        return _rejection(str(exc))

    features = np.vstack([source, target])
    labels = np.concatenate(
        [np.zeros(source.shape[0], dtype=int), np.ones(target.shape[0], dtype=int)]
    )
    sample_weights = np.concatenate([weights, np.ones(target.shape[0], dtype=float)])
    if np.unique(features, axis=0).shape[0] < 2:
        return _rejection("constant_domain_features")

    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    out_of_fold = np.full(labels.size, np.nan, dtype=float)
    try:
        for train_index, test_index in splitter.split(features, labels):
            classifier = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=1.0,
                            solver="liblinear",
                            max_iter=1000,
                            random_state=int(seed),
                        ),
                    ),
                ]
            )
            classifier.fit(
                features[train_index],
                labels[train_index],
                logistic__sample_weight=sample_weights[train_index],
            )
            out_of_fold[test_index] = classifier.predict_proba(
                features[test_index]
            )[:, 1]
        auc = float(
            roc_auc_score(labels, out_of_fold, sample_weight=sample_weights)
        )
    except (ValueError, FloatingPointError) as exc:
        return _rejection(f"domain_fit_failed:{exc}")

    if not np.isfinite(auc):
        return _rejection("nonfinite_domain_auc")
    residual_auc = float(max(auc, 1.0 - auc))
    return {
        "valid": True,
        "residual_auc": residual_auc,
        "accepted": bool(residual_auc <= 0.60),
        "reason": "accepted" if residual_auc <= 0.60 else "residual_shift_detected",
    }


def gate_consensus_probability(
    source_probability: np.ndarray,
    consensus_probability: np.ndarray,
    residual_result: dict,
    *,
    threshold: float = 0.60,
) -> dict:
    """Choose consensus correction or source fallback from diagnostics only."""
    source_probability = np.asarray(source_probability, dtype=float)
    consensus_probability = np.asarray(consensus_probability, dtype=float)
    if (
        source_probability.ndim != 1
        or consensus_probability.ndim != 1
        or source_probability.size != consensus_probability.size
        or source_probability.size == 0
        or not np.isfinite(source_probability).all()
        or not np.isfinite(consensus_probability).all()
    ):
        raise ValueError("probabilities must be finite one-dimensional arrays of equal size")
    threshold = float(threshold)
    if not 0.5 <= threshold < 1.0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and in [0.5, 1)")
    valid = bool(residual_result.get("valid", False))
    residual_auc = residual_result.get("residual_auc")
    accepted = valid and residual_auc is not None and float(residual_auc) <= threshold
    if accepted:
        probability = np.clip(consensus_probability, EPS, 1.0 - EPS)
        return {
            "probability": probability,
            "used_fallback": False,
            "accepted": True,
            "reason": "accepted",
        }
    probability = np.clip(source_probability, EPS, 1.0 - EPS)
    return {
        "probability": probability,
        "used_fallback": True,
        "accepted": False,
        "reason": str(residual_result.get("reason", "invalid_diagnostic")),
    }


def shrink_consensus_probability(
    source_probability: np.ndarray,
    consensus_probability: np.ndarray,
    residual_result: dict,
    *,
    lower: float = 0.55,
    upper: float = 0.75,
) -> dict:
    """Shrink consensus correction continuously as residual shift increases."""
    source_probability = np.asarray(source_probability, dtype=float)
    consensus_probability = np.asarray(consensus_probability, dtype=float)
    if (
        source_probability.ndim != 1
        or consensus_probability.ndim != 1
        or source_probability.size != consensus_probability.size
        or source_probability.size == 0
        or not np.isfinite(source_probability).all()
        or not np.isfinite(consensus_probability).all()
    ):
        raise ValueError("probabilities must be finite one-dimensional arrays of equal size")
    lower = float(lower)
    upper = float(upper)
    if not (0.5 <= lower < upper < 1.0) or not np.isfinite([lower, upper]).all():
        raise ValueError("lower and upper must satisfy 0.5 <= lower < upper < 1")
    residual_auc = residual_result.get("residual_auc")
    valid = bool(residual_result.get("valid", False)) and residual_auc is not None
    if valid:
        weight = float(np.clip((upper - float(residual_auc)) / (upper - lower), 0.0, 1.0))
    else:
        weight = 0.0
    probability = np.clip(
        source_probability + weight * (consensus_probability - source_probability),
        EPS,
        1.0 - EPS,
    )
    return {
        "probability": probability,
        "adaptation_weight": weight,
        "used_fallback": bool(weight <= 0.0),
        "reason": "weighted_consensus" if weight > 0.0 else str(residual_result.get("reason", "invalid_diagnostic")),
    }


def uncertainty_aware_shrinkage(
    source_probability: np.ndarray,
    consensus_probability: np.ndarray,
    residual_result: dict,
    stability_result: dict,
    *,
    lower: float = 0.55,
    upper: float = 0.75,
) -> dict:
    """Combine residual-shift shrinkage with label-free prior stability.

    The residual diagnostic controls whether the target domain is plausibly
    compatible with label-shift correction.  The stability diagnostic controls
    whether the estimated target prior is reproducible from the unlabeled
    target context.  Multiplication makes either failure sufficient for a
    source-calibration fallback.
    """
    base = shrink_consensus_probability(
        source_probability,
        consensus_probability,
        residual_result,
        lower=lower,
        upper=upper,
    )
    raw_stability = stability_result.get("stability_weight")
    try:
        stability_weight = float(raw_stability)
    except (TypeError, ValueError):
        stability_weight = 0.0
    if not np.isfinite(stability_weight):
        stability_weight = 0.0
    stability_weight = float(np.clip(stability_weight, 0.0, 1.0))
    adaptation_weight = float(base["adaptation_weight"] * stability_weight)
    source_probability = np.asarray(source_probability, dtype=float)
    consensus_probability = np.asarray(consensus_probability, dtype=float)
    probability = np.clip(
        source_probability
        + adaptation_weight * (consensus_probability - source_probability),
        EPS,
        1.0 - EPS,
    )
    if adaptation_weight <= 0.0:
        if base["adaptation_weight"] <= 0.0:
            reason = str(base["reason"])
        else:
            reason = "unstable_target_prior"
    else:
        reason = "residual_and_stability_weighted_consensus"
    return {
        "probability": probability,
        "residual_adaptation_weight": float(base["adaptation_weight"]),
        "stability_weight": stability_weight,
        "adaptation_weight": adaptation_weight,
        "used_fallback": bool(adaptation_weight <= 0.0),
        "reason": reason,
    }


def regime_adaptive_shrinkage(
    v51_probability: np.ndarray,
    rsc_probability: np.ndarray,
    residual_result: dict,
    *,
    threshold: float = 0.60,
) -> dict:
    """Use V5.1 in the accepted regime and RSC beyond the frozen threshold."""
    v51_probability = np.asarray(v51_probability, dtype=float)
    rsc_probability = np.asarray(rsc_probability, dtype=float)
    if (
        v51_probability.ndim != 1
        or rsc_probability.ndim != 1
        or v51_probability.size == 0
        or v51_probability.size != rsc_probability.size
        or not np.isfinite(v51_probability).all()
        or not np.isfinite(rsc_probability).all()
    ):
        raise ValueError("probabilities must be finite one-dimensional arrays of equal size")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be finite and in [0.5, 1)")
    residual_auc = residual_result.get("residual_auc")
    use_rsc = bool(
        residual_result.get("valid", False)
        and residual_auc is not None
        and np.isfinite(float(residual_auc))
        and float(residual_auc) > threshold
    )
    probability = np.clip(
        rsc_probability if use_rsc else v51_probability,
        EPS,
        1.0 - EPS,
    )
    return {
        "probability": probability,
        "selected_policy": "rsc" if use_rsc else "v51_shrinkage",
        "threshold": threshold,
        "reason": "residual_shift_above_threshold" if use_rsc else "accepted_or_invalid_detector",
    }
