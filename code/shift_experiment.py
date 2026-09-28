"""Controlled label-shift and covariate-shift stress tests for the Elor follow-up."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from factor_experiment import (
    LEARNER_PARAMS,
    METHOD_PARAMS,
    fit_factor_model,
    parse_learner_spec,
    parse_method_spec,
)
from full_experiment import (
    METRICS,
    RAW_COLUMNS,
    aggregate_results,
    apply_sampling,
    build_data_manifest,
    compute_metrics,
    sha_bytes,
    split_indices,
    stable_hash,
    tune_threshold,
    validate_raw_record,
    _probability,
)


LABEL_SHIFT_PRIORS = {0: 0.05, 1: 0.20, 2: 0.50}
COVARIATE_SHIFT_SEVERITIES = (0, 1, 2)


def target_prior(severity: int) -> float:
    try:
        return LABEL_SHIFT_PRIORS[int(severity)]
    except KeyError as exc:
        raise ValueError(f"unknown label-shift severity: {severity}") from exc


def apply_covariate_shift(x, mean, scale, severity: int):
    """Shift the first non-constant feature by a fixed number of train SDs."""
    x = np.asarray(x, dtype=float).copy()
    scale = np.asarray(scale, dtype=float)
    candidates = np.flatnonzero(np.isfinite(scale) & (np.abs(scale) > 1e-12))
    if len(candidates) == 0:
        raise ValueError("no non-constant feature available for covariate shift")
    feature_index = int(candidates[0])
    x[:, feature_index] += int(severity) * scale[feature_index]
    return x, feature_index


def build_shift_specs(*, datasets, learners, methods, seeds):
    specs = []
    for dataset in datasets:
        for learner in learners:
            for seed in seeds:
                for method in methods:
                    scenarios = [("iid", 0)]
                    scenarios.extend(("label_shift", severity) for severity in LABEL_SHIFT_PRIORS)
                    scenarios.extend(
                        ("covariate_shift", severity)
                        for severity in COVARIATE_SHIFT_SEVERITIES
                    )
                    for scenario, severity in scenarios:
                        specs.append(
                            {
                                "run_id": f"{dataset}__{learner}__s{seed}__{method}__{scenario}__sev{severity}",
                                "dataset": str(dataset),
                                "learner": str(learner),
                                "method": str(method),
                                "seed": int(seed),
                                "scenario": scenario,
                                "severity": int(severity),
                            }
                        )
    return specs


def source_prior_weights(y, target_pi: float):
    y = np.asarray(y, dtype=int)
    source_pi = float(np.mean(y))
    if not 0.0 < source_pi < 1.0:
        raise ValueError("source test fold must contain both classes")
    weights = np.where(
        y == 1,
        target_pi / source_pi,
        (1.0 - target_pi) / (1.0 - source_pi),
    )
    return weights


def compute_shift_metrics(y, probability, threshold, sample_weight=None):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    weights = (
        np.ones_like(p, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    losses = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    prediction = (p > 0.5).astype(int)
    tuned_prediction = (p > threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, p, sample_weight=weights)),
        "ap": float(average_precision_score(y, p, sample_weight=weights)),
        "logloss": float(np.average(losses, weights=weights)),
        "loss_minority": float(np.average(losses[y == 1], weights=weights[y == 1])),
        "loss_majority": float(np.average(losses[y == 0], weights=weights[y == 0])),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
        "f1_fixed": float(
            f1_score(y, prediction, sample_weight=weights, zero_division=0)
        ),
        "f1_tuned": float(
            f1_score(y, tuned_prediction, sample_weight=weights, zero_division=0)
        ),
    }


def run_shift_experiment(output: Path, *, datasets, learners, methods, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    specs = build_shift_specs(
        datasets=datasets, learners=learners, methods=methods, seeds=seeds
    )
    protocol = {
        "datasets": list(datasets),
        "learners": list(learners),
        "learner_params": {name: LEARNER_PARAMS[name][1] for name in learners},
        "methods": list(methods),
        "method_params": {name: METHOD_PARAMS[name][1] for name in methods},
        "seeds": list(seeds),
        "scenarios": ["iid", "label_shift", "covariate_shift"],
        "label_shift_target_priors": LABEL_SHIFT_PRIORS,
        "covariate_shift_severities": list(COVARIATE_SHIFT_SEVERITIES),
        "covariate_shift_feature_rule": "first non-constant feature by train-fold standard deviation",
        "metrics": list(METRICS),
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
    }
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": stable_hash(protocol),
        "expected_fit_count": len(specs),
        "expected_metric_rows": len(specs) * len(METRICS),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "catboost",
            )
        },
    }
    datasets_loaded = fetch_datasets(
        filter_data=tuple(datasets), download_if_missing=False
    )
    data_manifest = build_data_manifest(datasets_loaded, datasets)
    manifest["data_manifest_hash"] = stable_hash(data_manifest)
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing = (
        pd.read_csv(raw_path)
        if raw_path.exists()
        else pd.DataFrame(columns=RAW_COLUMNS)
    )
    completed = set(existing["run_id"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict(orient="records")
    for spec in specs:
        if spec["run_id"] in completed:
            continue
        dataset = datasets_loaded[spec["dataset"]]
        x = np.asarray(dataset["data"], dtype=float)
        y = (np.asarray(dataset["target"]) == 1).astype(int)
        train, validation, test = split_indices(y, spec["seed"])
        data_hash = sha_bytes(x, y)
        split_hash = stable_hash(
            {
                "train": train.tolist(),
                "validation": validation.tolist(),
                "test": test.tolist(),
            }
        )
        method, method_params = parse_method_spec(spec["method"])
        x_train, y_train = apply_sampling(
            x[train],
            y[train],
            method,
            spec["seed"],
            ratio=method_params.get("ratio", 0.5),
            k=method_params.get("k", 5),
        )
        family, learner_params = parse_learner_spec(spec["learner"])
        started = time.perf_counter()
        model = fit_factor_model(
            family,
            learner_params,
            x_train,
            y_train,
            x[validation],
            y[validation],
            spec["seed"],
        )
        validation_probability = _probability(model, x[validation])
        threshold = tune_threshold(y[validation], validation_probability)
        x_eval = x[test]
        if spec["scenario"] == "covariate_shift":
            train_mean = np.mean(x[train], axis=0)
            train_scale = np.std(x[train], axis=0)
            x_eval, shift_feature = apply_covariate_shift(
                x_eval, train_mean, train_scale, spec["severity"]
            )
            sample_weight = np.ones(len(test), dtype=float)
        elif spec["scenario"] == "label_shift":
            shift_feature = None
            sample_weight = source_prior_weights(
                y[test], target_prior(spec["severity"])
            )
        else:
            shift_feature = None
            sample_weight = np.ones(len(test), dtype=float)
        test_probability = _probability(model, x_eval)
        values = compute_shift_metrics(
            y[test], test_probability, threshold, sample_weight=sample_weight
        )
        runtime = time.perf_counter() - started
        config = {
            "learner_spec": spec["learner"],
            "learner_family": family,
            "learner_params": learner_params,
            "method_spec": spec["method"],
            "method": method,
            "method_params": method_params,
            "scenario": spec["scenario"],
            "severity": spec["severity"],
            "target_prior": (
                target_prior(spec["severity"])
                if spec["scenario"] == "label_shift"
                else None
            ),
            "covariate_feature_index": shift_feature,
            "threshold_source": "validation_only",
        }
        config_hash = stable_hash(config)
        for metric, value in values.items():
            rows.append(
                validate_raw_record(
                    {
                        **spec,
                        "metric": metric,
                        "value": value,
                        "config_hash": config_hash,
                        "data_hash": data_hash,
                        "split_hash": split_hash,
                        "runtime_seconds": runtime,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
        pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw_path, index=False)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame.to_csv(raw_path, index=False)
    summary = aggregate_results(raw_path, output / "summary.csv")
    state = {
        "status": "COMPLETE" if frame["run_id"].nunique() == len(specs) else "INCOMPLETE",
        "fit_count": int(frame["run_id"].nunique()),
        "expected_fit_count": len(specs),
        "raw_metric_rows": len(frame),
        "expected_metric_rows": len(specs) * len(METRICS),
        "summary_rows": len(summary),
        "protocol_hash": manifest["protocol_hash"],
        "data_manifest_hash": manifest["data_manifest_hash"],
    }
    (output / "status.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state
