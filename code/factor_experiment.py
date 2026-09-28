"""Capacity and sampling-factor experiment for the Elor follow-up."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from xgboost import XGBClassifier

from full_experiment import (
    METRICS,
    RAW_COLUMNS,
    _probability,
    _fit_catboost,
    _fit_decision_tree,
    aggregate_results,
    apply_sampling,
    build_data_manifest,
    build_run_specs,
    compute_metrics,
    sha_bytes,
    split_indices,
    stable_hash,
    tune_threshold,
    validate_raw_record,
)


LEARNER_PARAMS = {
    "dt_leaf005": ("dt", {"min_samples_leaf_factor": 0.005}),
    "dt_leaf01": ("dt", {"min_samples_leaf_factor": 0.01}),
    "dt_leaf02": ("dt", {"min_samples_leaf_factor": 0.02}),
    "dt_leaf04": ("dt", {"min_samples_leaf_factor": 0.04}),
    "cat_default": ("cat", {}),
    "cat_iter500": ("cat", {"iterations": 500}),
    "xgb_gate": ("xgb", {"n_estimators": 1000, "learning_rate": 0.1}),
}

METHOD_PARAMS = {
    "none": ("none", {}),
    "random_q02": ("random", {"ratio": 0.2}),
    "random_q05": ("random", {"ratio": 0.5}),
    "smote_q02_k3": ("smote", {"ratio": 0.2, "k": 3}),
    "smote_q02_k5": ("smote", {"ratio": 0.2, "k": 5}),
    "smote_q05_k3": ("smote", {"ratio": 0.5, "k": 3}),
    "smote_q05_k5": ("smote", {"ratio": 0.5, "k": 5}),
}


def parse_learner_spec(name: str):
    try:
        family, params = LEARNER_PARAMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown learner specification: {name}") from exc
    return family, dict(params)


def parse_method_spec(name: str):
    try:
        method, params = METHOD_PARAMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown method specification: {name}") from exc
    return method, dict(params)


def package_names_for_learners(learners):
    """Return the runtime packages required by the requested learner families."""
    names = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost"]
    if any(LEARNER_PARAMS[name][0] == "xgb" for name in learners):
        names.append("xgboost")
    return tuple(names)


def build_factor_specs(*, datasets, learners, methods, seeds):
    return build_run_specs(
        datasets=datasets,
        learners=learners,
        methods=methods,
        seeds=seeds,
        scenarios=("iid",),
        severities=(0,),
    )


def fit_factor_model(
    family,
    params,
    x_train,
    y_train,
    x_validation,
    y_validation,
    seed,
    sample_weight=None,
):
    """Fit a factor configuration without silently replacing its parameters."""
    if family == "dt":
        return _fit_decision_tree(
            x_train,
            y_train,
            x_validation,
            y_validation,
            seed,
            dict(params),
        )
    if family == "cat":
        return _fit_catboost(
            x_train,
            y_train,
            x_validation,
            y_validation,
            seed,
            dict(params),
        )
    if family == "xgb":
        # XGBoost 3.x moved early_stopping_rounds to the estimator constructor.
        # Keep this held-out learner path explicit and single-threaded so that
        # the protocol is reproducible and does not silently use test labels.
        model = XGBClassifier(
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
            early_stopping_rounds=10,
            tree_method="hist",
            **dict(params),
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        return model
    raise ValueError(f"unsupported factor learner family: {family}")


def run_factor_experiment(
    output: Path,
    *,
    datasets,
    learners,
    methods,
    seeds,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    specs = build_factor_specs(
        datasets=datasets, learners=learners, methods=methods, seeds=seeds
    )
    protocol = {
        "datasets": list(datasets),
        "learners": list(learners),
        "learner_params": {name: LEARNER_PARAMS[name][1] for name in learners},
        "methods": list(methods),
        "method_params": {name: METHOD_PARAMS[name][1] for name in methods},
        "seeds": list(seeds),
        "scenarios": ["iid"],
        "severities": [0],
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
            for name in package_names_for_learners(learners)
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
        data_hash = sha_bytes(x, y)
        train, validation, test = split_indices(y, spec["seed"])
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
        test_probability = _probability(model, x[test])
        threshold = tune_threshold(y[validation], validation_probability)
        values = compute_metrics(y[test], test_probability, threshold)
        runtime = time.perf_counter() - started
        config = {
            "learner_spec": spec["learner"],
            "learner_family": family,
            "learner_params": learner_params,
            "method_spec": spec["method"],
            "method": method,
            "method_params": method_params,
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
