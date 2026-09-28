"""Frozen Balanced Random Forest learner-family experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.ensemble import BalancedRandomForestClassifier

from full_experiment import (
    METRICS,
    RAW_COLUMNS,
    _probability,
    apply_sampling,
    build_data_manifest,
    build_run_specs,
    compute_metrics,
    sha_bytes,
    split_indices,
    stable_hash,
    tune_threshold,
)


LEARNER = "brf_gate"
METHOD_PARAMS = {
    "none": ("none", {}),
    "random_q05": ("random", {"ratio": 0.5}),
    "smote_q05_k5": ("smote", {"ratio": 0.5, "k": 5}),
}
METHODS = tuple(METHOD_PARAMS)
DECISION_COSTS = {"fn": 5.0, "fp": 1.0}
COST_METRIC = "decision_utility_c5"
ALL_METRICS = tuple(METRICS) + (COST_METRIC,)
BRF_PARAMS = {
    "n_estimators": 300,
    "replacement": True,
    "sampling_strategy": "all",
    "max_features": "sqrt",
}


def fit_brf(x_train, y_train, *, seed: int, n_estimators: int = 300):
    model = BalancedRandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=1,
        replacement=BRF_PARAMS["replacement"],
        sampling_strategy=BRF_PARAMS["sampling_strategy"],
        max_features=BRF_PARAMS["max_features"],
    )
    return model.fit(x_train, y_train)


def tune_cost_threshold(y_validation, probability, *, fn_cost=5.0, fp_cost=1.0):
    y_validation = np.asarray(y_validation, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if y_validation.shape != probability.shape:
        raise ValueError("validation labels and probabilities must have matching shape")
    candidates = np.arange(0.0, 1.0, 0.01)
    costs = []
    for threshold in candidates:
        prediction = probability > threshold
        fn = ((y_validation == 1) & ~prediction).sum()
        fp = ((y_validation == 0) & prediction).sum()
        costs.append(float(fn_cost * fn + fp_cost * fp))
    return float(candidates[int(np.argmin(costs))])


def decision_utility(y_test, probability, threshold, *, fn_cost=5.0, fp_cost=1.0):
    y_test = np.asarray(y_test, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability > float(threshold)
    fn = ((y_test == 1) & ~prediction).sum()
    fp = ((y_test == 0) & prediction).sum()
    return float(-(fn_cost * fn + fp_cost * fp) / len(y_test))


def build_brf_specs(*, datasets, seeds):
    return build_run_specs(
        datasets=datasets,
        learners=(LEARNER,),
        methods=METHODS,
        seeds=seeds,
        scenarios=("iid",),
        severities=(0,),
    )


def validate_raw_record(record):
    missing = [name for name in RAW_COLUMNS if name not in record]
    if missing:
        raise ValueError("missing raw-record fields: " + ", ".join(missing))
    if record["metric"] not in ALL_METRICS:
        raise ValueError(f"unknown metric: {record['metric']}")
    for field in ("config_hash", "data_hash", "split_hash"):
        if not str(record[field]):
            raise ValueError(f"empty provenance hash in {field}")
    if not np.isfinite(float(record["value"])):
        raise ValueError("metric value is not finite")
    return record


def aggregate_brf_results(raw_csv: Path, summary_csv: Path):
    frame = pd.read_csv(raw_csv)
    for record in frame.to_dict(orient="records"):
        validate_raw_record(record)
    summary = (
        frame.groupby(
            ["dataset", "learner", "method", "scenario", "severity", "metric"],
            dropna=False,
        )["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    return summary


def run_brf_experiment(output: Path, *, datasets, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    specs = build_brf_specs(datasets=datasets, seeds=seeds)
    protocol = {
        "datasets": list(datasets),
        "learner": LEARNER,
        "learner_params": BRF_PARAMS,
        "methods": list(METHODS),
        "method_params": {name: METHOD_PARAMS[name][1] for name in METHODS},
        "seeds": list(seeds),
        "scenarios": ["iid"],
        "severities": [0],
        "metrics": list(ALL_METRICS),
        "decision_costs": DECISION_COSTS,
        "utility_threshold": "validation_cost_grid_0.01_lowest_tie",
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
    }
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": stable_hash(protocol),
        "expected_fit_count": len(specs),
        "expected_metric_rows": len(specs) * len(ALL_METRICS),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost")
        },
    }
    datasets_loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = build_data_manifest(datasets_loaded, datasets)
    manifest["data_manifest_hash"] = stable_hash(data_manifest)
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
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
        split_hash = stable_hash({"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()})
        method, method_params = METHOD_PARAMS[spec["method"]]
        x_train, y_train = apply_sampling(
            x[train], y[train], method, spec["seed"],
            ratio=method_params.get("ratio", 0.5), k=method_params.get("k", 5)
        )
        started = time.perf_counter()
        model = fit_brf(x_train, y_train, seed=spec["seed"], n_estimators=BRF_PARAMS["n_estimators"])
        validation_probability = _probability(model, x[validation])
        test_probability = _probability(model, x[test])
        f1_threshold = tune_threshold(y[validation], validation_probability)
        cost_threshold = tune_cost_threshold(
            y[validation],
            validation_probability,
            fn_cost=DECISION_COSTS["fn"],
            fp_cost=DECISION_COSTS["fp"],
        )
        values = compute_metrics(y[test], test_probability, f1_threshold)
        values[COST_METRIC] = decision_utility(
            y[test],
            test_probability,
            cost_threshold,
            fn_cost=DECISION_COSTS["fn"],
            fp_cost=DECISION_COSTS["fp"],
        )
        runtime = time.perf_counter() - started
        config_hash = stable_hash({
            "learner": LEARNER,
            "learner_params": BRF_PARAMS,
            "method_spec": spec["method"],
            "method": method,
            "method_params": method_params,
            "f1_threshold_source": "validation_only",
            "utility_threshold_source": "validation_only_cost_grid",
            "decision_costs": DECISION_COSTS,
        })
        for metric, value in values.items():
            rows.append(validate_raw_record({
                **spec,
                "metric": metric,
                "value": value,
                "config_hash": config_hash,
                "data_hash": data_hash,
                "split_hash": split_hash,
                "runtime_seconds": runtime,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))
        pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw_path, index=False)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame.to_csv(raw_path, index=False)
    summary = aggregate_brf_results(raw_path, output / "summary.csv")
    state = {
        "status": "COMPLETE" if frame["run_id"].nunique() == len(specs) else "INCOMPLETE",
        "fit_count": int(frame["run_id"].nunique()),
        "expected_fit_count": len(specs),
        "raw_metric_rows": len(frame),
        "expected_metric_rows": len(specs) * len(ALL_METRICS),
        "summary_rows": len(summary),
        "protocol_hash": manifest["protocol_hash"],
        "data_manifest_hash": manifest["data_manifest_hash"],
    }
    (output / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    datasets = tuple(x for x in args.datasets.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    print(run_brf_experiment(args.output, datasets=datasets, seeds=seeds))


if __name__ == "__main__":
    main()
