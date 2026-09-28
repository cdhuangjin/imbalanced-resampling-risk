"""Frozen support-filtered SMOTE policy against matched XGBoost treatments."""

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
from sklearn.neighbors import NearestNeighbors

from cost_sensitive_xgb_experiment import (
    DECISION_COSTS,
    decision_utility,
    tune_cost_threshold,
)
from factor_experiment import fit_factor_model, parse_learner_spec
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


LEARNER = "xgb_gate"
METHOD_PARAMS = {
    "none": ("none", {}),
    "random_q05": ("random", {"ratio": 0.5}),
    "smote_q05_k5": ("smote", {"ratio": 0.5, "k": 5}),
    "support_smote_q05": ("support_smote", {"ratio": 0.5, "k": 5, "support_threshold": 0.5}),
}
METHODS = tuple(METHOD_PARAMS)
COST_METRIC = "decision_utility_c5"
ALL_METRICS = tuple(METRICS) + (COST_METRIC,)


def _minority_neighbour_support(x, y, minority_indices, k):
    n_neighbors = min(int(k) + 1, len(x))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    neighbor_indices = neighbors.kneighbors(x[minority_indices], return_distance=False)
    supports = []
    for global_index, row in zip(minority_indices, neighbor_indices):
        row = [index for index in row if int(index) != int(global_index)][: int(k)]
        supports.append(float(np.mean(y[row] == 1)) if row else 0.0)
    return np.asarray(supports, dtype=float)


def support_filtered_smote(
    x,
    y,
    *,
    seed: int,
    ratio: float = 0.5,
    k: int = 5,
    support_threshold: float = 0.5,
):
    """Generate SMOTE rows only from training minority seeds with local support."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    minority_indices = np.flatnonzero(y == 1)
    majority_indices = np.flatnonzero(y == 0)
    if len(minority_indices) < 2 or len(majority_indices) == 0:
        return x.copy(), y.copy(), {"safe_fraction": 0.0, "fallback": True, "n_synthetic": 0}
    desired_minority = int(np.ceil(float(ratio) * len(majority_indices)))
    n_synthetic = max(0, desired_minority - len(minority_indices))
    supports = _minority_neighbour_support(x, y, minority_indices, k)
    safe_positions = np.flatnonzero(supports >= float(support_threshold))
    fallback = len(safe_positions) < 2
    if fallback:
        safe_positions = np.arange(len(minority_indices))
    safe_indices = minority_indices[safe_positions]
    if n_synthetic == 0:
        return x.copy(), y.copy(), {
            "safe_fraction": float(len(safe_positions) / len(minority_indices)),
            "fallback": bool(fallback),
            "n_synthetic": 0,
        }
    safe_x = x[safe_indices]
    neighbor_count = min(int(k) + 1, len(safe_x))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(safe_x)
    neighbor_indices = neighbors.kneighbors(safe_x, return_distance=False)
    rng = np.random.default_rng(seed)
    synthetic = []
    for _ in range(n_synthetic):
        seed_position = int(rng.integers(0, len(safe_x)))
        candidates = [int(index) for index in neighbor_indices[seed_position] if int(index) != seed_position]
        if not candidates:
            candidates = [seed_position]
        neighbor_position = int(rng.choice(candidates))
        interpolation = float(rng.random())
        synthetic.append(safe_x[seed_position] + interpolation * (safe_x[neighbor_position] - safe_x[seed_position]))
    x_augmented = np.vstack([x, np.asarray(synthetic, dtype=float)])
    y_augmented = np.concatenate([y, np.ones(n_synthetic, dtype=int)])
    return x_augmented, y_augmented, {
        "safe_fraction": float(len(safe_positions) / len(minority_indices)),
        "fallback": bool(fallback),
        "n_synthetic": int(n_synthetic),
    }


def validate_raw_record(record):
    missing = [name for name in RAW_COLUMNS if name not in record]
    if missing:
        raise ValueError("missing raw-record fields: " + ", ".join(missing))
    if record["metric"] not in ALL_METRICS:
        raise ValueError(f"unknown metric: {record['metric']}")
    if not np.isfinite(float(record["value"])):
        raise ValueError("metric value is not finite")
    for field in ("config_hash", "data_hash", "split_hash"):
        if not str(record[field]):
            raise ValueError(f"empty provenance hash in {field}")
    return record


def run_support_smote_experiment(output: Path, *, datasets, seeds):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    diagnostic_path = output / "support_diagnostics.csv"
    specs = build_run_specs(datasets=datasets, learners=(LEARNER,), methods=METHODS, seeds=seeds, scenarios=("iid",), severities=(0,))
    protocol = {
        "datasets": list(datasets), "learner": LEARNER,
        "learner_params": {"n_estimators": 1000, "learning_rate": 0.1},
        "methods": list(METHODS), "method_params": {name: METHOD_PARAMS[name][1] for name in METHODS},
        "seeds": list(seeds), "scenarios": ["iid"], "severities": [0], "metrics": list(ALL_METRICS),
        "decision_costs": DECISION_COSTS,
        "support_definition": "minority seed kNN minority fraction >= 0.5; k=5; training split only",
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
    }
    manifest = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "protocol_hash": stable_hash(protocol), "expected_fit_count": len(specs),
        "expected_metric_rows": len(specs) * len(ALL_METRICS),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost", "xgboost")},
    }
    datasets_loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = build_data_manifest(datasets_loaded, datasets); manifest["data_manifest_hash"] = stable_hash(data_manifest)
    (output / "data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    diagnostics = pd.read_csv(diagnostic_path) if diagnostic_path.exists() else pd.DataFrame(columns=("run_id", "safe_fraction", "fallback", "n_synthetic"))
    completed = set(existing["run_id"].astype(str)) if not existing.empty else set(); rows = existing.to_dict(orient="records")
    diagnostic_rows = diagnostics.to_dict(orient="records")
    for spec in specs:
        if spec["run_id"] in completed: continue
        dataset = datasets_loaded[spec["dataset"]]; x = np.asarray(dataset["data"], dtype=float); y = (np.asarray(dataset["target"]) == 1).astype(int)
        data_hash = sha_bytes(x, y); train, validation, test = split_indices(y, spec["seed"])
        split_hash = stable_hash({"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()})
        method, method_params = METHOD_PARAMS[spec["method"]]; support_info = {"safe_fraction": None, "fallback": False, "n_synthetic": 0}
        if method == "support_smote":
            x_train, y_train, support_info = support_filtered_smote(x[train], y[train], seed=spec["seed"], **method_params)
        else:
            x_train, y_train = apply_sampling(x[train], y[train], method, spec["seed"], ratio=method_params.get("ratio", 0.5), k=method_params.get("k", 5))
        family, learner_params = parse_learner_spec(LEARNER); started = time.perf_counter()
        model = fit_factor_model(family, learner_params, x_train, y_train, x[validation], y[validation], spec["seed"])
        validation_probability = _probability(model, x[validation]); test_probability = _probability(model, x[test])
        f1_threshold = tune_threshold(y[validation], validation_probability); cost_threshold = tune_cost_threshold(y[validation], validation_probability, fn_cost=DECISION_COSTS["fn"], fp_cost=DECISION_COSTS["fp"])
        values = compute_metrics(y[test], test_probability, f1_threshold); values[COST_METRIC] = decision_utility(y[test], test_probability, cost_threshold, fn_cost=DECISION_COSTS["fn"], fp_cost=DECISION_COSTS["fp"])
        runtime = time.perf_counter() - started
        config_hash = stable_hash({"learner": LEARNER, "learner_params": learner_params, "method_spec": spec["method"], "method": method, "method_params": method_params, "support_info": support_info, "decision_costs": DECISION_COSTS, "threshold_sources": ["validation_only_f1", "validation_only_cost_grid"]})
        for metric, value in values.items():
            rows.append(validate_raw_record({**spec, "metric": metric, "value": value, "config_hash": config_hash, "data_hash": data_hash, "split_hash": split_hash, "runtime_seconds": runtime, "timestamp": datetime.now(timezone.utc).isoformat()}))
        diagnostic_rows.append({"run_id": spec["run_id"], **support_info})
        pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw_path, index=False); pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS); frame.to_csv(raw_path, index=False); pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    summary = frame.groupby(["dataset", "learner", "method", "scenario", "severity", "metric"], dropna=False)["value"].agg(mean="mean", std="std", n="count").reset_index(); summary["std"] = summary["std"].fillna(0.0); summary.to_csv(output / "summary.csv", index=False)
    state = {"status": "COMPLETE" if frame["run_id"].nunique() == len(specs) else "INCOMPLETE", "fit_count": int(frame["run_id"].nunique()), "expected_fit_count": len(specs), "raw_metric_rows": len(frame), "expected_metric_rows": len(specs) * len(ALL_METRICS), "summary_rows": len(summary), "protocol_hash": manifest["protocol_hash"], "data_manifest_hash": manifest["data_manifest_hash"]}
    (output / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"); return state


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--datasets", required=True); parser.add_argument("--seeds", default="0,1,2,3,4"); args = parser.parse_args(); datasets = tuple(x for x in args.datasets.split(",") if x); seeds = tuple(int(x) for x in args.seeds.split(",") if x); print(run_support_smote_experiment(args.output, datasets=datasets, seeds=seeds))


if __name__ == "__main__": main()
