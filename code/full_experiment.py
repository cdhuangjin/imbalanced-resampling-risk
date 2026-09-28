"""Resumable, provenance-first runner for the Elor follow-up experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official_code" / "src"))
from classifiers import _fit_catboost, _fit_decision_tree  # noqa: E402


class ProvenanceError(RuntimeError):
    """Raised when an artifact cannot be safely included in an analysis."""


RAW_COLUMNS = (
    "run_id",
    "dataset",
    "learner",
    "method",
    "seed",
    "scenario",
    "severity",
    "metric",
    "value",
    "config_hash",
    "data_hash",
    "split_hash",
    "runtime_seconds",
    "timestamp",
)
METRICS = (
    "auc",
    "ap",
    "logloss",
    "loss_minority",
    "loss_majority",
    "brier",
    "f1_fixed",
    "f1_tuned",
)
REFERENCE_PACKAGE_VERSIONS = {"scikit-learn": "1.8.0"}


def build_run_specs(
    *,
    datasets,
    learners,
    methods,
    seeds,
    scenarios=("iid",),
    severities=(0,),
):
    """Return deterministic fit specifications in the supplied order."""
    specs = []
    for dataset in datasets:
        for learner in learners:
            for seed in seeds:
                for method in methods:
                    for scenario in scenarios:
                        for severity in severities:
                            specs.append(
                                {
                                    "run_id": f"{dataset}__{learner}__s{seed}__{method}__{scenario}__sev{severity}",
                                    "dataset": str(dataset),
                                    "learner": str(learner),
                                    "method": str(method),
                                    "seed": int(seed),
                                    "scenario": str(scenario),
                                    "severity": int(severity),
                                }
                            )
    return specs


def order_run_specs(specs, schedule_path: Path | None = None):
    """Apply a complete, auditable randomized run order when one is supplied."""
    if schedule_path is None:
        return list(specs)

    schedule = pd.read_csv(Path(schedule_path))
    required = {"run_order", "dataset", "learner", "seed", "method"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ProvenanceError("run-order schedule missing columns: " + ", ".join(missing))
    if schedule[list(required)].isna().any().any():
        raise ProvenanceError("run-order schedule contains empty required values")

    try:
        run_order = schedule["run_order"].astype(int).tolist()
        schedule_keys = [
            (str(row.dataset), str(row.learner), int(row.seed), str(row.method))
            for row in schedule.itertuples(index=False)
        ]
        spec_keys = [
            (str(spec["dataset"]), str(spec["learner"]), int(spec["seed"]), str(spec["method"]))
            for spec in specs
        ]
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProvenanceError("run-order schedule has invalid key values") from exc

    expected_order = list(range(1, len(schedule) + 1))
    if run_order != expected_order:
        raise ProvenanceError("run_order must be the contiguous sequence 1..N")
    if len(schedule_keys) != len(set(schedule_keys)):
        raise ProvenanceError("run-order schedule contains duplicate fit conditions")
    if len(spec_keys) != len(set(spec_keys)):
        raise ProvenanceError("requested run specification contains duplicate fit conditions")
    if set(schedule_keys) != set(spec_keys):
        missing_specs = sorted(set(spec_keys) - set(schedule_keys))
        extra_schedule = sorted(set(schedule_keys) - set(spec_keys))
        raise ProvenanceError(
            "run-order schedule does not match requested fits "
            f"(missing={missing_specs[:3]}, extra={extra_schedule[:3]})"
        )

    by_key = {key: spec for key, spec in zip(spec_keys, specs)}
    return [by_key[key] for key in schedule_keys]


def package_version_differences(packages: dict):
    """Report differences from the main-study runtime for interpretation."""
    return {
        name: {"expected": expected, "observed": packages.get(name)}
        for name, expected in REFERENCE_PACKAGE_VERSIONS.items()
        if packages.get(name) != expected
    }


def split_indices(y: np.ndarray, seed: int):
    """Create disjoint stratified train/validation/test indices."""
    indices = np.arange(len(y), dtype=int)
    train_validation, test = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=seed
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=y[train_validation],
        random_state=seed,
    )
    return train, validation, test


def validate_raw_record(record: dict):
    missing = [name for name in RAW_COLUMNS if name not in record]
    if missing:
        raise ProvenanceError("missing raw-record fields: " + ", ".join(missing))
    if not str(record["config_hash"]):
        raise ProvenanceError("config_hash is empty")
    if not str(record["data_hash"]):
        raise ProvenanceError("data_hash is empty")
    if not str(record["split_hash"]):
        raise ProvenanceError("split_hash is empty")
    if record["metric"] not in METRICS:
        raise ProvenanceError(f"unknown metric: {record['metric']}")
    if not np.isfinite(float(record["value"])):
        raise ProvenanceError("metric value is not finite")
    return record


def aggregate_results(raw_csv: Path, summary_csv: Path, expected_metrics=METRICS):
    """Validate raw rows and write a seed-level mean/std summary."""
    raw_csv = Path(raw_csv)
    summary_csv = Path(summary_csv)
    frame = pd.read_csv(raw_csv)
    missing = [name for name in RAW_COLUMNS if name not in frame.columns]
    if missing:
        raise ProvenanceError("missing raw-result columns: " + ", ".join(missing))
    for record in frame.to_dict(orient="records"):
        validate_raw_record(record)
    expected = set(expected_metrics)
    observed = set(frame["metric"].dropna().astype(str))
    if not expected.issubset(observed):
        raise ProvenanceError(
            "metric coverage incomplete: missing "
            + ", ".join(sorted(expected - observed))
        )
    group_columns = ["dataset", "learner", "method", "scenario", "severity", "metric"]
    summary = (
        frame.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_csv.with_name(summary_csv.name + ".tmp")
    summary.to_csv(temporary, index=False)
    temporary.replace(summary_csv)
    return summary


def sha_bytes(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def build_data_manifest(datasets_loaded, names):
    """Record immutable shape, label counts, and content hashes for each task."""
    manifest = {}
    for name in names:
        dataset = datasets_loaded[name]
        x = np.asarray(dataset["data"], dtype=float)
        y = np.asarray(dataset["target"])
        labels, counts = np.unique(y, return_counts=True)
        manifest[str(name)] = {
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "class_counts": {
                str(label): int(count) for label, count in zip(labels, counts)
            },
            "data_hash": sha_bytes(x, y),
        }
    return manifest


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_metrics(y: np.ndarray, probability: np.ndarray, threshold: float):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    losses = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    prediction = (p > 0.5).astype(int)
    tuned_prediction = (p > threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "logloss": float(losses.mean()),
        "loss_minority": float(losses[y == 1].mean()),
        "loss_majority": float(losses[y == 0].mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "f1_fixed": float(f1_score(y, prediction, zero_division=0)),
        "f1_tuned": float(f1_score(y, tuned_prediction, zero_division=0)),
    }


def tune_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.arange(0.0, 1.0, 0.01)
    scores = [f1_score(y, probability > candidate, zero_division=0) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def apply_sampling(x, y, method: str, seed: int, ratio: float = 0.5, k: int = 5):
    if method == "none":
        return np.asarray(x), np.asarray(y)
    if method == "random":
        sampler = RandomOverSampler(sampling_strategy=ratio, random_state=seed)
    elif method == "smote":
        sampler = SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=seed)
    else:
        raise ValueError(f"unknown sampling method: {method}")
    return sampler.fit_resample(x, y)


def _fit(learner, x_train, y_train, x_validation, y_validation, seed):
    if learner == "dt":
        return _fit_decision_tree(
            x_train,
            y_train,
            x_validation,
            y_validation,
            seed,
            {"min_samples_leaf_factor": 0.01},
        )
    if learner == "cat":
        return _fit_catboost(
            x_train,
            y_train,
            x_validation,
            y_validation,
            seed,
            {"iterations": 500},
        )
    raise ValueError(f"unknown learner: {learner}")


def _probability(model, x):
    return np.asarray(model.predict_proba(x))[:, 1]


def run_slice(
    output: Path,
    datasets=("ecoli", "yeast_me2", "abalone_19", "mammography"),
    learners=("dt", "cat"),
    methods=("none", "random", "smote"),
    seeds=(0, 1, 2),
    run_order: Path | None = None,
):
    """Run the first Gate-unlocked full-experiment slice with resume support."""
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost")
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    specs = build_run_specs(
        datasets=datasets,
        learners=learners,
        methods=methods,
        seeds=seeds,
        scenarios=("iid",),
        severities=(0,),
    )
    specs = order_run_specs(specs, run_order)
    protocol = {
        "datasets": list(datasets),
        "learners": list(learners),
        "methods": list(methods),
        "seeds": list(seeds),
        "scenarios": ["iid"],
        "severities": [0],
        "metrics": list(METRICS),
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
        "reference_package_versions": REFERENCE_PACKAGE_VERSIONS,
        "package_version_differences_from_main": package_version_differences(packages),
        "interpretation_scope": "within-task paired descriptions under this recorded runtime; do not pool with the 24-task main results",
    }
    if run_order is not None:
        schedule_path = Path(run_order)
        protocol["run_order_schedule"] = schedule_path.name
        protocol["run_order_sha256"] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": stable_hash(protocol),
        "expected_fit_count": len(specs),
        "expected_metric_rows": len(specs) * len(METRICS),
        "packages": packages,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    completed = set(existing["run_id"].astype(str)) if not existing.empty else set()
    datasets_loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = build_data_manifest(datasets_loaded, datasets)
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["data_manifest_hash"] = stable_hash(data_manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
        )
        x_train, y_train = apply_sampling(x[train], y[train], spec["method"], spec["seed"])
        started = time.perf_counter()
        model = _fit(
            spec["learner"],
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
            "learner": spec["learner"],
            "method": spec["method"],
            "ratio": 0.5 if spec["method"] != "none" else None,
            "k": 5 if spec["method"] == "smote" else None,
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
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        frame.to_csv(raw_path, index=False)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame.to_csv(raw_path, index=False)
    summary = aggregate_results(raw_path, output / "summary.csv")
    state = {
        "status": "COMPLETE" if len(set(frame["run_id"])) == len(specs) else "INCOMPLETE",
        "fit_count": int(frame["run_id"].nunique()),
        "expected_fit_count": len(specs),
        "raw_metric_rows": len(frame),
        "expected_metric_rows": len(specs) * len(METRICS),
        "summary_rows": len(summary),
        "protocol_hash": manifest["protocol_hash"],
    }
    (output / "status.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-slice", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--learners", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--run-order", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the selected runs, schedule, cached data, and software without fitting models",
    )
    args = parser.parse_args()
    if args.dry_run and not args.run_slice:
        parser.error("--dry-run requires --run-slice")

    datasets = tuple(args.datasets or ("ecoli", "yeast_me2", "abalone_19", "mammography"))
    learners = tuple(args.learners or ("dt", "cat"))
    methods = tuple(args.methods or ("none", "random", "smote"))
    seeds = tuple(args.seeds or (0, 1, 2))

    if args.dry_run:
        specs = build_run_specs(
            datasets=datasets,
            learners=learners,
            methods=methods,
            seeds=seeds,
            scenarios=("iid",),
            severities=(0,),
        )
        specs = order_run_specs(specs, args.run_order)
        datasets_loaded = fetch_datasets(filter_data=datasets, download_if_missing=False)
        data_manifest = build_data_manifest(datasets_loaded, datasets)
        packages = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost")
        }
        version_differences = package_version_differences(packages)
        schedule_hash = (
            hashlib.sha256(args.run_order.read_bytes()).hexdigest()
            if args.run_order is not None
            else None
        )
        result = {
            "status": "DRY_RUN_OK",
            "expected_fit_count": len(specs),
            "expected_metric_rows": len(specs) * len(METRICS),
            "ordered_run_ids": [spec["run_id"] for spec in specs],
            "run_order_sha256": schedule_hash,
            "data_manifest": data_manifest,
            "packages": packages,
            "reference_package_versions": REFERENCE_PACKAGE_VERSIONS,
            "package_version_differences_from_main": version_differences,
            "interpretation_scope": "within-task paired descriptions only; no direct pooled comparison to the 24-task main results",
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if not args.run_slice:
        specs = build_run_specs(
            datasets=datasets,
            learners=learners,
            methods=methods,
            seeds=seeds,
        )
        print(json.dumps({"expected_fit_count": len(specs), "expected_metric_rows": len(specs) * len(METRICS)}))
        return 0
    state = run_slice(
        args.output,
        datasets=datasets,
        learners=learners,
        methods=methods,
        seeds=seeds,
        run_order=args.run_order,
    )
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
