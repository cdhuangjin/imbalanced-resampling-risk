"""Run the utility-first selective-treatment Gate."""

from __future__ import annotations

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

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

import run_adaptive as base  # noqa: E402
from confidence_guarded import attach_raw_metrics  # noqa: E402
from utility_guarded import (  # noqa: E402
    tune_cost_threshold,
    utility_guarded_select,
    utility_rows,
)


CANDIDATES = base.CANDIDATES
POLICIES = CANDIDATES + ("utility_guarded",)
SCENARIOS = base.SCENARIOS
METRICS = base.METRICS + ("decision_utility_c5",)
RAW_COLUMNS = tuple(base.RAW_COLUMNS) + ("utility_threshold",)
DECISION_COLUMNS = tuple(base.DECISION_COLUMNS) + ("utility_threshold",)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame, path, columns):
    frame.loc[:, list(columns)].to_csv(path, index=False, float_format="%.15g")


def weighted_utility(y, probability, threshold, sample_weight=None):
    values = utility_rows(y, probability, threshold)
    return float(np.mean(values) if sample_weight is None else np.average(values, weights=sample_weight))


def run(output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = base.GATE_TASKS
    learners = base.LEARNERS
    seeds = base.GATE_SEEDS
    specs = base.build_fit_specs(datasets, learners, seeds)
    protocol = {
        "name": "utility_first_risk_guard",
        "datasets": list(datasets),
        "learners": list(learners),
        "seeds": list(seeds),
        "candidates": list(CANDIDATES),
        "policies": list(POLICIES),
        "scenarios": [name for name, _, _ in SCENARIOS],
        "metrics": list(METRICS),
        "threshold_grid": "0.01:0.99 step 0.01, lowest tie",
        "decision_costs": {"fn": 5.0, "fp": 1.0},
        "majority_logloss_tolerance": 0.02,
        "brier_tolerance": 0.005,
        "validation_bootstrap_replicates": 400,
        "validation_bootstrap_seed": 20260917,
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
        "primary_outputs_immutable": True,
    }
    protocol_hash = base.stable_hash(protocol)
    loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = base.build_data_manifest(loaded, datasets)
    data_manifest_hash = base.stable_hash(data_manifest)
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "expected_fit_count": len(specs),
        "expected_candidate_fits": len(specs) * len(CANDIDATES),
        "expected_raw_rows": len(specs) * len(POLICIES) * len(SCENARIOS) * len(METRICS),
        "runner_sha256": file_sha256(Path(__file__)),
        "base_runner_sha256": file_sha256(PACKAGE_DIR / "run_adaptive.py"),
        "selector_sha256": file_sha256(PACKAGE_DIR / "utility_guarded.py"),
        "data_manifest_hash": data_manifest_hash,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost", "scipy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    decision_rows = []
    for spec in specs:
        dataset = loaded[spec["dataset"]]
        x = np.asarray(dataset["data"], dtype=float)
        y = (np.asarray(dataset["target"]) == 1).astype(int)
        train, validation, test = base.split_indices(y, spec["seed"])
        split_hash = base.stable_hash({"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()})
        data_hash = base.sha_bytes(x, y)
        candidate_models = {}
        support_diagnostics = {}
        candidate_runtimes = {}
        family, learner_params = base.parse_learner_spec(spec["learner"])
        for candidate in CANDIDATES:
            x_candidate, y_candidate, diagnostics = base._candidate_training(x[train], y[train], candidate, spec["seed"])
            support_diagnostics[candidate] = diagnostics
            started = time.perf_counter()
            model = base.fit_factor_model(family, learner_params, x_candidate, y_candidate, x[validation], y[validation], spec["seed"])
            candidate_runtimes[candidate] = time.perf_counter() - started
            candidate_models[candidate] = {
                "validation_probability": base._probability(model, x[validation]),
                "test_probability": base._probability(model, x[test]),
                "source_prior": float(np.mean(y[train])),
                "resampled_prior": float(np.mean(y_candidate)),
                "utility_threshold": tune_cost_threshold(y[validation], base._probability(model, x[validation])),
                "model_config": {
                    "learner_spec": spec["learner"],
                    "learner_family": family,
                    "learner_params": learner_params,
                    "candidate": candidate,
                    "support_diagnostics": diagnostics,
                },
            }
        selector_priors = tuple(sorted(set(base.SELECTOR_TARGET_PRIORS + (candidate_models["none"]["source_prior"],))))
        validation_metrics = {
            candidate: attach_raw_metrics(
                base.validation_policy_metrics(y[validation], values["validation_probability"], target_priors=selector_priors),
                y=y[validation],
                probability=values["validation_probability"],
            )
            for candidate, values in candidate_models.items()
        }
        selection = utility_guarded_select(
            validation_metrics,
            candidate_order=CANDIDATES,
            majority_logloss_tolerance=0.02,
            brier_tolerance=0.005,
            bootstrap_replicates=400,
            bootstrap_seed=20260917,
        )
        selected_candidate = selection["selected"]
        selection_score = float(selection["utility_score"][selected_candidate])
        total_runtime = float(sum(candidate_runtimes.values()))
        timestamp = datetime.now(timezone.utc).isoformat()
        decision_rows.append({
            "fit_id": spec["fit_id"], "dataset": spec["dataset"], "learner": spec["learner"], "seed": spec["seed"],
            "selected_candidate": selected_candidate, "selected_is_none": bool(selected_candidate == "none"),
            "selection_score": selection_score, "feasibility_json": base._json(selection["feasibility"]),
            "minority_score_json": base._json(selection["utility_score"]),
            "constraint_diagnostics_json": base._json(selection["constraint_diagnostics"]),
            "support_diagnostics_json": base._json(support_diagnostics),
            "validation_hash": base.sha_bytes(y[validation], candidate_models[selected_candidate]["validation_probability"]),
            "runtime_seconds": total_runtime, "timestamp": timestamp,
            "utility_threshold": float(selection["utility_thresholds"][selected_candidate]),
        })
        for scenario, severity, declared_target_prior in SCENARIOS:
            source_prior = candidate_models["none"]["source_prior"]
            target_prior = source_prior if declared_target_prior is None else float(declared_target_prior)
            weights = None if scenario == "iid" else base.source_prior_weights(y[test], target_prior)
            for policy in POLICIES:
                candidate = selected_candidate if policy == "utility_guarded" else policy
                model_values = candidate_models[candidate]
                values = base.compute_extended_metrics(y[test], model_values["test_probability"], sample_weight=weights)
                values["decision_utility_c5"] = weighted_utility(y[test], model_values["test_probability"], model_values["utility_threshold"], sample_weight=weights)
                config = {
                    **model_values["model_config"], "policy": policy, "training_candidate": candidate,
                    "selected_candidate": selected_candidate, "scenario": scenario, "severity": severity,
                    "target_prior": target_prior, "utility_threshold": model_values["utility_threshold"],
                    "selector": {"majority_logloss_tolerance": 0.02, "brier_tolerance": 0.005, "validation_bootstrap_replicates": 400},
                }
                config_hash = base.stable_hash(config)
                for metric in METRICS:
                    rows.append({
                        "fit_id": spec["fit_id"], "dataset": spec["dataset"], "learner": spec["learner"], "seed": spec["seed"],
                        "policy": policy, "training_candidate": candidate, "selected_candidate": selected_candidate,
                        "scenario": scenario, "severity": severity, "metric": metric, "value": float(values[metric]),
                        "source_prior": source_prior, "resampled_prior": model_values["resampled_prior"], "target_prior": target_prior,
                        "support_eligible_seed_count": support_diagnostics[candidate]["eligible_seed_count"],
                        "support_generated_count": support_diagnostics[candidate]["generated_count"], "selection_score": selection_score,
                        "config_hash": config_hash, "data_hash": data_hash, "split_hash": split_hash,
                        "runtime_seconds": total_runtime, "timestamp": timestamp, "utility_threshold": model_values["utility_threshold"],
                    })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    decisions = pd.DataFrame(decision_rows, columns=DECISION_COLUMNS)
    _write_csv(frame, output / "raw_results.csv", RAW_COLUMNS)
    _write_csv(decisions, output / "selection_decisions.csv", DECISION_COLUMNS)
    state = {
        "status": "COMPLETE" if frame["fit_id"].nunique() == len(specs) and len(frame) == manifest["expected_raw_rows"] and len(decisions) == len(specs) else "INCOMPLETE",
        "fit_count": int(frame["fit_id"].nunique()), "expected_fit_count": len(specs),
        "candidate_fit_count": int(frame[["fit_id", "training_candidate"]].drop_duplicates().shape[0]),
        "expected_candidate_fit_count": manifest["expected_candidate_fits"],
        "raw_rows": int(len(frame)), "expected_raw_rows": manifest["expected_raw_rows"], "decision_rows": int(len(decisions)),
        "protocol_hash": protocol_hash, "data_manifest_hash": data_manifest_hash,
    }
    (output / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = run(args.output)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
