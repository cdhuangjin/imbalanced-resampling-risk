"""Run the frozen residual-shift safety-gate experiment."""

from __future__ import annotations

import argparse
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


EXTENSION_DIR = Path(__file__).resolve().parent
CODE_DIR = EXTENSION_DIR.parent / "code"
CALIBRATION_DIR = EXTENSION_DIR.parent / "calibration_extension"
SHIFT_DIR = EXTENSION_DIR.parent / "shift_correction_extension"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(SHIFT_DIR))

from factor_experiment import (  # noqa: E402
    LEARNER_PARAMS,
    METHOD_PARAMS,
    fit_factor_model,
    parse_learner_spec,
    parse_method_spec,
)
from full_experiment import (  # noqa: E402
    _probability,
    apply_sampling,
    build_data_manifest,
    sha_bytes,
    split_indices,
    stable_hash,
)
from calibration_extension import (  # noqa: E402
    compute_extended_metrics,
    fit_platt_calibrator,
    prior_correct_probability,
    source_prior_weights,
    target_sample_indices,
)
from run_shift_correction import build_postprocessor_variants  # noqa: E402
from shift_experiment import apply_covariate_shift  # noqa: E402
from residual_shift_gate import (  # noqa: E402
    cross_fitted_domain_auc,
    gate_consensus_probability,
    regime_adaptive_shrinkage,
    shrink_consensus_probability,
    uncertainty_aware_shrinkage,
)
from shift_correction import estimate_prior_stability  # noqa: E402


GATE_TASKS = (
    "ecoli",
    "yeast_me2",
    "abalone_19",
    "mammography",
    "oil",
    "spectrometer",
)
FULL_TASKS = (
    "abalone",
    "abalone_19",
    "arrhythmia",
    "car_eval_34",
    "car_eval_4",
    "coil_2000",
    "ecoli",
    "letter_img",
    "libras_move",
    "mammography",
    "oil",
    "optical_digits",
    "ozone_level",
    "pen_digits",
    "satimage",
    "scene",
    "sick_euthyroid",
    "solar_flare_m0",
    "spectrometer",
    "thyroid_sick",
    "us_crime",
    "wine_quality",
    "yeast_me2",
    "yeast_ml8",
)
LEARNERS = ("dt_leaf01", "cat_iter500")
METHODS = ("none", "random_q05", "smote_q05_k5")
LABEL_SHIFT_PRIORS = {0: 0.05, 1: 0.20, 2: 0.50}
COVARIATE_SHIFT_SEVERITIES = (1, 2)
POSTPROCESSORS = (
    "platt_source",
    "platt_target_consensus",
    "platt_target_gated",
    "platt_target_oracle",
)
SHRINKAGE_POSTPROCESSORS = POSTPROCESSORS[:-1] + ("platt_target_shrinkage",) + POSTPROCESSORS[-1:]
STABILITY_POSTPROCESSORS = POSTPROCESSORS[:-1] + (
    "platt_target_shrinkage",
    "platt_target_rsc",
) + POSTPROCESSORS[-1:]
REGIME_POSTPROCESSORS = STABILITY_POSTPROCESSORS[:-1] + (
    "platt_target_rass",
) + STABILITY_POSTPROCESSORS[-1:]
METRICS = ("auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10")
RAW_COLUMNS = (
    "fit_id",
    "dataset",
    "learner",
    "method",
    "seed",
    "scenario",
    "severity",
    "target_prior",
    "covariate_severity",
    "postprocessor",
    "metric",
    "value",
    "source_prior",
    "resampled_prior",
    "estimated_target_prior",
    "em_estimate",
    "bbse_hard_estimate",
    "bbse_soft_estimate",
    "em_valid",
    "bbse_hard_valid",
    "bbse_soft_valid",
    "em_iterations",
    "em_converged",
    "bbse_hard_condition_number",
    "bbse_soft_condition_number",
    "bbse_hard_residual",
    "bbse_soft_residual",
    "consensus_used_fallback",
    "consensus_reason",
    "residual_auc",
    "residual_valid",
    "residual_accepted",
    "residual_reason",
    "gated_used_fallback",
    "gated_reason",
    "shrinkage_adaptation_weight",
    "rsc_residual_adaptation_weight",
    "rsc_stability_weight",
    "prior_stability_valid_rate",
    "prior_stability_agreement_rate",
    "prior_stability_em_std",
    "prior_stability_replicates",
    "rass_selected_policy",
    "rass_reason",
    "rass_threshold",
    "data_hash",
    "split_hash",
    "target_sample_hash",
    "config_hash",
    "runtime_seconds",
    "timestamp",
)


def build_fit_specs(datasets=GATE_TASKS, learners=LEARNERS, methods=METHODS, seeds=(0, 1, 2)):
    return [
        {
            "fit_id": f"{dataset}__{learner}__s{seed}__{method}",
            "dataset": str(dataset),
            "learner": str(learner),
            "seed": int(seed),
            "method": str(method),
        }
        for dataset in datasets
        for learner in learners
        for seed in seeds
        for method in methods
    ]


def build_scenarios():
    scenarios = [{"name": "iid", "severity": 0, "target_prior": None, "covariate_severity": 0}]
    scenarios.extend(
        {
            "name": "label_shift",
            "severity": int(severity),
            "target_prior": float(prior),
            "covariate_severity": 0,
        }
        for severity, prior in LABEL_SHIFT_PRIORS.items()
    )
    scenarios.extend(
        {
            "name": "covariate_shift",
            "severity": int(severity),
            "target_prior": None,
            "covariate_severity": int(severity),
        }
        for severity in COVARIATE_SHIFT_SEVERITIES
    )
    joint_severity = 0
    for prior in (0.20, 0.50):
        for covariate_severity in COVARIATE_SHIFT_SEVERITIES:
            scenarios.append(
                {
                    "name": "joint_shift",
                    "severity": joint_severity,
                    "target_prior": float(prior),
                    "covariate_severity": int(covariate_severity),
                }
            )
            joint_severity += 1
    return scenarios


def _target_sample_hash(indices, scenario):
    payload = np.asarray(indices, dtype=np.int64).tobytes()
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(json.dumps(scenario, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def prepare_scenario(
    x_test,
    y_test,
    scenario,
    *,
    train_mean,
    train_scale,
    seed,
):
    """Create an evaluation domain and an unlabeled target-feature sample.

    `y_test` is used only by this synthetic-data constructor to realize a
    declared target prior. The returned target feature matrix is the only
    target object passed to adaptation and detection code.
    """
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)
    declared_prior = scenario["target_prior"]
    target_prior = float(np.mean(y_test)) if declared_prior is None else float(declared_prior)
    x_eval = x_test.copy()
    if int(scenario["covariate_severity"]) > 0:
        x_eval, _ = apply_covariate_shift(
            x_eval, np.asarray(train_mean), np.asarray(train_scale), int(scenario["covariate_severity"])
        )
    target_indices = target_sample_indices(
        y_test,
        target_prior,
        seed=int(seed),
    )
    return {
        "x_eval": x_eval,
        "target_features": x_eval[target_indices],
        "target_prior": target_prior,
        "target_indices": target_indices,
        "evaluation_weights": (
            source_prior_weights(y_test, target_prior)
            if declared_prior is not None
            else np.ones(y_test.size, dtype=float)
        ),
    }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def build_gated_variants(
    *,
    validation_features,
    validation_labels,
    validation_probability,
    test_probability,
    target_sample_features,
    source_prior,
    target_prior,
    seed,
    include_stability=False,
    include_regime=False,
):
    """Build V4 variants and residual policies without receiving target labels."""
    v4 = build_postprocessor_variants(
        validation_probability=validation_probability,
        validation_label=validation_labels,
        test_probability=test_probability,
        target_sample_probability=np.asarray(target_sample_features["probability"]),
        source_prior=source_prior,
        resampled_prior=source_prior,
        target_prior=target_prior,
    )
    consensus = v4["platt_target_consensus"]
    prior_hat = float(consensus["estimated_target_prior"])
    try:
        source_weights = source_prior_weights(validation_labels, prior_hat)
        residual = cross_fitted_domain_auc(
            validation_features,
            target_sample_features["features"],
            source_weights,
            seed=int(seed),
        )
    except (TypeError, ValueError) as exc:
        residual = {
            "valid": False,
            "residual_auc": None,
            "accepted": False,
            "reason": f"detector_input_failed:{exc}",
        }
    gated = gate_consensus_probability(
        v4["platt_source"]["probability"],
        consensus["probability"],
        residual,
    )
    shrinkage = shrink_consensus_probability(
        v4["platt_source"]["probability"],
        consensus["probability"],
        residual,
    )
    build_stability = bool(include_stability or include_regime)
    if not build_stability:
        return v4, gated, shrinkage, residual

    calibrator = fit_platt_calibrator(validation_probability, validation_labels)
    stability = estimate_prior_stability(
        calibrator(validation_probability),
        validation_labels,
        calibrator(np.asarray(target_sample_features["probability"])),
        source_prior=source_prior,
        seed=int(seed) + 17000,
        replicates=16,
        agreement_tolerance=0.10,
        stability_scale=0.05,
    )
    rsc = uncertainty_aware_shrinkage(
        v4["platt_source"]["probability"],
        consensus["probability"],
        residual,
        stability,
    )
    if not include_regime:
        return v4, gated, shrinkage, rsc, residual, stability
    rass = regime_adaptive_shrinkage(
        shrinkage["probability"],
        rsc["probability"],
        residual,
        threshold=0.60,
    )
    return v4, gated, shrinkage, rsc, rass, residual, stability


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_raw_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, RAW_COLUMNS].to_csv(path, index=False, float_format="%.15g")


def _validate_row(row):
    missing = [column for column in RAW_COLUMNS if column not in row]
    if missing:
        raise ValueError("missing raw row fields: " + ", ".join(missing))
    if row["postprocessor"] not in REGIME_POSTPROCESSORS or row["metric"] not in METRICS:
        raise ValueError("unknown postprocessor or metric")
    if not np.isfinite(float(row["value"])):
        raise ValueError("metric value must be finite")
    return row


def _protocol(
    datasets,
    learners,
    methods,
    seeds,
    postprocessors,
    include_shrinkage,
    include_stability,
    include_regime,
):
    return {
        "datasets": list(datasets),
        "learners": list(learners),
        "learner_params": {name: LEARNER_PARAMS[name][1] for name in learners},
        "methods": list(methods),
        "method_params": {name: METHOD_PARAMS[name][1] for name in methods},
        "seeds": list(seeds),
        "scenarios": build_scenarios(),
        "postprocessors": list(postprocessors),
        "policy_variant": (
            "v5.6_regime_adaptive_stability"
            if include_regime
            else "v5.5_residual_stability_calibration"
            if include_stability
            else "v5.1_posthoc_shrinkage"
            if include_shrinkage
            else "v5_binary_gate"
        ),
        "metrics": list(METRICS),
        "residual_auc_threshold": 0.60,
        "residual_auc_sensitivity_thresholds": [0.55, 0.65],
        "shrinkage_knots": {"lower": 0.55, "upper": 0.75},
        "prior_stability": {
            "replicates": 16,
            "agreement_tolerance": 0.10,
            "stability_scale": 0.05,
            "resampling": "target_probability_bootstrap_only",
        },
        "regime_switch": {
            "threshold": 0.60,
            "accepted_regime_policy": "v5.1_shrinkage",
            "detected_shift_policy": "v5.5_rsc",
        },
        "domain_classifier": {
            "model": "standardized_logistic_regression",
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 1000,
            "cross_folds": 3,
        },
        "target_sample_size": "max(2000, n_test)",
        "covariate_shift_rule": "first non-constant feature by train-fold standard deviation",
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
        "primary_outputs_immutable": True,
    }


def _existing_completed(existing, expected_rows):
    if existing.empty:
        return set()
    counts = existing.groupby("fit_id", dropna=False).size()
    return set(counts[counts == expected_rows].index.astype(str))


def run_residual_shift_gate(
    output: Path,
    *,
    datasets=GATE_TASKS,
    learners=LEARNERS,
    methods=METHODS,
    seeds=(0, 1, 2),
    include_shrinkage=False,
    include_stability=False,
    include_regime=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    specs = build_fit_specs(datasets, learners, methods, seeds)
    scenarios = build_scenarios()
    if include_regime:
        postprocessors = REGIME_POSTPROCESSORS
    elif include_stability:
        postprocessors = STABILITY_POSTPROCESSORS
    else:
        postprocessors = SHRINKAGE_POSTPROCESSORS if include_shrinkage else POSTPROCESSORS
    protocol = _protocol(
        datasets,
        learners,
        methods,
        seeds,
        postprocessors,
        include_shrinkage,
        include_stability,
        include_regime,
    )
    protocol_hash = stable_hash(protocol)
    expected_rows_per_fit = len(scenarios) * len(postprocessors) * len(METRICS)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("protocol_hash") != protocol_hash:
            raise ValueError("cannot resume with a changed protocol hash")
    manifest = {
        "schema_version": 1,
        "raw_columns": list(RAW_COLUMNS),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "expected_fit_count": len(specs),
        "expected_raw_rows": len(specs) * expected_rows_per_fit,
        "runner_sha256": file_sha256(Path(__file__)),
        "diagnostic_sha256": file_sha256(EXTENSION_DIR / "residual_shift_gate.py"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "catboost",
                "scipy",
            )
        },
    }
    loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = build_data_manifest(loaded, datasets)
    manifest["data_manifest_hash"] = stable_hash(data_manifest)
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    if not existing.empty:
        missing = [column for column in RAW_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError("existing raw output has missing columns: " + ", ".join(missing))
    completed = _existing_completed(existing, expected_rows_per_fit)
    rows = existing.to_dict(orient="records")
    for spec in specs:
        if spec["fit_id"] in completed:
            continue
        dataset = loaded[spec["dataset"]]
        x = np.asarray(dataset["data"], dtype=float)
        y = (np.asarray(dataset["target"]) == 1).astype(int)
        train, validation, test = split_indices(y, spec["seed"])
        data_hash = sha_bytes(x, y)
        split_hash = stable_hash(
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
        )
        method, method_params = parse_method_spec(spec["method"])
        x_train, y_train = apply_sampling(
            x[train], y[train], method, spec["seed"],
            ratio=method_params.get("ratio", 0.5), k=method_params.get("k", 5),
        )
        source_prior = float(np.mean(y[train]))
        resampled_prior = float(np.mean(y_train))
        family, learner_params = parse_learner_spec(spec["learner"])
        started = time.perf_counter()
        model = fit_factor_model(
            family, learner_params, x_train, y_train, x[validation], y[validation], spec["seed"]
        )
        validation_probability = _probability(model, x[validation])
        test_probability_base = _probability(model, x[test])
        train_mean = np.mean(x[train], axis=0)
        train_scale = np.std(x[train], axis=0)
        runtime = time.perf_counter() - started
        for scenario in scenarios:
            scenario_data = prepare_scenario(
                x[test], y[test], scenario,
                train_mean=train_mean, train_scale=train_scale,
                seed=spec["seed"] + 10000 + int(scenario["severity"]),
            )
            test_probability = _probability(model, scenario_data["x_eval"])
            target_sample_features = {
                "features": scenario_data["target_features"],
                "probability": _probability(model, scenario_data["target_features"]),
            }
            built_variants = build_gated_variants(
                validation_features=x[validation],
                validation_labels=y[validation],
                validation_probability=validation_probability,
                test_probability=test_probability,
                target_sample_features=target_sample_features,
                source_prior=source_prior,
                target_prior=scenario_data["target_prior"],
                seed=spec["seed"] + int(scenario["severity"]),
                include_stability=include_stability,
                include_regime=include_regime,
            )
            if include_regime:
                v4, gated, shrinkage, rsc, rass, residual, stability = built_variants
            elif include_stability:
                v4, gated, shrinkage, rsc, residual, stability = built_variants
            else:
                v4, gated, shrinkage, residual = built_variants
                rsc = None
                rass = None
                stability = {}
            variants = {
                "platt_source": v4["platt_source"]["probability"],
                "platt_target_consensus": v4["platt_target_consensus"]["probability"],
                "platt_target_gated": gated["probability"],
                "platt_target_shrinkage": shrinkage["probability"],
                "platt_target_oracle": v4["platt_target_oracle"]["probability"],
            }
            if include_stability or include_regime:
                variants["platt_target_rsc"] = rsc["probability"]
            if include_regime:
                variants["platt_target_rass"] = rass["probability"]
            diagnostics = v4["platt_target_consensus"]
            target_sample_hash = _target_sample_hash(scenario_data["target_indices"], scenario)
            config = {
                "learner_spec": spec["learner"],
                "learner_params": learner_params,
                "method_spec": spec["method"],
                "method_params": method_params,
                "scenario": scenario,
                "residual_auc_threshold": 0.60,
                "threshold_source": "frozen_protocol",
            }
            config_hash = stable_hash(config)
            for postprocessor in postprocessors:
                probability = variants[postprocessor]
                metrics = compute_extended_metrics(
                    y[test], probability, sample_weight=scenario_data["evaluation_weights"]
                )
                for metric, value in metrics.items():
                    gated_fallback = postprocessor == "platt_target_gated" and bool(gated["used_fallback"])
                    row = _validate_row(
                        {
                            **spec,
                            "scenario": scenario["name"],
                            "severity": int(scenario["severity"]),
                            "target_prior": float(scenario_data["target_prior"]),
                            "covariate_severity": int(scenario["covariate_severity"]),
                            "postprocessor": postprocessor,
                            "metric": metric,
                            "value": value,
                            "source_prior": source_prior,
                            "resampled_prior": resampled_prior,
                            "estimated_target_prior": diagnostics["estimated_target_prior"],
                            "em_estimate": diagnostics["em_estimate"],
                            "bbse_hard_estimate": diagnostics["bbse_hard_estimate"],
                            "bbse_soft_estimate": diagnostics["bbse_soft_estimate"],
                            "em_valid": diagnostics["em_valid"],
                            "bbse_hard_valid": diagnostics["bbse_hard_valid"],
                            "bbse_soft_valid": diagnostics["bbse_soft_valid"],
                            "em_iterations": diagnostics["em_iterations"],
                            "em_converged": diagnostics["em_converged"],
                            "bbse_hard_condition_number": diagnostics["bbse_hard_condition_number"],
                            "bbse_soft_condition_number": diagnostics["bbse_soft_condition_number"],
                            "bbse_hard_residual": diagnostics["bbse_hard_residual"],
                            "bbse_soft_residual": diagnostics["bbse_soft_residual"],
                            "consensus_used_fallback": diagnostics["used_fallback"],
                            "consensus_reason": diagnostics["consensus_reason"],
                            "residual_auc": _finite_or_none(residual.get("residual_auc")),
                            "residual_valid": bool(residual.get("valid", False)),
                            "residual_accepted": bool(residual.get("accepted", False)),
                            "residual_reason": residual.get("reason"),
                            "gated_used_fallback": gated_fallback,
                            "gated_reason": gated["reason"] if postprocessor == "platt_target_gated" else "not_applicable",
                            "shrinkage_adaptation_weight": (
                                shrinkage["adaptation_weight"]
                                if include_shrinkage or include_stability or include_regime
                                else None
                            ),
                            "rsc_residual_adaptation_weight": (
                                rsc["residual_adaptation_weight"]
                                if include_stability or include_regime
                                else None
                            ),
                            "rsc_stability_weight": (
                                rsc["stability_weight"]
                                if include_stability or include_regime
                                else None
                            ),
                            "prior_stability_valid_rate": (
                                stability.get("valid_rate")
                                if include_stability or include_regime
                                else None
                            ),
                            "prior_stability_agreement_rate": (
                                stability.get("agreement_rate")
                                if include_stability or include_regime
                                else None
                            ),
                            "prior_stability_em_std": (
                                stability.get("em_std")
                                if include_stability or include_regime
                                else None
                            ),
                            "prior_stability_replicates": (
                                stability.get("replicates")
                                if include_stability or include_regime
                                else None
                            ),
                            "rass_selected_policy": (
                                rass["selected_policy"] if include_regime else "not_applicable"
                            ),
                            "rass_reason": (
                                rass["reason"] if include_regime else "not_applicable"
                            ),
                            "rass_threshold": (
                                rass["threshold"] if include_regime else None
                            ),
                            "data_hash": data_hash,
                            "split_hash": split_hash,
                            "target_sample_hash": target_sample_hash,
                            "config_hash": config_hash,
                            "runtime_seconds": runtime,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    rows.append(row)
        write_raw_frame(pd.DataFrame(rows, columns=RAW_COLUMNS), raw_path)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    write_raw_frame(frame, raw_path)
    state = {
        "status": "COMPLETE" if frame["fit_id"].nunique() == len(specs) else "INCOMPLETE",
        "fit_count": int(frame["fit_id"].nunique()),
        "expected_fit_count": len(specs),
        "raw_rows": int(len(frame)),
        "expected_raw_rows": int(manifest["expected_raw_rows"]),
        "protocol_hash": protocol_hash,
        "data_manifest_hash": manifest["data_manifest_hash"],
    }
    (output / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--shrinkage", action="store_true")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--regime", action="store_true")
    args = parser.parse_args()
    datasets = tuple(v.strip() for v in args.datasets.split(",") if v.strip()) if args.datasets else GATE_TASKS
    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip()) if args.seeds else (0, 1, 2)
    state = run_residual_shift_gate(
        args.output,
        datasets=datasets,
        seeds=seeds,
        include_shrinkage=bool(args.shrinkage),
        include_stability=bool(args.stability),
        include_regime=bool(args.regime),
    )
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
