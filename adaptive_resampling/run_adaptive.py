"""Resumable Gate runner for risk-constrained support-gated resampling."""

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


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent / "code"
CALIBRATION_DIR = PACKAGE_DIR.parent / "calibration_extension"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CALIBRATION_DIR))

from adaptive_resampling import (  # noqa: E402
    select_risk_constrained_candidate,
    support_gated_smote,
    validation_policy_metrics,
)
from calibration_extension import compute_extended_metrics, source_prior_weights  # noqa: E402
from factor_experiment import (  # noqa: E402
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


DEFAULT_TASKS = (
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
GATE_TASKS = (
    "ecoli",
    "yeast_me2",
    "abalone_19",
    "mammography",
    "oil",
    "spectrometer",
)
GATE_SEEDS = (0, 1, 2)
FULL_SEEDS = (0, 1, 2, 3, 4)
LEARNERS = ("dt_leaf01", "cat_iter500")
CANDIDATES = (
    "none",
    "random_q05",
    "smote_q05_k5",
    "support_smote_q05_k5",
)
POLICIES = CANDIDATES + ("adaptive_rc",)
METHOD_PARAMS = {
    "none": ("none", {}),
    "random_q05": ("random", {"ratio": 0.5}),
    "smote_q05_k5": ("smote", {"ratio": 0.5, "k": 5}),
}
SELECTOR_TARGET_PRIORS = (0.05, 0.20, 0.50)
LABEL_SHIFT_PRIORS = (0.05, 0.20, 0.50)
METRICS = ("auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10")
SCENARIOS = (
    ("iid", 0, None),
    ("label_shift", 0, 0.05),
    ("label_shift", 1, 0.20),
    ("label_shift", 2, 0.50),
)
RAW_COLUMNS = (
    "fit_id",
    "dataset",
    "learner",
    "seed",
    "policy",
    "training_candidate",
    "selected_candidate",
    "scenario",
    "severity",
    "metric",
    "value",
    "source_prior",
    "resampled_prior",
    "target_prior",
    "support_eligible_seed_count",
    "support_generated_count",
    "selection_score",
    "config_hash",
    "data_hash",
    "split_hash",
    "runtime_seconds",
    "timestamp",
)
DECISION_COLUMNS = (
    "fit_id",
    "dataset",
    "learner",
    "seed",
    "selected_candidate",
    "selected_is_none",
    "selection_score",
    "feasibility_json",
    "minority_score_json",
    "constraint_diagnostics_json",
    "support_diagnostics_json",
    "validation_hash",
    "runtime_seconds",
    "timestamp",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fit_specs(datasets, learners, seeds):
    return [
        {
            "fit_id": f"{dataset}__{learner}__s{int(seed)}",
            "dataset": str(dataset),
            "learner": str(learner),
            "seed": int(seed),
        }
        for dataset in datasets
        for learner in learners
        for seed in seeds
    ]


def build_scenarios():
    return list(SCENARIOS)


def _parse_csv_values(value, parser):
    if value is None:
        return None
    parsed = tuple(parser(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("override must contain at least one value")
    return parsed


def resolve_matrix(*, full, datasets_arg, seeds_arg, gate_decision):
    """Resolve the CLI matrix while preventing partial full-run overrides."""
    requested_datasets = _parse_csv_values(datasets_arg, str)
    requested_seeds = _parse_csv_values(seeds_arg, int)
    if full:
        if gate_decision != "GO":
            raise ValueError("--full requires a GO Gate decision")
        if requested_datasets is not None and requested_datasets != DEFAULT_TASKS:
            raise ValueError("--full requires the frozen 24-task dataset list")
        if requested_seeds is not None and requested_seeds != FULL_SEEDS:
            raise ValueError("--full requires the frozen five-seed list")
        return DEFAULT_TASKS, FULL_SEEDS
    return requested_datasets or GATE_TASKS, requested_seeds or GATE_SEEDS


def _write_csv(frame: pd.DataFrame, path: Path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(columns)].to_csv(path, index=False, float_format="%.15g")


def _candidate_training(x_train, y_train, candidate, seed):
    if candidate == "support_smote_q05_k5":
        return support_gated_smote(
            x_train,
            y_train,
            ratio=0.5,
            k=5,
            min_support=0.5,
            seed=seed,
        )
    method, params = METHOD_PARAMS[candidate]
    sampled_x, sampled_y = apply_sampling(
        x_train,
        y_train,
        method,
        seed,
        ratio=params.get("ratio", 0.5),
        k=params.get("k", 5),
    )
    return (
        np.asarray(sampled_x),
        np.asarray(sampled_y),
        {
            "ratio": params.get("ratio", None),
            "k": params.get("k", None),
            "min_support": None,
            "minority_count": int(np.sum(y_train == 1)),
            "majority_count": int(np.sum(y_train == 0)),
            "eligible_seed_count": int(np.sum(y_train == 1)),
            "generated_count": int(np.sum(np.asarray(sampled_y) == 1) - np.sum(y_train == 1)),
        },
    )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fit_id_counts(frame):
    if frame.empty:
        return {}
    return frame.groupby("fit_id").size().astype(int).to_dict()


def run_adaptive(output: Path, *, datasets, learners=LEARNERS, seeds=(0, 1, 2)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    decisions_path = output / "selection_decisions.csv"
    specs = build_fit_specs(datasets, learners, seeds)
    scenarios = build_scenarios()
    protocol = {
        "datasets": list(datasets),
        "learners": list(learners),
        "seeds": list(seeds),
        "candidate_order": list(CANDIDATES),
        "policies": list(POLICIES),
        "sampling": {
            "random_q05": {"ratio": 0.5},
            "smote_q05_k5": {"ratio": 0.5, "k": 5},
            "support_smote_q05_k5": {"ratio": 0.5, "k": 5, "min_support": 0.5},
        },
        "selector_target_priors": list(SELECTOR_TARGET_PRIORS),
        "majority_logloss_tolerance": 0.02,
        "brier_tolerance": 0.005,
        "scenarios": [name for name, _, _ in scenarios],
        "label_shift_target_priors": list(LABEL_SHIFT_PRIORS),
        "metrics": list(METRICS),
        "ece_bins": 10,
        "data_cache": "<local path omitted>",
        "source": "https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz",
        "primary_outputs_immutable": True,
    }
    protocol_hash = stable_hash(protocol)
    manifest_path = output / "manifest.json"
    current_code_hash = file_sha256(Path(__file__))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("protocol_hash") != protocol_hash:
            raise ValueError("existing adaptive output has a different protocol hash")
        if previous.get("runner_sha256") != current_code_hash:
            raise ValueError("existing adaptive output has a different runner hash")

    loaded = fetch_datasets(filter_data=tuple(datasets), download_if_missing=False)
    data_manifest = build_data_manifest(loaded, datasets)
    data_manifest_hash = stable_hash(data_manifest)
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "expected_fit_count": len(specs),
        "expected_candidate_fits": len(specs) * len(CANDIDATES),
        "expected_raw_rows": len(specs) * len(POLICIES) * len(scenarios) * len(METRICS),
        "runner_sha256": current_code_hash,
        "adaptive_module_sha256": file_sha256(PACKAGE_DIR / "adaptive_resampling.py"),
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
        "data_manifest_hash": data_manifest_hash,
    }
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_raw = (
        pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    )
    existing_decisions = (
        pd.read_csv(decisions_path)
        if decisions_path.exists()
        else pd.DataFrame(columns=DECISION_COLUMNS)
    )
    expected_rows_per_fit = len(POLICIES) * len(scenarios) * len(METRICS)
    counts = _fit_id_counts(existing_raw)
    decision_ids = set(existing_decisions.get("fit_id", pd.Series(dtype=str)).astype(str))
    completed = set()
    for spec in specs:
        count = counts.get(spec["fit_id"], 0)
        has_decision = spec["fit_id"] in decision_ids
        if count not in (0, expected_rows_per_fit):
            raise ValueError(f"partial raw rows for fit {spec['fit_id']}")
        if count == expected_rows_per_fit and not has_decision:
            raise ValueError(f"missing selection decision for complete fit {spec['fit_id']}")
        if count == expected_rows_per_fit and has_decision:
            completed.add(spec["fit_id"])

    rows = existing_raw.to_dict(orient="records")
    decision_rows = existing_decisions.to_dict(orient="records")
    changed = False
    for spec in specs:
        if spec["fit_id"] in completed:
            continue
        dataset = loaded[spec["dataset"]]
        x = np.asarray(dataset["data"], dtype=float)
        y = (np.asarray(dataset["target"]) == 1).astype(int)
        train, validation, test = split_indices(y, spec["seed"])
        split_hash = stable_hash(
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
        )
        data_hash = sha_bytes(x, y)
        candidate_models = {}
        support_diagnostics = {}
        candidate_runtimes = {}
        for candidate in CANDIDATES:
            x_candidate, y_candidate, diagnostics = _candidate_training(
                x[train], y[train], candidate, spec["seed"]
            )
            support_diagnostics[candidate] = diagnostics
            family, learner_params = parse_learner_spec(spec["learner"])
            started = time.perf_counter()
            model = fit_factor_model(
                family,
                learner_params,
                x_candidate,
                y_candidate,
                x[validation],
                y[validation],
                spec["seed"],
            )
            candidate_runtimes[candidate] = time.perf_counter() - started
            candidate_models[candidate] = {
                "validation_probability": _probability(model, x[validation]),
                "test_probability": _probability(model, x[test]),
                "source_prior": float(np.mean(y[train])),
                "resampled_prior": float(np.mean(y_candidate)),
                "model_config": {
                    "learner_spec": spec["learner"],
                    "learner_family": family,
                    "learner_params": learner_params,
                    "candidate": candidate,
                    "support_diagnostics": diagnostics,
                },
            }
        selector_priors = tuple(
            sorted(set(SELECTOR_TARGET_PRIORS + (candidate_models["none"]["source_prior"],)))
        )
        validation_metrics = {
            candidate: validation_policy_metrics(
                y[validation], values["validation_probability"], target_priors=selector_priors
            )
            for candidate, values in candidate_models.items()
        }
        selection = select_risk_constrained_candidate(
            validation_metrics,
            candidate_order=CANDIDATES,
            majority_logloss_tolerance=0.02,
            brier_tolerance=0.005,
        )
        selected_candidate = selection["selected"]
        selection_score = float(selection["minority_score"][selected_candidate])
        total_runtime = float(sum(candidate_runtimes.values()))
        decision_timestamp = datetime.now(timezone.utc).isoformat()
        decision_rows.append(
            {
                "fit_id": spec["fit_id"],
                "dataset": spec["dataset"],
                "learner": spec["learner"],
                "seed": spec["seed"],
                "selected_candidate": selected_candidate,
                "selected_is_none": bool(selected_candidate == "none"),
                "selection_score": selection_score,
                "feasibility_json": _json(selection["feasibility"]),
                "minority_score_json": _json(selection["minority_score"]),
                "constraint_diagnostics_json": _json(selection["constraint_diagnostics"]),
                "support_diagnostics_json": _json(support_diagnostics),
                "validation_hash": sha_bytes(
                    y[validation], candidate_models[selected_candidate]["validation_probability"]
                ),
                "runtime_seconds": total_runtime,
                "timestamp": decision_timestamp,
            }
        )

        for scenario, severity, declared_target_prior in scenarios:
            source_prior = candidate_models["none"]["source_prior"]
            target_prior = source_prior if declared_target_prior is None else float(declared_target_prior)
            weights = (
                None
                if scenario == "iid"
                else source_prior_weights(y[test], target_prior)
            )
            policy_models = {
                candidate: candidate_models[candidate] for candidate in CANDIDATES
            }
            policy_models["adaptive_rc"] = candidate_models[selected_candidate]
            for policy in POLICIES:
                candidate = policy if policy != "adaptive_rc" else selected_candidate
                values = compute_extended_metrics(
                    y[test], policy_models[policy]["test_probability"], sample_weight=weights
                )
                diagnostics = support_diagnostics[candidate]
                config = {
                    **candidate_models[candidate]["model_config"],
                    "policy": policy,
                    "training_candidate": candidate,
                    "selected_candidate": selected_candidate,
                    "scenario": scenario,
                    "severity": severity,
                    "target_prior": target_prior,
                    "selector": {
                        "majority_logloss_tolerance": 0.02,
                        "brier_tolerance": 0.005,
                        "target_priors": selector_priors,
                    },
                }
                config_hash = stable_hash(config)
                for metric, value in values.items():
                    rows.append(
                        {
                            "fit_id": spec["fit_id"],
                            "dataset": spec["dataset"],
                            "learner": spec["learner"],
                            "seed": spec["seed"],
                            "policy": policy,
                            "training_candidate": candidate,
                            "selected_candidate": selected_candidate,
                            "scenario": scenario,
                            "severity": severity,
                            "metric": metric,
                            "value": float(value),
                            "source_prior": source_prior,
                            "resampled_prior": candidate_models[candidate]["resampled_prior"],
                            "target_prior": target_prior,
                            "support_eligible_seed_count": diagnostics["eligible_seed_count"],
                            "support_generated_count": diagnostics["generated_count"],
                            "selection_score": selection_score,
                            "config_hash": config_hash,
                            "data_hash": data_hash,
                            "split_hash": split_hash,
                            "runtime_seconds": total_runtime,
                            "timestamp": decision_timestamp,
                        }
                    )
        changed = True
        _write_csv(pd.DataFrame(rows, columns=RAW_COLUMNS), raw_path, RAW_COLUMNS)
        _write_csv(pd.DataFrame(decision_rows, columns=DECISION_COLUMNS), decisions_path, DECISION_COLUMNS)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    decisions = pd.DataFrame(decision_rows, columns=DECISION_COLUMNS)
    if changed or not raw_path.exists():
        _write_csv(frame, raw_path, RAW_COLUMNS)
    if changed or not decisions_path.exists():
        _write_csv(decisions, decisions_path, DECISION_COLUMNS)
    state = {
        "status": "COMPLETE"
        if frame["fit_id"].nunique() == len(specs)
        and len(frame) == manifest["expected_raw_rows"]
        and len(decisions) == len(specs)
        else "INCOMPLETE",
        "fit_count": int(frame["fit_id"].nunique()),
        "expected_fit_count": len(specs),
        "candidate_fit_count": int(frame[["fit_id", "training_candidate"]].drop_duplicates().shape[0]),
        "expected_candidate_fit_count": manifest["expected_candidate_fits"],
        "raw_rows": int(len(frame)),
        "expected_raw_rows": manifest["expected_raw_rows"],
        "decision_rows": int(len(decisions)),
        "protocol_hash": protocol_hash,
        "data_manifest_hash": data_manifest_hash,
    }
    (output / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument(
        "--gate-decision",
        type=Path,
        default=None,
        help="machine-readable Gate summary required before --full",
    )
    args = parser.parse_args()
    gate_decision = None
    if args.full:
        if args.gate_decision is None or not args.gate_decision.exists():
            raise SystemExit("--full requires --gate-decision with a GO result")
        gate = json.loads(args.gate_decision.read_text(encoding="utf-8"))
        gate_decision = gate.get("decision")
        if gate_decision != "GO":
            raise SystemExit("--full is locked because the Gate decision is not GO")
    try:
        datasets, seeds = resolve_matrix(
            full=args.full,
            datasets_arg=args.datasets,
            seeds_arg=args.seeds,
            gate_decision=gate_decision,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    state = run_adaptive(args.output, datasets=datasets, seeds=seeds)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
