"""Run the confidence-guarded adaptive policy as a separate frozen branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

import run_adaptive as base_runner  # noqa: E402
from confidence_guarded import (  # noqa: E402
    attach_raw_metrics,
    confidence_guarded_select,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output: Path):
    output = Path(output)
    original_metrics = base_runner.validation_policy_metrics

    def metrics_with_raw(y, probability, *, target_priors):
        metrics = original_metrics(y, probability, target_priors=target_priors)
        return attach_raw_metrics(metrics, y=np.asarray(y), probability=np.asarray(probability))

    base_runner.validation_policy_metrics = metrics_with_raw
    base_runner.select_risk_constrained_candidate = confidence_guarded_select
    state = base_runner.run_adaptive(
        output,
        datasets=base_runner.GATE_TASKS,
        learners=base_runner.LEARNERS,
        seeds=base_runner.GATE_SEEDS,
    )
    base_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    protocol = {
        "name": "confidence_guarded_rc",
        "base_runner": str((PACKAGE_DIR / "run_adaptive.py").resolve()),
        "base_runner_sha256": file_sha256(PACKAGE_DIR / "run_adaptive.py"),
        "selector": str((PACKAGE_DIR / "confidence_guarded.py").resolve()),
        "selector_sha256": file_sha256(PACKAGE_DIR / "confidence_guarded.py"),
        "wrapper_sha256": file_sha256(Path(__file__)),
        "majority_logloss_tolerance": 0.02,
        "brier_tolerance": 0.005,
        "minority_required_improvement": 0.0,
        "validation_bootstrap_replicates": 400,
        "validation_bootstrap_seed": 20260917,
        "gate_tasks": list(base_runner.GATE_TASKS),
        "gate_learners": list(base_runner.LEARNERS),
        "gate_seeds": list(base_runner.GATE_SEEDS),
        "base_manifest_protocol_hash": base_manifest["protocol_hash"],
        "base_manifest_data_manifest_hash": base_manifest["data_manifest_hash"],
    }
    (output / "confidence_guarded_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
