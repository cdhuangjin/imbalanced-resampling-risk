"""Task-cluster analysis and frozen Gate decision for adaptive resampling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


GATE_TASKS = (
    "ecoli",
    "yeast_me2",
    "abalone_19",
    "mammography",
    "oil",
    "spectrometer",
)
POLICIES = (
    "none",
    "random_q05",
    "smote_q05_k5",
    "support_smote_q05_k5",
    "adaptive_rc",
)
METRICS = ("auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10")


def cluster_bootstrap_mean(values, n_boot=10000, seed=20260916):
    """Return a mean and percentile interval after resampling task clusters."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    if int(n_boot) < 100:
        raise ValueError("n_boot must be at least 100")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def task_selection_rates(decisions: pd.DataFrame, expected_tasks):
    """Return per-task non-baseline selection rates or None on coverage mismatch."""
    required = {"dataset", "selected_candidate"}
    if not required.issubset(decisions.columns):
        return None
    observed = set(decisions["dataset"].astype(str).unique())
    expected = set(map(str, expected_tasks))
    if observed != expected:
        return None
    selected = decisions["selected_candidate"].astype(str) != "none"
    rates = selected.groupby(decisions["dataset"].astype(str)).mean()
    return {task: float(rates[task]) for task in sorted(expected)}


def gate_decision(
    *,
    complete,
    selection_task_count,
    adaptive_smote_majority_logloss_pass,
    adaptive_smote_brier_pass,
    minority_task_count,
    minority_ci_pass,
):
    """Apply the immutable protocol rule."""
    if not complete:
        return "INCONCLUSIVE"
    if (
        int(selection_task_count) >= 4
        and bool(adaptive_smote_majority_logloss_pass)
        and bool(adaptive_smote_brier_pass)
        and int(minority_task_count) >= 4
        and bool(minority_ci_pass)
    ):
        return "GO"
    return "NO-GO"


def _validate_raw(frame: pd.DataFrame):
    required = {
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
        "data_hash",
        "split_hash",
        "config_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing adaptive raw columns: " + ", ".join(missing))
    if not frame["value"].map(np.isfinite).all():
        raise ValueError("adaptive raw values must be finite")
    if not frame["policy"].isin(POLICIES).all():
        raise ValueError("adaptive raw contains unknown policies")
    if not frame["metric"].isin(METRICS).all():
        raise ValueError("adaptive raw contains unknown metrics")
    key = ["fit_id", "policy", "scenario", "severity", "metric"]
    if frame.duplicated(key).any():
        raise ValueError("adaptive raw contains duplicate metric keys")


def _paired_task_delta(frame, *, scenario, severity, candidate, reference, metric):
    subset = frame[
        (frame["scenario"] == scenario)
        & (frame["severity"].astype(int) == int(severity))
        & (frame["policy"].isin([candidate, reference]))
        & (frame["metric"] == metric)
    ]
    pivot = subset.pivot_table(
        index=["dataset", "learner", "seed"],
        columns="policy",
        values="value",
        aggfunc="first",
    )
    if candidate not in pivot or reference not in pivot:
        raise ValueError(f"missing paired values for {candidate} and {reference}")
    delta = (pivot[candidate] - pivot[reference]).dropna()
    if delta.empty:
        raise ValueError("paired comparison has no complete rows")
    return delta.groupby(level="dataset").mean()


def summarize_comparison(
    frame,
    *,
    scenario,
    severity,
    candidate,
    reference,
    metric,
    seed=20260916,
):
    task_delta = _paired_task_delta(
        frame,
        scenario=scenario,
        severity=severity,
        candidate=candidate,
        reference=reference,
        metric=metric,
    )
    mean, low, high = cluster_bootstrap_mean(task_delta.to_numpy(), seed=seed)
    return {
        "scenario": scenario,
        "severity": int(severity),
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "mean_delta": mean,
        "ci_low": low,
        "ci_high": high,
        "n_tasks": int(len(task_delta)),
        "task_deltas": {str(index): float(value) for index, value in task_delta.items()},
    }


def _relative_minority_deltas(frame):
    subset = frame[
        (frame["scenario"] == "iid")
        & (frame["severity"].astype(int) == 0)
        & (frame["policy"].isin(["adaptive_rc", "none"]))
        & (frame["metric"] == "loss_minority")
    ]
    pivot = subset.pivot_table(
        index=["dataset", "learner", "seed"],
        columns="policy",
        values="value",
        aggfunc="first",
    )
    if "adaptive_rc" not in pivot or "none" not in pivot:
        raise ValueError("missing paired minority-loss values")
    relative = ((pivot["adaptive_rc"] - pivot["none"]) / pivot["none"]).dropna()
    return relative.groupby(level="dataset").mean()


def _complete_from_artifacts(frame, decisions, input_dir, expected_tasks):
    status_path = Path(input_dir) / "status.json"
    manifest_path = Path(input_dir) / "manifest.json"
    if not status_path.exists() or not manifest_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        status.get("status") == "COMPLETE"
        and int(status.get("raw_rows", -1)) == int(status.get("expected_raw_rows", -2))
        and int(status.get("decision_rows", -1)) == int(status.get("expected_fit_count", -2))
        and frame["fit_id"].nunique() == int(status.get("expected_fit_count", -3))
        and decisions["fit_id"].nunique() == int(status.get("expected_fit_count", -4))
        and set(frame["dataset"].astype(str).unique()) == set(map(str, expected_tasks))
        and set(decisions["dataset"].astype(str).unique()) == set(map(str, expected_tasks))
        and frame["data_hash"].astype(str).str.len().gt(0).all()
        and frame["split_hash"].astype(str).str.len().gt(0).all()
        and frame["config_hash"].astype(str).str.len().gt(0).all()
        and manifest.get("protocol_hash") == status.get("protocol_hash")
        and manifest.get("data_manifest_hash") == status.get("data_manifest_hash")
    )


def analyze_gate(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_dir / "raw_results.csv")
    decisions = pd.read_csv(input_dir / "selection_decisions.csv")
    _validate_raw(frame)

    complete = _complete_from_artifacts(frame, decisions, input_dir, GATE_TASKS)
    rates = task_selection_rates(decisions, GATE_TASKS)
    selection_task_count = (
        sum(rate >= 0.5 for rate in rates.values()) if rates is not None else 0
    )
    summaries = [
        summarize_comparison(
            frame,
            scenario="iid",
            severity=0,
            candidate="adaptive_rc",
            reference="smote_q05_k5",
            metric="loss_majority",
            seed=20260916,
        ),
        summarize_comparison(
            frame,
            scenario="iid",
            severity=0,
            candidate="adaptive_rc",
            reference="smote_q05_k5",
            metric="brier",
            seed=20260917,
        ),
        summarize_comparison(
            frame,
            scenario="iid",
            severity=0,
            candidate="adaptive_rc",
            reference="none",
            metric="loss_minority",
            seed=20260918,
        ),
    ]
    relative = _relative_minority_deltas(frame)
    minority_task_count = int(np.sum(relative.to_numpy() <= -0.05))
    minority_summary = summaries[2]
    majority_summary = summaries[0]
    brier_summary = summaries[1]
    conditions = {
        "selection_task_count": int(selection_task_count),
        "selection_rates": rates,
        "adaptive_smote_majority_logloss_pass": bool(majority_summary["ci_high"] <= 0.0),
        "adaptive_smote_majority_logloss_ci_high": float(majority_summary["ci_high"]),
        "adaptive_smote_brier_pass": bool(brier_summary["ci_high"] <= 0.002),
        "adaptive_smote_brier_ci_high": float(brier_summary["ci_high"]),
        "minority_task_count": minority_task_count,
        "minority_relative_task_deltas": {
            str(index): float(value) for index, value in relative.items()
        },
        "minority_ci_pass": bool(minority_summary["ci_high"] < 0.0),
        "minority_delta_ci_high": float(minority_summary["ci_high"]),
    }
    decision = gate_decision(
        complete=complete,
        selection_task_count=selection_task_count,
        adaptive_smote_majority_logloss_pass=conditions["adaptive_smote_majority_logloss_pass"],
        adaptive_smote_brier_pass=conditions["adaptive_smote_brier_pass"],
        minority_task_count=minority_task_count,
        minority_ci_pass=conditions["minority_ci_pass"],
    )
    payload = {
        "decision": decision,
        "complete": bool(complete),
        "conditions": conditions,
        "summaries": summaries,
        "input": str(input_dir.resolve()),
        "bootstrap_n": 10000,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "gate_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "task_deltas"} for row in summaries]
    ).to_csv(output_dir / "gate_effects.csv", index=False)
    _write_gate_report(payload, output_dir / "gate_report.md")
    return payload


def _write_gate_report(payload, path):
    conditions = payload["conditions"]
    lines = [
        "# Adaptive Resampling Gate Report",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        f"Completeness: `{payload['complete']}`; task clusters with non-baseline selection rate at least 0.5: `{conditions['selection_task_count']}/6`.",
        "",
        "## Gate Conditions",
        "",
        "| Condition | Result |",
        "|---|---:|",
        f"| Adaptive vs fixed SMOTE majority-logloss upper bound <= 0 | `{conditions['adaptive_smote_majority_logloss_ci_high']:.6g}`; `{conditions['adaptive_smote_majority_logloss_pass']}` |",
        f"| Adaptive vs fixed SMOTE Brier upper bound <= 0.002 | `{conditions['adaptive_smote_brier_ci_high']:.6g}`; `{conditions['adaptive_smote_brier_pass']}` |",
        f"| Adaptive improves minority logloss by >=5% in task clusters | `{conditions['minority_task_count']}/6` |",
        f"| Adaptive vs none minority-logloss upper bound < 0 | `{conditions['minority_delta_ci_high']:.6g}`; `{conditions['minority_ci_pass']}` |",
        "",
        "The selector was frozen before the Gate and uses validation predictions only. A non-GO result does not unlock the 24-task extension and does not justify changing the support threshold, risk tolerances, candidates, or task list.",
        "",
        "## Selection Rates",
        "",
    ]
    for task, rate in sorted((conditions["selection_rates"] or {}).items()):
        lines.append(f"- `{task}`: `{rate:.3f}` non-baseline selections")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_gate(args.input, args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

