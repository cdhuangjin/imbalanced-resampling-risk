"""Analyze the frozen cost-sensitive versus resampling XGBoost comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cost_sensitive_xgb_experiment import ALL_METRICS, COST_METRIC, LEARNER, METHODS


GATE_TASKS = (
    "ecoli",
    "yeast_me2",
    "abalone_19",
    "mammography",
    "oil",
    "spectrometer",
)
PRIMARY_METRICS = (COST_METRIC, "loss_minority", "loss_majority", "brier")
SECONDARY_METRICS = tuple(
    metric for metric in ALL_METRICS if metric not in PRIMARY_METRICS
)
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10
PRACTICAL_UTILITY_TOLERANCE = 0.02
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260917


def paired_bootstrap_ci(values, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n_tasks": int(values.size),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def validate_raw(frame: pd.DataFrame):
    required = {
        "run_id",
        "dataset",
        "learner",
        "method",
        "seed",
        "metric",
        "value",
        "config_hash",
        "data_hash",
        "split_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw results missing columns: {missing}")
    if set(frame["learner"].astype(str)) != {LEARNER}:
        raise ValueError(f"cost-sensitive comparison must contain only {LEARNER}")
    observed_methods = set(frame["method"].astype(str))
    if not observed_methods.issubset(set(METHODS)):
        raise ValueError("unexpected treatment in raw results")
    observed_metrics = set(frame["metric"].astype(str))
    if COST_METRIC not in observed_metrics:
        raise ValueError(f"raw results require {COST_METRIC}")
    if not set(ALL_METRICS).issubset(observed_metrics):
        raise ValueError("raw results are missing one or more frozen metrics")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("raw metric values must be finite")
    for column in ("config_hash", "data_hash", "split_hash"):
        if frame[column].astype(str).eq("").any():
            raise ValueError(f"empty provenance hash in {column}")


def build_effects(frame: pd.DataFrame, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    task_means = (
        frame.groupby(["dataset", "method", "metric"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "task_mean"})
    )
    effects = []
    for method in METHODS[1:]:
        for metric in ALL_METRICS:
            subset = task_means[task_means["metric"].eq(metric)]
            pivot = subset.pivot(index="dataset", columns="method", values="task_mean")
            if not {"none", method}.issubset(pivot.columns):
                raise ValueError(f"incomplete task-method coverage for {method}/{metric}")
            pivot = pivot.dropna(subset=["none", method])
            values = (pivot[method] - pivot["none"]).to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values, replicates=replicates, seed=seed)
            effects.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_tasks": ci["n_tasks"],
                    "mean": ci["mean"],
                    "ci_low": ci["low"],
                    "ci_high": ci["high"],
                    "min_task_effect": float(values.min()),
                    "max_task_effect": float(values.max()),
                    "positive_task_count": int((values > 0).sum()),
                    "negative_task_count": int((values < 0).sum()),
                    "bootstrap_replicates": ci["replicates"],
                    "bootstrap_seed": ci["seed"],
                }
            )
    return pd.DataFrame(effects)


def summarize_gate(frame, effects, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4)):
    tasks = tuple(tasks)
    seeds = tuple(seeds)
    expected_fits = len(tasks) * len(METHODS) * len(seeds)
    observed_fits = int(frame["run_id"].nunique())
    expected_rows = expected_fits * len(ALL_METRICS)
    integrity_pass = (
        observed_fits == expected_fits
        and len(frame) == expected_rows
        and set(frame["dataset"].astype(str)) == set(tasks)
        and set(frame["method"].astype(str)) == set(METHODS)
        and set(frame["metric"].astype(str)) == set(ALL_METRICS)
    )
    primary = effects[effects["metric"].isin(PRIMARY_METRICS)]
    candidate = primary[primary["method"].eq("cost_weight_q05")]
    all_safety_failures = []
    candidate_safety_failures = []
    for row in primary.itertuples(index=False):
        failed = False
        if row.metric == "brier" and row.max_task_effect > PRACTICAL_BRIER_MARGIN:
            failed = True
        if row.metric == "loss_majority" and row.max_task_effect > PRACTICAL_LOGLOSS_MARGIN:
            failed = True
        if failed:
            all_safety_failures.append(f"{row.method}/{row.metric}")
            if row.method == "cost_weight_q05":
                candidate_safety_failures.append(f"{row.method}/{row.metric}")
    utility_row = candidate[candidate["metric"].eq(COST_METRIC)]
    minority_row = candidate[candidate["metric"].eq("loss_minority")]
    if utility_row.empty or minority_row.empty:
        raise ValueError("candidate primary effects are incomplete")
    utility = utility_row.iloc[0]
    minority = minority_row.iloc[0]
    positive_utility_signal = bool(utility["ci_low"] > 0.0)
    conditional_minority_signal = bool(
        utility["mean"] >= -PRACTICAL_UTILITY_TOLERANCE and minority["ci_high"] < 0.0
    )
    candidate_utility_signal = positive_utility_signal or conditional_minority_signal
    decision = (
        "GO"
        if integrity_pass and not candidate_safety_failures and candidate_utility_signal
        else "NO-GO"
    )
    return {
        "decision": decision,
        "expected_fits": expected_fits,
        "observed_fits": observed_fits,
        "expected_raw_metric_rows": expected_rows,
        "observed_raw_metric_rows": int(len(frame)),
        "integrity_pass": bool(integrity_pass),
        "candidate_utility_signal": bool(candidate_utility_signal),
        "positive_utility_signal": positive_utility_signal,
        "conditional_minority_signal": conditional_minority_signal,
        "candidate_safety_failures": candidate_safety_failures,
        "all_safety_failures": all_safety_failures,
        "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
        "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
        "practical_utility_tolerance": PRACTICAL_UTILITY_TOLERANCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def analyze(input_dir: Path, output_dir: Path, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), stage="gate"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    frame = pd.read_csv(input_dir / "raw_results.csv")
    validate_raw(frame)
    effects = build_effects(frame)
    summary = summarize_gate(frame, effects, tasks=tasks, seeds=seeds)
    summary["stage"] = str(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "effects.csv", index=False)
    (output_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Cost-sensitive XGBoost {stage} report",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Fits: `{summary['observed_fits']}/{summary['expected_fits']}`",
        f"- Raw metric rows: `{summary['observed_raw_metric_rows']}/{summary['expected_raw_metric_rows']}`",
        f"- Integrity: `{summary['integrity_pass']}`",
        f"- Candidate utility signal: `{summary['candidate_utility_signal']}`",
        f"- Candidate safety failures: `{', '.join(summary['candidate_safety_failures']) if summary['candidate_safety_failures'] else 'none'}`",
        f"- All treatment safety failures: `{', '.join(summary['all_safety_failures']) if summary['all_safety_failures'] else 'none'}`",
        "",
        "Effects are treatment minus `none`; inference uses task-level paired bootstrap.",
        "",
        "| method | metric | mean | 95% CI | tasks |",
        "|---|---|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.metric} | {row.mean:.6f} | "
            f"[{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.n_tasks} |"
        )
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--stage", default="gate")
    args = parser.parse_args()
    tasks = tuple(x for x in (args.tasks or "").split(",") if x) or GATE_TASKS
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    analyze(args.input, args.output, tasks=tasks, seeds=seeds, stage=args.stage)


if __name__ == "__main__":
    main()
