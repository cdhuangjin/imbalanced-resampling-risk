"""Analyze the held-out XGBoost learner Gate at the task level."""

from __future__ import annotations

import argparse
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
METHODS = ("none", "random_q05", "smote_q05_k5")
PRIMARY_METRICS = ("auc", "loss_minority", "loss_majority", "brier")
SECONDARY_METRICS = ("ap", "logloss", "f1_fixed", "f1_tuned")
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10
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


def _validate_raw(frame: pd.DataFrame):
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
    if set(frame["learner"].astype(str)) != {"xgb_gate"}:
        raise ValueError("held-out Gate must contain only xgb_gate")
    if not set(frame["method"].astype(str)).issubset(set(METHODS)):
        raise ValueError("unexpected treatment in raw results")
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
        for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
            subset = task_means[task_means["metric"].eq(metric)]
            pivot = subset.pivot(index="dataset", columns="method", values="task_mean")
            if not set(METHODS[:2]).issubset(pivot.columns) or method not in pivot.columns:
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


def summarize_gate(frame: pd.DataFrame, effects: pd.DataFrame, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4)):
    tasks = tuple(tasks)
    seeds = tuple(seeds)
    expected_fits = len(tasks) * 1 * len(METHODS) * len(seeds)
    observed_fits = int(frame["run_id"].nunique())
    expected_rows = expected_fits * 8
    integrity_pass = (
        observed_fits == expected_fits
        and len(frame) == expected_rows
        and set(frame["dataset"].astype(str)) == set(tasks)
        and set(frame["method"].astype(str)) == set(METHODS)
        and set(frame["metric"].astype(str))
        == set((*PRIMARY_METRICS, *SECONDARY_METRICS))
    )
    primary = effects[effects["metric"].isin(PRIMARY_METRICS)]
    safety_failures = []
    for row in primary.itertuples(index=False):
        if row.metric == "brier" and row.max_task_effect > PRACTICAL_BRIER_MARGIN:
            safety_failures.append(f"{row.method}/{row.metric}")
        if row.metric == "loss_majority" and row.max_task_effect > PRACTICAL_LOGLOSS_MARGIN:
            safety_failures.append(f"{row.method}/{row.metric}")
    utility_signal = bool(
        ((primary["ci_low"] > 0) | (primary["ci_high"] < 0)).any()
    )
    decision = "GO" if integrity_pass and not safety_failures and utility_signal else "NO-GO"
    return {
        "decision": decision,
        "expected_fits": expected_fits,
        "observed_fits": observed_fits,
        "expected_raw_metric_rows": expected_rows,
        "observed_raw_metric_rows": int(len(frame)),
        "integrity_pass": bool(integrity_pass),
        "utility_signal": utility_signal,
        "safety_failures": safety_failures,
        "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
        "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def analyze(input_dir: Path, output_dir: Path, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), stage="gate"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    frame = pd.read_csv(input_dir / "raw_results.csv")
    _validate_raw(frame)
    effects = build_effects(frame)
    gate = summarize_gate(frame, effects, tasks=tasks, seeds=seeds)
    gate["stage"] = str(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "effects.csv", index=False)
    (output_dir / "gate_summary.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Held-out XGBoost learner {stage} report",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Fits: `{gate['observed_fits']}/{gate['expected_fits']}`",
        f"- Raw metric rows: `{gate['observed_raw_metric_rows']}/{gate['expected_raw_metric_rows']}`",
        f"- Integrity: `{gate['integrity_pass']}`",
        f"- Utility signal: `{gate['utility_signal']}`",
        f"- Safety failures: `{', '.join(gate['safety_failures']) if gate['safety_failures'] else 'none'}`",
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
    return gate, effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--stage", type=str, default="gate")
    args = parser.parse_args()
    tasks = tuple(x for x in (args.tasks or "").split(",") if x) or GATE_TASKS
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    analyze(args.input, args.output, tasks=tasks, seeds=seeds, stage=args.stage)


if __name__ == "__main__":
    main()
