"""Analyze BRF against the matched XGBoost learner-family baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brf_experiment import ALL_METRICS, COST_METRIC, LEARNER as BRF_LEARNER, METHODS as BRF_METHODS


XGB_LEARNER = "xgb_gate"
GATE_TASKS = ("ecoli", "yeast_me2", "abalone_19", "mammography", "oil", "spectrometer")
PRIMARY_METRICS = (COST_METRIC, "loss_minority", "loss_majority", "brier")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260917
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10
PRACTICAL_UTILITY_TOLERANCE = 0.02


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


def _validate(frame, learner, methods):
    required = {"run_id", "dataset", "learner", "method", "seed", "metric", "value", "config_hash", "data_hash", "split_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw results missing columns: {missing}")
    if set(frame["learner"].astype(str)) != {learner}:
        raise ValueError(f"expected learner {learner}")
    if not set(frame["method"].astype(str)).issubset(set(methods)):
        raise ValueError(f"unexpected treatment for {learner}")
    if not set(ALL_METRICS).issubset(set(frame["metric"].astype(str))):
        raise ValueError(f"{learner} results require all frozen metrics")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("raw metric values must be finite")


def validate_frames(brf_frame, xgb_frame):
    if brf_frame.empty or xgb_frame.empty:
        raise ValueError("BRF and XGBoost comparator inputs are required")
    _validate(brf_frame, BRF_LEARNER, BRF_METHODS)
    xgb_subset = xgb_frame[xgb_frame["method"].isin(BRF_METHODS)]
    if xgb_subset.empty:
        raise ValueError("XGBoost comparator has no matched treatments")
    _validate(xgb_subset, XGB_LEARNER, BRF_METHODS)


def build_effects(brf_frame, xgb_frame, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    effects = []
    brf_means = brf_frame.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean()
    xgb_means = xgb_frame.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean()
    for method in BRF_METHODS:
        for metric in ALL_METRICS:
            b = brf_means[(brf_means["method"] == method) & (brf_means["metric"] == metric)].set_index("dataset")["value"]
            x = xgb_means[(xgb_means["method"] == method) & (xgb_means["metric"] == metric)].set_index("dataset")["value"]
            joined = pd.concat({"brf": b, "xgb": x}, axis=1).dropna()
            values = (joined["brf"] - joined["xgb"]).to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values, replicates=replicates, seed=seed)
            effects.append({
                "comparison": "brf_minus_xgb",
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
            })
    for method in BRF_METHODS[1:]:
        for metric in ALL_METRICS:
            subset = brf_means[brf_means["metric"] == metric].pivot(index="dataset", columns="method", values="value").dropna(subset=["none", method])
            values = (subset[method] - subset["none"]).to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values, replicates=replicates, seed=seed)
            effects.append({
                "comparison": "brf_treatment_minus_none",
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
            })
    return pd.DataFrame(effects)


def summarize_gate(brf_frame, xgb_frame, effects, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4)):
    tasks = tuple(tasks)
    seeds = tuple(seeds)
    expected_brf_fits = len(tasks) * len(BRF_METHODS) * len(seeds)
    expected_xgb_fits = len(tasks) * len(BRF_METHODS) * len(seeds)
    brf_subset = brf_frame[
        brf_frame["dataset"].isin(tasks) & brf_frame["seed"].isin(seeds)
    ]
    observed_brf_fits = int(brf_subset["run_id"].nunique())
    xgb_subset = xgb_frame[
        xgb_frame["method"].isin(BRF_METHODS)
        & xgb_frame["dataset"].isin(tasks)
        & xgb_frame["seed"].isin(seeds)
    ]
    observed_xgb_fits = int(xgb_subset["run_id"].nunique())
    expected_rows = expected_brf_fits * len(ALL_METRICS)
    integrity_pass = (
        observed_brf_fits == expected_brf_fits
        and observed_xgb_fits == expected_xgb_fits
        and len(brf_subset) == expected_rows
        and len(xgb_subset) == expected_rows
        and set(brf_frame["dataset"].astype(str)) == set(tasks)
        and set(xgb_subset["dataset"].astype(str)) == set(tasks)
    )
    candidate = effects[(effects["comparison"] == "brf_minus_xgb") & (effects["method"] == "none") & effects["metric"].isin(PRIMARY_METRICS)]
    utility = candidate[candidate["metric"] == COST_METRIC].iloc[0]
    minority = candidate[candidate["metric"] == "loss_minority"].iloc[0]
    utility_signal = bool(utility["ci_low"] > 0.0)
    conditional_signal = bool(utility["mean"] >= -PRACTICAL_UTILITY_TOLERANCE and minority["ci_high"] < 0.0)
    safety_failures = []
    for row in candidate.itertuples(index=False):
        if row.metric == "brier" and row.max_task_effect > PRACTICAL_BRIER_MARGIN:
            safety_failures.append("brf_minus_xgb/none/brier")
        if row.metric == "loss_majority" and row.max_task_effect > PRACTICAL_LOGLOSS_MARGIN:
            safety_failures.append("brf_minus_xgb/none/loss_majority")
    signal = utility_signal or conditional_signal
    decision = "GO" if integrity_pass and signal and not safety_failures else "NO-GO"
    return {
        "decision": decision,
        "expected_brf_fits": expected_brf_fits,
        "observed_brf_fits": observed_brf_fits,
        "expected_xgb_comparator_fits": expected_xgb_fits,
        "observed_xgb_comparator_fits": observed_xgb_fits,
        "expected_raw_metric_rows": expected_rows,
        "observed_brf_raw_metric_rows": int(len(brf_subset)),
        "observed_xgb_comparator_metric_rows": int(len(xgb_subset)),
        "integrity_pass": bool(integrity_pass),
        "candidate_utility_signal": bool(signal),
        "positive_utility_signal": utility_signal,
        "conditional_minority_signal": conditional_signal,
        "candidate_safety_failures": safety_failures,
        "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
        "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
        "practical_utility_tolerance": PRACTICAL_UTILITY_TOLERANCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def analyze(brf_input: Path, xgb_input: Path, output_dir: Path, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), stage="gate"):
    brf_frame = pd.read_csv(Path(brf_input) / "raw_results.csv")
    xgb_frame = pd.read_csv(Path(xgb_input) / "raw_results.csv")
    validate_frames(brf_frame, xgb_frame)
    effects = build_effects(brf_frame, xgb_frame)
    summary = summarize_gate(brf_frame, xgb_frame, effects, tasks=tasks, seeds=seeds)
    summary["stage"] = str(stage)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "effects.csv", index=False)
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Balanced Random Forest learner-family {stage} report",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- BRF fits: `{summary['observed_brf_fits']}/{summary['expected_brf_fits']}`",
        f"- Matched XGBoost fits: `{summary['observed_xgb_comparator_fits']}/{summary['expected_xgb_comparator_fits']}`",
        f"- Integrity: `{summary['integrity_pass']}`",
        f"- Candidate utility signal: `{summary['candidate_utility_signal']}`",
        f"- Candidate safety failures: `{', '.join(summary['candidate_safety_failures']) if summary['candidate_safety_failures'] else 'none'}`",
        "",
        "Effects in the first block are BRF minus matched XGBoost; the second block is BRF treatment minus BRF none.",
        "",
        "| comparison | method | metric | mean | 95% CI | tasks |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        lines.append(f"| {row.comparison} | {row.method} | {row.metric} | {row.mean:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.n_tasks} |")
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brf-input", type=Path, required=True)
    parser.add_argument("--xgb-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--stage", default="gate")
    args = parser.parse_args()
    tasks = tuple(x for x in (args.tasks or "").split(",") if x) or GATE_TASKS
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    analyze(args.brf_input, args.xgb_input, args.output, tasks=tasks, seeds=seeds, stage=args.stage)


if __name__ == "__main__":
    main()
