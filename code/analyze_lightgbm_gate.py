"""Analyze LightGBM against matched XGBoost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brf_experiment import ALL_METRICS, COST_METRIC, METHODS


LEARNER = "lgbm_gate"
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
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)), "n_tasks": int(values.size), "replicates": int(replicates), "seed": int(seed)}


def _validate(frame, learner):
    required = {"run_id", "dataset", "learner", "method", "seed", "metric", "value", "config_hash", "data_hash", "split_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw results missing columns: {missing}")
    if set(frame["learner"].astype(str)) != {learner}:
        raise ValueError(f"expected learner {learner}")
    if not set(frame["method"].astype(str)).issubset(set(METHODS)):
        raise ValueError(f"unexpected treatment for {learner}")
    if not set(ALL_METRICS).issubset(set(frame["metric"].astype(str))):
        raise ValueError(f"{learner} results require all frozen metrics")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("raw metric values must be finite")


def validate_frames(lgbm_frame, xgb_frame):
    if lgbm_frame.empty or xgb_frame.empty:
        raise ValueError("LightGBM and XGBoost comparator inputs are required")
    _validate(lgbm_frame, LEARNER)
    xgb_subset = xgb_frame[xgb_frame["method"].isin(METHODS)]
    if xgb_subset.empty:
        raise ValueError("XGBoost comparator has no matched treatments")
    _validate(xgb_subset, XGB_LEARNER)


def _effect_rows(left, right, *, comparison, methods):
    left_means = left.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean()
    right_means = right.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean()
    rows = []
    for method in methods:
        for metric in ALL_METRICS:
            l = left_means[(left_means["method"] == method) & (left_means["metric"] == metric)].set_index("dataset")["value"]
            r = right_means[(right_means["method"] == method) & (right_means["metric"] == metric)].set_index("dataset")["value"]
            joined = pd.concat({"left": l, "right": r}, axis=1).dropna()
            values = (joined["left"] - joined["right"]).to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values)
            rows.append({"comparison": comparison, "method": method, "metric": metric, "n_tasks": ci["n_tasks"], "mean": ci["mean"], "ci_low": ci["low"], "ci_high": ci["high"], "min_task_effect": float(values.min()), "max_task_effect": float(values.max()), "positive_task_count": int((values > 0).sum()), "negative_task_count": int((values < 0).sum()), "bootstrap_replicates": ci["replicates"], "bootstrap_seed": ci["seed"]})
    return rows


def build_effects(lgbm_frame, xgb_frame):
    rows = _effect_rows(lgbm_frame, xgb_frame[xgb_frame["method"].isin(METHODS)], comparison="lgbm_minus_xgb", methods=METHODS)
    lgbm_means = lgbm_frame.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean()
    for method in METHODS[1:]:
        for metric in ALL_METRICS:
            pivot = lgbm_means[lgbm_means["metric"] == metric].pivot(index="dataset", columns="method", values="value").dropna(subset=["none", method])
            values = (pivot[method] - pivot["none"]).to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values)
            rows.append({"comparison": "lgbm_treatment_minus_none", "method": method, "metric": metric, "n_tasks": ci["n_tasks"], "mean": ci["mean"], "ci_low": ci["low"], "ci_high": ci["high"], "min_task_effect": float(values.min()), "max_task_effect": float(values.max()), "positive_task_count": int((values > 0).sum()), "negative_task_count": int((values < 0).sum()), "bootstrap_replicates": ci["replicates"], "bootstrap_seed": ci["seed"]})
    return pd.DataFrame(rows)


def summarize_gate(lgbm_frame, xgb_frame, effects, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4)):
    tasks = tuple(tasks); seeds = tuple(seeds)
    expected = len(tasks) * len(METHODS) * len(seeds)
    lgbm_subset = lgbm_frame[lgbm_frame["dataset"].isin(tasks) & lgbm_frame["seed"].isin(seeds)]
    xgb_subset = xgb_frame[xgb_frame["method"].isin(METHODS) & xgb_frame["dataset"].isin(tasks) & xgb_frame["seed"].isin(seeds)]
    integrity = int(lgbm_subset["run_id"].nunique()) == expected and int(xgb_subset["run_id"].nunique()) == expected and len(lgbm_subset) == expected * len(ALL_METRICS) and len(xgb_subset) == expected * len(ALL_METRICS)
    candidate = effects[(effects["comparison"] == "lgbm_minus_xgb") & (effects["method"] == "none") & effects["metric"].isin(PRIMARY_METRICS)]
    utility = candidate[candidate["metric"] == COST_METRIC].iloc[0]; minority = candidate[candidate["metric"] == "loss_minority"].iloc[0]
    positive = bool(utility["ci_low"] > 0.0)
    conditional = bool(utility["mean"] >= -PRACTICAL_UTILITY_TOLERANCE and minority["ci_high"] < 0.0)
    safety = []
    for row in candidate.itertuples(index=False):
        if row.metric == "brier" and row.max_task_effect > PRACTICAL_BRIER_MARGIN: safety.append("lgbm_minus_xgb/none/brier")
        if row.metric == "loss_majority" and row.max_task_effect > PRACTICAL_LOGLOSS_MARGIN: safety.append("lgbm_minus_xgb/none/loss_majority")
    return {"decision": "GO" if integrity and (positive or conditional) and not safety else "NO-GO", "expected_fits": expected, "observed_lgbm_fits": int(lgbm_subset["run_id"].nunique()), "observed_xgb_comparator_fits": int(xgb_subset["run_id"].nunique()), "expected_raw_metric_rows": expected * len(ALL_METRICS), "observed_lgbm_raw_metric_rows": int(len(lgbm_subset)), "observed_xgb_comparator_metric_rows": int(len(xgb_subset)), "integrity_pass": bool(integrity), "candidate_utility_signal": bool(positive or conditional), "positive_utility_signal": positive, "conditional_minority_signal": conditional, "candidate_safety_failures": safety, "practical_brier_margin": PRACTICAL_BRIER_MARGIN, "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN, "practical_utility_tolerance": PRACTICAL_UTILITY_TOLERANCE, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED}


def analyze(lgbm_input: Path, xgb_input: Path, output_dir: Path, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), stage="gate"):
    lgbm_frame = pd.read_csv(Path(lgbm_input) / "raw_results.csv")
    xgb_frame = pd.read_csv(Path(xgb_input) / "raw_results.csv")
    validate_frames(lgbm_frame, xgb_frame)
    effects = build_effects(lgbm_frame, xgb_frame)
    summary = summarize_gate(lgbm_frame, xgb_frame, effects, tasks=tasks, seeds=seeds); summary["stage"] = str(stage)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "effects.csv", index=False)
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# LightGBM learner-family {stage} report", "", f"- Decision: `{summary['decision']}`", f"- LightGBM fits: `{summary['observed_lgbm_fits']}/{summary['expected_fits']}`", f"- Matched XGBoost fits: `{summary['observed_xgb_comparator_fits']}/{summary['expected_fits']}`", f"- Integrity: `{summary['integrity_pass']}`", f"- Candidate utility signal: `{summary['candidate_utility_signal']}`", f"- Candidate safety failures: `{', '.join(summary['candidate_safety_failures']) if summary['candidate_safety_failures'] else 'none'}`", "", "Effects are LightGBM minus matched XGBoost; treatment interactions follow.", "", "| comparison | method | metric | mean | 95% CI | tasks |", "|---|---|---|---:|---:|---:|"]
    for row in effects.itertuples(index=False): lines.append(f"| {row.comparison} | {row.method} | {row.metric} | {row.mean:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.n_tasks} |")
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, effects


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--lgbm-input", type=Path, required=True); parser.add_argument("--xgb-input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--tasks", default=None); parser.add_argument("--seeds", default="0,1,2,3,4"); parser.add_argument("--stage", default="gate")
    args = parser.parse_args(); tasks = tuple(x for x in (args.tasks or "").split(",") if x) or GATE_TASKS; seeds = tuple(int(x) for x in args.seeds.split(",") if x); analyze(args.lgbm_input, args.xgb_input, args.output, tasks=tasks, seeds=seeds, stage=args.stage)


if __name__ == "__main__":
    main()
