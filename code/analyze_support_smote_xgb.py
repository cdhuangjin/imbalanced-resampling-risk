"""Analyze the support-filtered SMOTE policy Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from support_smote_xgb_experiment import ALL_METRICS, COST_METRIC, LEARNER, METHODS


GATE_TASKS = ("ecoli", "yeast_me2", "abalone_19", "mammography", "oil", "spectrometer")
PRIMARY_METRICS = (COST_METRIC, "loss_minority", "loss_majority", "brier")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260917
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10
PRACTICAL_UTILITY_TOLERANCE = 0.02


def paired_bootstrap_ci(values, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all(): raise ValueError("values must be a non-empty finite one-dimensional array")
    rng = np.random.default_rng(seed); indices = rng.integers(0, values.size, size=(int(replicates), values.size)); means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)), "n_tasks": int(values.size), "replicates": int(replicates), "seed": int(seed)}


def validate_raw(frame):
    required = {"run_id", "dataset", "learner", "method", "seed", "metric", "value", "config_hash", "data_hash", "split_hash"}; missing = sorted(required - set(frame.columns))
    if missing: raise ValueError(f"raw results missing columns: {missing}")
    if set(frame["learner"].astype(str)) != {LEARNER}: raise ValueError(f"expected learner {LEARNER}")
    if not set(frame["method"].astype(str)).issubset(set(METHODS)): raise ValueError("unexpected treatment")
    if not set(ALL_METRICS).issubset(set(frame["metric"].astype(str))): raise ValueError("raw results require all frozen metrics including decision_utility_c5")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all(): raise ValueError("raw metric values must be finite")
    for field in ("config_hash", "data_hash", "split_hash"):
        if frame[field].astype(str).eq("").any(): raise ValueError(f"empty provenance hash in {field}")


def build_effects(frame):
    means = frame.groupby(["dataset", "method", "metric"], as_index=False)["value"].mean(); effects = []
    for method in METHODS[1:]:
        for metric in ALL_METRICS:
            pivot = means[means["metric"] == metric].pivot(index="dataset", columns="method", values="value").dropna(subset=["none", method]); values = (pivot[method] - pivot["none"]).to_numpy(dtype=float); ci = paired_bootstrap_ci(values)
            effects.append({"comparison": "treatment_minus_none", "method": method, "metric": metric, "n_tasks": ci["n_tasks"], "mean": ci["mean"], "ci_low": ci["low"], "ci_high": ci["high"], "min_task_effect": float(values.min()), "max_task_effect": float(values.max()), "positive_task_count": int((values > 0).sum()), "negative_task_count": int((values < 0).sum()), "bootstrap_replicates": ci["replicates"], "bootstrap_seed": ci["seed"]})
    candidate = means[means["method"].isin(("support_smote_q05", "smote_q05_k5"))]
    for metric in ALL_METRICS:
        pivot = candidate[candidate["metric"] == metric].pivot(index="dataset", columns="method", values="value").dropna(subset=["smote_q05_k5", "support_smote_q05"]); values = (pivot["support_smote_q05"] - pivot["smote_q05_k5"]).to_numpy(dtype=float); ci = paired_bootstrap_ci(values)
        effects.append({"comparison": "support_smote_minus_smote", "method": "support_smote_q05", "metric": metric, "n_tasks": ci["n_tasks"], "mean": ci["mean"], "ci_low": ci["low"], "ci_high": ci["high"], "min_task_effect": float(values.min()), "max_task_effect": float(values.max()), "positive_task_count": int((values > 0).sum()), "negative_task_count": int((values < 0).sum()), "bootstrap_replicates": ci["replicates"], "bootstrap_seed": ci["seed"]})
    return pd.DataFrame(effects)


def summarize_gate(frame, effects, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), diagnostics=None):
    tasks = tuple(tasks); seeds = tuple(seeds); expected = len(tasks) * len(METHODS) * len(seeds); subset = frame[frame["dataset"].isin(tasks) & frame["seed"].isin(seeds)]
    integrity = int(subset["run_id"].nunique()) == expected and len(subset) == expected * len(ALL_METRICS) and set(subset["method"].astype(str)) == set(METHODS)
    candidate = effects[(effects["comparison"] == "treatment_minus_none") & (effects["method"] == "support_smote_q05") & effects["metric"].isin(PRIMARY_METRICS)]
    utility = candidate[candidate["metric"] == COST_METRIC].iloc[0]; minority = candidate[candidate["metric"] == "loss_minority"].iloc[0]
    positive = bool(utility["ci_low"] > 0); conditional = bool(utility["mean"] >= -PRACTICAL_UTILITY_TOLERANCE and minority["ci_high"] < 0)
    safety = []
    for row in candidate.itertuples(index=False):
        if row.metric == "brier" and row.max_task_effect > PRACTICAL_BRIER_MARGIN: safety.append("support_smote_q05/brier")
        if row.metric == "loss_majority" and row.max_task_effect > PRACTICAL_LOGLOSS_MARGIN: safety.append("support_smote_q05/loss_majority")
    diagnostic_summary = {}
    if diagnostics is not None and not diagnostics.empty:
        d = diagnostics[diagnostics["run_id"].isin(subset["run_id"].unique())]
        diagnostic_summary = {"fallback_rate": float(d["fallback"].mean()), "mean_safe_fraction": float(d["safe_fraction"].mean()), "mean_synthetic_rows": float(d["n_synthetic"].mean())}
    return {"decision": "GO" if integrity and (positive or conditional) and not safety else "NO-GO", "expected_fits": expected, "observed_fits": int(subset["run_id"].nunique()), "expected_raw_metric_rows": expected * len(ALL_METRICS), "observed_raw_metric_rows": int(len(subset)), "integrity_pass": bool(integrity), "candidate_utility_signal": bool(positive or conditional), "positive_utility_signal": positive, "conditional_minority_signal": conditional, "candidate_safety_failures": safety, "practical_brier_margin": PRACTICAL_BRIER_MARGIN, "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN, "practical_utility_tolerance": PRACTICAL_UTILITY_TOLERANCE, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "diagnostics": diagnostic_summary}


def analyze(input_dir: Path, output_dir: Path, *, tasks=GATE_TASKS, seeds=(0, 1, 2, 3, 4), stage="gate"):
    input_dir = Path(input_dir); frame = pd.read_csv(input_dir / "raw_results.csv"); validate_raw(frame); diagnostics = pd.read_csv(input_dir / "support_diagnostics.csv") if (input_dir / "support_diagnostics.csv").exists() else None; effects = build_effects(frame); summary = summarize_gate(frame, effects, tasks=tasks, seeds=seeds, diagnostics=diagnostics); summary["stage"] = str(stage); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); effects.to_csv(output_dir / "effects.csv", index=False); (output_dir / "gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Support-filtered SMOTE XGBoost {stage} report", "", f"- Decision: `{summary['decision']}`", f"- Fits: `{summary['observed_fits']}/{summary['expected_fits']}`", f"- Raw metric rows: `{summary['observed_raw_metric_rows']}/{summary['expected_raw_metric_rows']}`", f"- Integrity: `{summary['integrity_pass']}`", f"- Candidate utility signal: `{summary['candidate_utility_signal']}`", f"- Candidate safety failures: `{', '.join(summary['candidate_safety_failures']) if summary['candidate_safety_failures'] else 'none'}`", f"- Diagnostics: `{summary['diagnostics']}`", "", "| comparison | method | metric | mean | 95% CI | tasks |", "|---|---|---|---:|---:|---:|"]
    for row in effects.itertuples(index=False): lines.append(f"| {row.comparison} | {row.method} | {row.metric} | {row.mean:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.n_tasks} |")
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); return summary, effects


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--tasks", default=None); parser.add_argument("--seeds", default="0,1,2,3,4"); parser.add_argument("--stage", default="gate"); args = parser.parse_args(); tasks = tuple(x for x in (args.tasks or "").split(",") if x) or GATE_TASKS; seeds = tuple(int(x) for x in args.seeds.split(",") if x); analyze(args.input, args.output, tasks=tasks, seeds=seeds, stage=args.stage)


if __name__ == "__main__": main()
