"""Analyze the utility-first selective-treatment Gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TASKS = ("ecoli", "yeast_me2", "abalone_19", "mammography", "oil", "spectrometer")
EXPECTED_FITS = 36
EXPECTED_ROWS = 5040


def bootstrap(values, *, seed, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)), "n_tasks": int(values.size)}


def task_delta(frame, candidate, reference, metric):
    subset = frame[(frame["scenario"] == "iid") & (frame["severity"].astype(int) == 0) & (frame["policy"].isin([candidate, reference])) & (frame["metric"] == metric)]
    pivot = subset.pivot_table(index=["dataset", "learner", "seed"], columns="policy", values="value", aggfunc="first")
    if candidate not in pivot or reference not in pivot:
        raise ValueError(f"missing paired metric {candidate} vs {reference} for {metric}")
    return (pivot[candidate] - pivot[reference]).dropna().groupby(level="dataset").mean()


def summarize(frame, candidate, reference, metric, seed):
    values = task_delta(frame, candidate, reference, metric)
    return {"candidate": candidate, "reference": reference, "metric": metric, **bootstrap(values.to_numpy(), seed=seed), "task_deltas": {str(k): float(v) for k, v in values.items()}}


def analyze(input_dir: Path, output_dir: Path):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_dir / "raw_results.csv")
    decisions = pd.read_csv(input_dir / "selection_decisions.csv")
    status = json.loads((input_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    duplicate_keys = int(frame.duplicated(["fit_id", "policy", "scenario", "severity", "metric"]).sum())
    required_metrics = {"auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10", "decision_utility_c5"}
    integrity = bool(
        status.get("status") == "COMPLETE" and status.get("fit_count") == EXPECTED_FITS and len(frame) == EXPECTED_ROWS
        and len(decisions) == EXPECTED_FITS and duplicate_keys == 0
        and set(frame["dataset"].astype(str).unique()) == set(EXPECTED_TASKS)
        and required_metrics.issubset(set(frame["metric"].astype(str)))
        and manifest.get("protocol_hash") == status.get("protocol_hash")
        and manifest.get("data_manifest_hash") == status.get("data_manifest_hash")
    )
    selection_rates = decisions.assign(nonbaseline=decisions["selected_candidate"].astype(str) != "none").groupby("dataset")["nonbaseline"].mean().sort_index().to_dict()
    coverage_count = int(sum(float(value) >= .5 for value in selection_rates.values()))
    summaries = [
        summarize(frame, "utility_guarded", "none", "decision_utility_c5", 20260916),
        summarize(frame, "utility_guarded", "smote_q05_k5", "decision_utility_c5", 20260917),
        summarize(frame, "utility_guarded", "none", "loss_majority", 20260918),
        summarize(frame, "utility_guarded", "none", "brier", 20260919),
        summarize(frame, "utility_guarded", "none", "loss_minority", 20260920),
    ]
    by_key = {(row["candidate"], row["reference"], row["metric"]): row for row in summaries}
    utility_none = by_key[("utility_guarded", "none", "decision_utility_c5")]
    utility_smote = by_key[("utility_guarded", "smote_q05_k5", "decision_utility_c5")]
    majority = by_key[("utility_guarded", "none", "loss_majority")]
    brier = by_key[("utility_guarded", "none", "brier")]
    conditions = {
        "selection_rates": {str(k): float(v) for k, v in selection_rates.items()},
        "coverage_task_count": coverage_count,
        "utility_vs_none_ci_low": utility_none["ci_low"], "utility_vs_none_ci_high": utility_none["ci_high"],
        "utility_vs_none_pass": bool(utility_none["ci_low"] > 0.0),
        "utility_vs_smote_ci_low": utility_smote["ci_low"], "utility_vs_smote_ci_high": utility_smote["ci_high"],
        "utility_vs_smote_pass": bool(utility_smote["ci_low"] >= -0.005),
        "majority_logloss_ci_high": majority["ci_high"], "majority_logloss_pass": bool(majority["ci_high"] <= .02),
        "brier_ci_high": brier["ci_high"], "brier_pass": bool(brier["ci_high"] <= .005),
        "duplicate_keys": duplicate_keys,
    }
    decision = "GO" if integrity and coverage_count >= 2 and conditions["utility_vs_none_pass"] and conditions["utility_vs_smote_pass"] and conditions["majority_logloss_pass"] and conditions["brier_pass"] else "NO-GO"
    payload = {"decision": decision, "integrity_pass": integrity, "expected_fits": EXPECTED_FITS, "observed_fits": int(frame["fit_id"].nunique()), "expected_raw_metric_rows": EXPECTED_ROWS, "observed_raw_metric_rows": int(len(frame)), "conditions": conditions, "summaries": summaries, "bootstrap_replicates": 10000, "bootstrap_seed": 20260917, "input": str(input_dir.resolve()), "created_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / "gate_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "task_deltas"} for row in summaries]).to_csv(output_dir / "gate_effects.csv", index=False)
    lines = ["# Utility-first selection Gate report", "", f"Decision: **{decision}**", "", f"Integrity: `{integrity}`; fits `{len(frame['fit_id'].unique())}/{EXPECTED_FITS}`; rows `{len(frame)}/{EXPECTED_ROWS}`.", "", "| condition | result |", "|---|---:|", f"| Non-baseline coverage tasks | `{coverage_count}/6` |", f"| Utility vs none CI | `[{utility_none['ci_low']:.6g}, {utility_none['ci_high']:.6g}]` |", f"| Utility vs SMOTE CI | `[{utility_smote['ci_low']:.6g}, {utility_smote['ci_high']:.6g}]` |", f"| Majority logloss upper bound vs none | `{majority['ci_high']:.6g}` |", f"| Brier upper bound vs none | `{brier['ci_high']:.6g}` |", "", "The selector and Gate were frozen before reading target results. A NO-GO does not authorize threshold, cost, task or candidate fishing.", "", "## Selection rates", ""]
    lines.extend(f"- `{task}`: `{float(rate):.3f}`" for task, rate in sorted(selection_rates.items()))
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
