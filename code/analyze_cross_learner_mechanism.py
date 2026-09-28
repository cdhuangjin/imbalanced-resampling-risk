"""Synthesize the completed six-task learner-family Gates without new fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("ecoli", "yeast_me2", "abalone_19", "mammography", "oil", "spectrometer")
METHODS = ("none", "random_q05", "smote_q05_k5")
METRICS = ("auc", "loss_minority", "loss_majority", "brier")
SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260917


def paired_bootstrap_ci(values, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)), "n_tasks": int(values.size), "replicates": int(replicates), "seed": int(seed)}


def _validate_frame(frame, *, learner_names, expected_methods=None):
    required = {"run_id", "dataset", "learner", "method", "seed", "metric", "value", "config_hash", "data_hash", "split_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw results missing columns: {missing}")
    if not set(frame["learner"].astype(str)).issubset(set(learner_names)):
        raise ValueError("unexpected learner in synthesis input")
    if expected_methods is not None and not set(frame["method"].astype(str)).issubset(set(expected_methods)):
        raise ValueError("unexpected method in synthesis input")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("raw values must be finite")
    for field in ("config_hash", "data_hash", "split_hash"):
        if frame[field].astype(str).eq("").any():
            raise ValueError(f"empty provenance hash in {field}")


def load_common_inputs(legacy_input, xgb_input, brf_input, lgbm_input):
    specs = [
        ("legacy", Path(legacy_input), ("dt_leaf005", "dt_leaf01", "dt_leaf02", "dt_leaf04", "cat_default", "cat_iter500")),
        ("xgb", Path(xgb_input), ("xgb_gate",)),
        ("brf", Path(brf_input), ("brf_gate",)),
        ("lgbm", Path(lgbm_input), ("lgbm_gate",)),
    ]
    frames = []
    input_hashes = {}
    for label, directory, learners in specs:
        frame = pd.read_csv(directory / "raw_results.csv")
        _validate_frame(frame, learner_names=learners)
        frame = frame[
            frame["dataset"].isin(TASKS)
            & frame["method"].isin(METHODS)
            & frame["metric"].isin(METRICS)
            & frame["seed"].isin(SEEDS)
        ].copy()
        frame["source"] = label
        frames.append(frame)
        digest = hashlib.sha256((directory / "raw_results.csv").read_bytes()).hexdigest()
        input_hashes[label] = {"path": str(directory), "raw_results_sha256": digest}
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise ValueError("common synthesis input is empty")
    return merged, input_hashes


def build_effects(frame):
    seed_means = frame.groupby(["source", "learner", "dataset", "method", "metric"], as_index=False)["value"].mean()
    task_learner_effects = []
    for method in METHODS[1:]:
        for metric in METRICS:
            current = seed_means[
                (seed_means["method"] == method) & (seed_means["metric"] == metric)
            ].set_index(["source", "learner", "dataset", "metric"])["value"]
            baseline = seed_means[
                (seed_means["method"] == "none") & (seed_means["metric"] == metric)
            ].set_index(["source", "learner", "dataset", "metric"])["value"]
            joined = pd.concat({"treatment": current, "none": baseline}, axis=1).dropna()
            joined["effect"] = joined["treatment"] - joined["none"]
            for index, row in joined.reset_index().iterrows():
                task_learner_effects.append({"source": row["source"], "learner": row["learner"], "dataset": row["dataset"], "method": method, "metric": metric, "effect": float(row["effect"])})
    cells = pd.DataFrame(task_learner_effects)
    task_effects = []
    for method in METHODS[1:]:
        for metric in METRICS:
            subset = cells[(cells["method"] == method) & (cells["metric"] == metric)]
            task_means = subset.groupby("dataset", as_index=False)["effect"].mean()
            values = task_means["effect"].to_numpy(dtype=float)
            ci = paired_bootstrap_ci(values)
            task_effects.append({"method": method, "metric": metric, "n_tasks": ci["n_tasks"], "mean": ci["mean"], "ci_low": ci["low"], "ci_high": ci["high"], "min_task_effect": float(values.min()), "max_task_effect": float(values.max()), "positive_task_count": int((values > 0).sum()), "negative_task_count": int((values < 0).sum()), "bootstrap_replicates": ci["replicates"], "bootstrap_seed": ci["seed"]})
    return pd.DataFrame(task_effects), cells


def summarize_mechanism(effects):
    conditions = []
    for method in METHODS[1:]:
        minority = effects[(effects["method"] == method) & (effects["metric"] == "loss_minority")].iloc[0]
        auc = effects[(effects["method"] == method) & (effects["metric"] == "auc")].iloc[0]
        conditions.append(bool(minority["ci_high"] < 0.0 and auc["ci_low"] <= 0.0 <= auc["ci_high"]))
    decision = "SUPPORTS_CONDITIONAL_MECHANISM" if any(conditions) else "BOUNDARY_OR_INCONCLUSIVE"
    return {"decision": decision, "minority_signal_methods": [method for method, ok in zip(METHODS[1:], conditions) if ok], "inference_unit": "six tasks after learner averaging", "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED}


def analyze(legacy_input, xgb_input, brf_input, lgbm_input, output_dir):
    frame, input_hashes = load_common_inputs(legacy_input, xgb_input, brf_input, lgbm_input)
    effects, cells = build_effects(frame)
    summary = summarize_mechanism(effects)
    summary.update({"input_hashes": input_hashes, "observed_rows": int(len(frame)), "observed_task_count": int(frame["dataset"].nunique()), "observed_learner_count": int(frame["learner"].nunique())})
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "task_effects.csv", index=False)
    cells.to_csv(output_dir / "task_learner_effects.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Cross-learner mechanism synthesis", "", f"- Decision: `{summary['decision']}`", f"- Input rows: `{summary['observed_rows']}`", f"- Tasks: `{summary['observed_task_count']}`", f"- Learner configurations: `{summary['observed_learner_count']}`", "", "| method | metric | mean effect | 95% CI | worst task range |", "|---|---|---:|---:|---:|"]
    for row in effects.itertuples(index=False): lines.append(f"| {row.method} | {row.metric} | {row.mean:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | [{row.min_task_effect:.6f}, {row.max_task_effect:.6f}] |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, effects, cells


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--legacy-input", type=Path, required=True); parser.add_argument("--xgb-input", type=Path, required=True); parser.add_argument("--brf-input", type=Path, required=True); parser.add_argument("--lgbm-input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); analyze(args.legacy_input, args.xgb_input, args.brf_input, args.lgbm_input, args.output)


if __name__ == "__main__":
    main()
