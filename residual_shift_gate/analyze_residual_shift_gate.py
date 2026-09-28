"""Task-cluster analysis and frozen Gate decision for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_METRICS = ("brier", "logloss")
ALL_METRICS = ("auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10")
REFERENCE = "platt_source"
CONSENSUS = "platt_target_consensus"
GATED = "platt_target_gated"
ORACLE = "platt_target_oracle"
SHRINKAGE = "platt_target_shrinkage"
SAFETY_LOGLOSS_LIMIT = 1.0
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10
RESIDUAL_THRESHOLDS = (0.55, 0.60, 0.65)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "| no rows |\n|---|"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bootstrap_task_ci(task_deltas: pd.Series, *, n_bootstrap: int = 10_000, seed: int = 0):
    values = pd.to_numeric(task_deltas, errors="raise").to_numpy(dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("task_deltas must be a non-empty finite vector")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_tasks": int(values.size),
    }


def _stable_seed(*parts):
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def _scenario_mask(frame: pd.DataFrame, scenario_group: str, target_prior=None):
    if scenario_group == "label_shift":
        mask = frame["scenario"].eq("label_shift")
        if target_prior is not None:
            mask &= np.isclose(frame["target_prior"].astype(float), float(target_prior))
        return mask
    if scenario_group == "covariate_shift":
        return frame["scenario"].eq("covariate_shift")
    if scenario_group == "joint_shift":
        return frame["scenario"].eq("joint_shift")
    if scenario_group == "mixed_shift":
        return frame["scenario"].isin(("covariate_shift", "joint_shift"))
    raise ValueError(f"unknown scenario group: {scenario_group}")


def _task_deltas(
    frame: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    scenario_group: str,
    method: str,
    metric: str,
    target_prior=None,
):
    selected = frame[
        _scenario_mask(frame, scenario_group, target_prior)
        & frame["method"].eq(method)
        & frame["metric"].eq(metric)
        & frame["postprocessor"].isin((reference, candidate))
    ]
    if selected.empty:
        return pd.Series(dtype=float)
    index = [
        "dataset", "learner", "seed", "scenario", "severity",
        "target_prior", "covariate_severity", "postprocessor",
    ]
    pivot = selected.pivot_table(index=index[:-1], columns="postprocessor", values="value", aggfunc="first")
    if reference not in pivot or candidate not in pivot:
        return pd.Series(dtype=float)
    delta = pivot[candidate] - pivot[reference]
    return delta.groupby(level="dataset").mean()


def build_effects(frame: pd.DataFrame, *, n_bootstrap: int = 10_000, seed: int = 20260916):
    rows = []
    methods = sorted(frame["method"].astype(str).unique())
    priors = sorted(
        float(value)
        for value in frame.loc[frame["scenario"].eq("label_shift"), "target_prior"].dropna().unique()
    )
    comparisons = []
    for prior in priors:
        candidates = (GATED, CONSENSUS, ORACLE)
        if SHRINKAGE in set(frame["postprocessor"].astype(str)):
            candidates = candidates + (SHRINKAGE,)
        for candidate in candidates:
            comparisons.append((candidate, REFERENCE, "label_shift", prior))
    if SHRINKAGE in set(frame["postprocessor"].astype(str)):
        comparisons.append((SHRINKAGE, GATED, "label_shift", None))
    for scenario_group in ("covariate_shift", "joint_shift", "mixed_shift"):
        if SHRINKAGE in set(frame["postprocessor"].astype(str)):
            comparisons.append((SHRINKAGE, REFERENCE, scenario_group, None))
        else:
            comparisons.append((GATED, CONSENSUS, scenario_group, None))
            comparisons.append((GATED, REFERENCE, scenario_group, None))
    for candidate, reference, scenario_group, target_prior in comparisons:
        for method in methods:
            metric_deltas = {}
            for metric in ALL_METRICS:
                deltas = _task_deltas(
                    frame,
                    candidate=candidate,
                    reference=reference,
                    scenario_group=scenario_group,
                    method=method,
                    metric=metric,
                    target_prior=target_prior,
                )
                if not deltas.empty:
                    metric_deltas[metric] = deltas
            if not metric_deltas:
                continue
            max_task_logloss_delta = (
                float(metric_deltas["logloss"].max()) if "logloss" in metric_deltas else float("nan")
            )
            catastrophic_count = (
                int((metric_deltas["logloss"] > SAFETY_LOGLOSS_LIMIT).sum())
                if "logloss" in metric_deltas else 0
            )
            for metric, deltas in metric_deltas.items():
                summary = bootstrap_task_ci(
                    deltas,
                    n_bootstrap=n_bootstrap,
                    seed=_stable_seed(seed, candidate, reference, scenario_group, target_prior, method, metric),
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "scenario_group": scenario_group,
                        "target_prior": target_prior,
                        "method": method,
                        "metric": metric,
                        **summary,
                        "max_task_logloss_delta": max_task_logloss_delta,
                        "catastrophic_task_count": catastrophic_count,
                    }
                )
    return pd.DataFrame(rows)


def build_behavior(frame: pd.DataFrame, *, threshold: float = 0.60):
    base = frame[frame["postprocessor"].eq(REFERENCE)].copy()
    key = ["fit_id", "dataset", "learner", "method", "scenario", "severity", "target_prior", "covariate_severity"]
    base = base.drop_duplicates(key)
    base["accepted_at_threshold"] = base["residual_auc"].notna() & (base["residual_auc"] <= float(threshold))
    base["scenario_family"] = np.select(
        [base["scenario"].eq("label_shift"), base["scenario"].isin(("covariate_shift", "joint_shift"))],
        ["pure_label_shift", "mixed_shift"],
        default="iid",
    )
    grouped = (
        base.groupby(["method", "scenario_family"], dropna=False)
        .agg(
            cells=("fit_id", "nunique"),
            acceptance_rate=("accepted_at_threshold", "mean"),
            residual_auc_mean=("residual_auc", "mean"),
            residual_auc_median=("residual_auc", "median"),
        )
        .reset_index()
    )
    grouped["rejection_rate"] = 1.0 - grouped["acceptance_rate"]
    grouped["threshold"] = float(threshold)
    return grouped


def threshold_sensitivity(frame: pd.DataFrame):
    rows = []
    for threshold in RESIDUAL_THRESHOLDS:
        behavior = build_behavior(frame, threshold=threshold)
        rows.extend(behavior.to_dict(orient="records"))
    return pd.DataFrame(rows)


def gate_decision(effects: pd.DataFrame, behavior: pd.DataFrame, *, complete: bool):
    if not complete:
        return {"decision": "INCONCLUSIVE", "criteria": {}, "qualifying_methods": []}
    if effects.empty or behavior.empty:
        return {"decision": "NO-GO", "criteria": {}, "qualifying_methods": []}

    if SHRINKAGE in set(effects["candidate"].astype(str)):
        utility_methods = []
        for method in sorted(behavior["method"].unique()):
            passed = True
            for prior in (0.20, 0.50):
                current = effects[
                    (effects["candidate"] == SHRINKAGE)
                    & (effects["reference"] == REFERENCE)
                    & (effects["scenario_group"] == "label_shift")
                    & np.isclose(effects["target_prior"].astype(float), prior)
                    & (effects["method"] == method)
                    & effects["metric"].isin(PRIMARY_METRICS)
                ]
                passed &= set(current["metric"]) == set(PRIMARY_METRICS) and bool((current["ci_high"] < 0.0).all())
            if passed:
                utility_methods.append(method)

        safety_methods = []
        for method in sorted(behavior["method"].unique()):
            current = effects[
                (effects["candidate"] == SHRINKAGE)
                & (effects["reference"] == REFERENCE)
                & (effects["scenario_group"] == "mixed_shift")
                & (effects["method"] == method)
                & effects["metric"].isin(PRIMARY_METRICS)
            ]
            by_metric = current.set_index("metric") if not current.empty else pd.DataFrame()
            passed = (
                set(current["metric"]) == set(PRIMARY_METRICS)
                and float(by_metric.loc["brier", "ci_high"]) <= PRACTICAL_BRIER_MARGIN
                and float(by_metric.loc["logloss", "ci_high"]) <= PRACTICAL_LOGLOSS_MARGIN
            )
            if passed:
                safety_methods.append(method)

        improvement_methods = []
        for method in sorted(behavior["method"].unique()):
            current = effects[
                (effects["candidate"] == SHRINKAGE)
                & (effects["reference"] == GATED)
                & (effects["scenario_group"] == "label_shift")
                & (effects["target_prior"].isna())
                & (effects["method"] == method)
                & effects["metric"].isin(PRIMARY_METRICS)
            ]
            if set(current["metric"]) == set(PRIMARY_METRICS) and bool((current["mean_delta"] <= 0.0).all()):
                improvement_methods.append(method)
        criteria = {
            "utility_pass_methods": utility_methods,
            "mixed_practical_safety_pass_methods": safety_methods,
            "shrinkage_vs_binary_mean_pass_methods": improvement_methods,
            "utility_pass": len(utility_methods) >= 2,
            "mixed_safety_pass": len(safety_methods) >= 2,
            "shrinkage_vs_binary_pass": len(improvement_methods) >= 2,
            "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
            "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
        }
        decision = "GO" if all(
            (criteria["utility_pass"], criteria["mixed_safety_pass"], criteria["shrinkage_vs_binary_pass"])
        ) else "NO-GO"
        return {
            "decision": decision,
            "criteria": criteria,
            "qualifying_methods": sorted(set(utility_methods) & set(safety_methods)),
        }

    utility_methods = []
    for method in sorted(behavior["method"].unique()):
        passed = True
        for prior in (0.20, 0.50):
            current = effects[
                (effects["candidate"] == GATED)
                & (effects["reference"] == REFERENCE)
                & (effects["scenario_group"] == "label_shift")
                & np.isclose(effects["target_prior"].astype(float), prior)
                & (effects["method"] == method)
                & effects["metric"].isin(PRIMARY_METRICS)
            ]
            passed &= set(current["metric"]) == set(PRIMARY_METRICS) and bool((current["ci_high"] < 0.0).all())
        if passed:
            utility_methods.append(method)

    mixed_no_worse_methods = []
    mixed_harm_reduction_methods = []
    for method in sorted(behavior["method"].unique()):
        direct = effects[
            (effects["candidate"] == GATED)
            & (effects["reference"] == CONSENSUS)
            & (effects["scenario_group"] == "mixed_shift")
            & (effects["method"] == method)
            & effects["metric"].isin(PRIMARY_METRICS)
        ]
        direct_pass = set(direct["metric"]) == set(PRIMARY_METRICS) and bool((direct["ci_high"] <= 0.0).all())
        gated_logloss = effects[
            (effects["candidate"] == GATED)
            & (effects["reference"] == REFERENCE)
            & (effects["scenario_group"] == "mixed_shift")
            & (effects["method"] == method)
            & (effects["metric"] == "logloss")
        ]
        consensus_logloss = effects[
            (effects["candidate"] == CONSENSUS)
            & (effects["reference"] == REFERENCE)
            & (effects["scenario_group"] == "mixed_shift")
            & (effects["method"] == method)
            & (effects["metric"] == "logloss")
        ]
        harm_pass = (
            not gated_logloss.empty
            and not consensus_logloss.empty
            and int(gated_logloss.iloc[0]["catastrophic_task_count"])
            <= int(0.75 * consensus_logloss.iloc[0]["catastrophic_task_count"])
            and float(gated_logloss.iloc[0]["mean_delta"]) <= SAFETY_LOGLOSS_LIMIT
        )
        if direct_pass:
            mixed_no_worse_methods.append(method)
        if harm_pass:
            mixed_harm_reduction_methods.append(method)

    behavior_index = behavior.set_index(["method", "scenario_family"])
    behavior_methods = []
    for method in sorted(behavior["method"].unique()):
        try:
            label_accept = float(behavior_index.loc[(method, "pure_label_shift"), "acceptance_rate"]) >= 0.50
            mixed_reject = float(behavior_index.loc[(method, "mixed_shift"), "rejection_rate"]) >= 0.50
        except KeyError:
            label_accept = mixed_reject = False
        if label_accept and mixed_reject:
            behavior_methods.append(method)

    criteria = {
        "utility_pass_methods": utility_methods,
        "mixed_no_worse_pass_methods": mixed_no_worse_methods,
        "mixed_harm_reduction_pass_methods": mixed_harm_reduction_methods,
        "behavior_pass_methods": behavior_methods,
        "utility_pass": len(utility_methods) >= 2,
        "mixed_safety_pass": len(mixed_no_worse_methods) >= 2 or len(mixed_harm_reduction_methods) >= 2,
        "behavior_pass": len(behavior_methods) >= 2,
    }
    decision = "GO" if all((criteria["utility_pass"], criteria["mixed_safety_pass"], criteria["behavior_pass"])) else "NO-GO"
    return {"decision": decision, "criteria": criteria, "qualifying_methods": sorted(set(utility_methods) & set(behavior_methods))}


def _complete_input(input_dir: Path, frame: pd.DataFrame):
    status = json.loads((input_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    required_hashes = ("data_hash", "split_hash", "target_sample_hash", "config_hash")
    hashes_present = all(frame[column].astype(str).str.len().gt(0).all() for column in required_hashes)
    key = ["fit_id", "scenario", "severity", "target_prior", "covariate_severity", "postprocessor", "metric"]
    duplicate_rows = int(frame.duplicated(key).sum())
    finite_values = frame["value"].map(np.isfinite).all()
    residual_consistent = frame.loc[frame["residual_valid"].astype(bool), "residual_auc"].map(np.isfinite).all()
    complete = bool(
        status.get("status") == "COMPLETE"
        and int(status.get("raw_rows", -1)) == int(status.get("expected_raw_rows", -2))
        and len(frame) == int(status.get("expected_raw_rows", -3))
        and finite_values
        and hashes_present
        and duplicate_rows == 0
        and residual_consistent
    )
    return complete, status, manifest, duplicate_rows


def run_analysis(input_dir: Path, output_dir: Path, *, n_bootstrap: int = 10_000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_dir / "raw_results.csv")
    complete, status, manifest, duplicate_rows = _complete_input(input_dir, frame)
    effects = build_effects(frame, n_bootstrap=n_bootstrap)
    behavior = build_behavior(frame)
    sensitivity = threshold_sensitivity(frame)
    gate = gate_decision(effects, behavior, complete=complete)
    effects.to_csv(output_dir / "gate_effects.csv", index=False)
    behavior.to_csv(output_dir / "gate_behavior.csv", index=False)
    sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    analysis_manifest = {
        "input": str((input_dir / "raw_results.csv").resolve()),
        "input_protocol_hash": manifest.get("protocol_hash"),
        "bootstrap_n": int(n_bootstrap),
        "cluster_unit": "dataset",
        "reference": REFERENCE,
        "primary_metrics": list(PRIMARY_METRICS),
        "safety_logloss_limit": SAFETY_LOGLOSS_LIMIT,
        "residual_threshold": 0.60,
        "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
        "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
        "complete_input": complete,
        "duplicate_key_rows": duplicate_rows,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        **gate,
        "complete": complete,
        "input_status": status,
        "effect_rows": int(len(effects)),
        "behavior_rows": int(len(behavior)),
    }
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Residual-shift Gate report",
        "",
        f"Decision: **{gate['decision']}**",
        f"Input complete: `{complete}`; effect rows: `{len(effects)}`; behavior rows: `{len(behavior)}`.",
        "",
        "Negative deltas mean the candidate improves the metric relative to its reference.",
        "",
        "## Gate criteria",
        "",
        "```json",
        json.dumps(gate.get("criteria", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Primary effects",
        "",
        "| Candidate | Reference | Scenario | Prior | Method | Metric | Mean delta | 95% CI |",
        "|---|---|---|---:|---|---|---:|---:|",
    ]
    for row in effects[effects["metric"].isin(PRIMARY_METRICS)].sort_values(
        ["candidate", "reference", "scenario_group", "target_prior", "method", "metric"]
    ).itertuples():
        prior = "" if pd.isna(row.target_prior) else f"{row.target_prior:g}"
        lines.append(
            f"| {row.candidate} | {row.reference} | {row.scenario_group} | {prior} | {row.method} | "
            f"{row.metric} | {row.mean_delta:.6g} | [{row.ci_low:.6g}, {row.ci_high:.6g}] |"
        )
    lines.extend(["", "## Gate behavior", "", _markdown_table(behavior), ""])
    (output_dir / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    summary = run_analysis(args.input, args.output, n_bootstrap=args.bootstrap)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["decision"] != "INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
