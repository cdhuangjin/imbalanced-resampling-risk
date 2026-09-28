"""Analyze the V5.5 residual-stability-calibrated policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "platt_source"
SHRINKAGE = "platt_target_shrinkage"
RSC = "platt_target_rsc"
RASS = "platt_target_rass"
PRIMARY_METRICS = ("brier", "logloss")
ALL_METRICS = ("auc", "logloss", "loss_minority", "loss_majority", "brier", "ece_10")
PRACTICAL_BRIER_MARGIN = 0.01
PRACTICAL_LOGLOSS_MARGIN = 0.10


def _stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def bootstrap_task_ci(values: pd.Series, *, n_bootstrap: int = 10_000, seed: int = 0):
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
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


def _scenario_mask(frame: pd.DataFrame, group: str, target_prior=None):
    if group == "label_shift":
        mask = frame["scenario"].eq("label_shift")
        if target_prior is not None:
            mask &= np.isclose(frame["target_prior"].astype(float), float(target_prior))
        return mask
    if group == "covariate_shift":
        return frame["scenario"].eq("covariate_shift")
    if group == "joint_shift":
        return frame["scenario"].eq("joint_shift")
    if group == "mixed_shift":
        return frame["scenario"].isin(("covariate_shift", "joint_shift"))
    raise ValueError(f"unknown scenario group: {group}")


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
    pivot = selected.pivot_table(
        index=index[:-1], columns="postprocessor", values="value", aggfunc="first"
    )
    if reference not in pivot or candidate not in pivot:
        return pd.Series(dtype=float)
    delta = pivot[candidate] - pivot[reference]
    return delta.groupby(level="dataset").mean()


def build_effects(frame: pd.DataFrame, *, n_bootstrap: int = 10_000, seed: int = 20260917):
    methods = sorted(frame["method"].astype(str).unique())
    priors = sorted(
        float(value)
        for value in frame.loc[frame["scenario"].eq("label_shift"), "target_prior"].dropna().unique()
    )
    comparisons = []
    for prior in priors:
        comparisons.extend(
            [
                (RSC, REFERENCE, "label_shift", prior),
                (SHRINKAGE, REFERENCE, "label_shift", prior),
                (RSC, SHRINKAGE, "label_shift", prior),
                (RASS, REFERENCE, "label_shift", prior),
                (RASS, SHRINKAGE, "label_shift", prior),
                (RASS, RSC, "label_shift", prior),
            ]
        )
    comparisons.extend(
        [
            (RSC, REFERENCE, "mixed_shift", None),
            (RSC, SHRINKAGE, "mixed_shift", None),
            (SHRINKAGE, REFERENCE, "mixed_shift", None),
            (RASS, REFERENCE, "mixed_shift", None),
            (RASS, SHRINKAGE, "mixed_shift", None),
            (RASS, RSC, "mixed_shift", None),
        ]
    )
    rows = []
    for candidate, reference, scenario_group, target_prior in comparisons:
        for method in methods:
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
                if deltas.empty:
                    continue
                rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "scenario_group": scenario_group,
                        "target_prior": target_prior,
                        "method": method,
                        "metric": metric,
                        **bootstrap_task_ci(
                            deltas,
                            n_bootstrap=n_bootstrap,
                            seed=_stable_seed(
                                seed, candidate, reference, scenario_group, target_prior, method, metric
                            ),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_behavior(frame: pd.DataFrame):
    required = {"postprocessor", "residual_auc", "rsc_stability_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing behavior columns: " + ", ".join(sorted(missing)))
    selected = frame[frame["postprocessor"].eq(RSC)].copy()
    key = [
        "fit_id", "dataset", "learner", "method", "scenario", "severity",
        "target_prior", "covariate_severity",
    ]
    selected = selected.drop_duplicates(key)
    selected["scenario_family"] = np.select(
        [
            selected["scenario"].eq("label_shift"),
            selected["scenario"].isin(("covariate_shift", "joint_shift")),
        ],
        ["pure_label_shift", "mixed_shift"],
        default="iid",
    )
    selected["rsc_stability_weight"] = pd.to_numeric(
        selected["rsc_stability_weight"], errors="coerce"
    )
    selected["residual_auc"] = pd.to_numeric(selected["residual_auc"], errors="coerce")
    return (
        selected.groupby(["method", "scenario_family"], dropna=False)
        .agg(
            cells=("fit_id", "nunique"),
            residual_auc_mean=("residual_auc", "mean"),
            residual_auc_median=("residual_auc", "median"),
            stability_weight_mean=("rsc_stability_weight", "mean"),
            stability_weight_median=("rsc_stability_weight", "median"),
        )
        .reset_index()
    )


def _effect(effects, candidate, reference, scenario_group, method, metric, target_prior=None):
    selected = effects[
        effects["candidate"].eq(candidate)
        & effects["reference"].eq(reference)
        & effects["scenario_group"].eq(scenario_group)
        & effects["method"].eq(method)
        & effects["metric"].eq(metric)
    ]
    if target_prior is None:
        selected = selected[selected["target_prior"].isna()]
    else:
        selected = selected[np.isclose(selected["target_prior"].astype(float), float(target_prior))]
    return selected.iloc[0] if len(selected) == 1 else None


def gate_decision(effects: pd.DataFrame, *, complete: bool):
    if not complete:
        return {"decision": "INCONCLUSIVE", "criteria": {}, "qualifying_methods": []}
    candidate = RASS if RASS in set(effects.get("candidate", ())) else RSC
    methods = sorted(effects["method"].astype(str).unique()) if not effects.empty else []
    utility_methods = []
    safety_methods = []
    mixed_noninferior_methods = []
    for method in methods:
        utility_pass = True
        for prior in (0.20, 0.50):
            for metric in PRIMARY_METRICS:
                row = _effect(effects, candidate, REFERENCE, "label_shift", method, metric, prior)
                utility_pass &= row is not None and float(row["ci_high"]) < 0.0
        if utility_pass:
            utility_methods.append(method)

        safety_pass = True
        for metric in PRIMARY_METRICS:
            row = _effect(effects, candidate, REFERENCE, "mixed_shift", method, metric)
            safety_pass &= (
                row is not None
                and float(row["ci_high"]) <= (
                    PRACTICAL_BRIER_MARGIN if metric == "brier" else PRACTICAL_LOGLOSS_MARGIN
                )
            )
        if safety_pass:
            safety_methods.append(method)

        noninferior = True
        for metric in PRIMARY_METRICS:
            row = _effect(effects, candidate, SHRINKAGE, "mixed_shift", method, metric)
            noninferior &= row is not None and float(row["ci_high"]) <= 0.0
        if noninferior:
            mixed_noninferior_methods.append(method)

    criteria = {
        "utility_pass_methods": utility_methods,
        "mixed_practical_safety_pass_methods": safety_methods,
        "mixed_noninferior_to_v51_methods": mixed_noninferior_methods,
        "utility_pass": len(utility_methods) >= 2,
        "mixed_safety_pass": len(safety_methods) >= 2,
        "mixed_noninferior_to_v51": len(mixed_noninferior_methods) >= 2,
        "practical_brier_margin": PRACTICAL_BRIER_MARGIN,
        "practical_logloss_margin": PRACTICAL_LOGLOSS_MARGIN,
    }
    decision = "GO" if all(
        (criteria["utility_pass"], criteria["mixed_safety_pass"], criteria["mixed_noninferior_to_v51"])
    ) else "NO-GO"
    return {
        "decision": decision,
        "candidate": candidate,
        "criteria": criteria,
        "qualifying_methods": sorted(
            set(utility_methods) & set(safety_methods) & set(mixed_noninferior_methods)
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_input(input_dir: Path, frame: pd.DataFrame):
    status = json.loads((input_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    required = ("data_hash", "split_hash", "target_sample_hash", "config_hash")
    hashes_present = all(frame[column].astype(str).str.len().gt(0).all() for column in required)
    key = [
        "fit_id", "scenario", "severity", "target_prior", "covariate_severity",
        "postprocessor", "metric",
    ]
    duplicate_rows = int(frame.duplicated(key).sum())
    finite_values = frame["value"].map(np.isfinite).all()
    expected = int(status.get("expected_raw_rows", -1))
    complete = bool(
        status.get("status") == "COMPLETE"
        and int(status.get("raw_rows", -1)) == expected
        and len(frame) == expected
        and finite_values
        and hashes_present
        and duplicate_rows == 0
        and RSC in set(frame["postprocessor"].astype(str))
        and RASS in set(frame["postprocessor"].astype(str))
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
    gate = gate_decision(effects, complete=complete)
    effects.to_csv(output_dir / "rsc_effects.csv", index=False)
    behavior.to_csv(output_dir / "rsc_behavior.csv", index=False)
    analysis_manifest = {
        "input": str((input_dir / "raw_results.csv").resolve()),
        "input_sha256": _file_sha256(input_dir / "raw_results.csv"),
        "input_protocol_hash": manifest.get("protocol_hash"),
        "bootstrap_n": int(n_bootstrap),
        "cluster_unit": "dataset",
        "reference": REFERENCE,
        "stability_variant": RSC,
        "regime_variant": RASS,
        "primary_metrics": list(PRIMARY_METRICS),
        "complete_input": complete,
        "duplicate_key_rows": duplicate_rows,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        **gate,
        "complete": complete,
        "input_status": status,
        "effect_rows": int(len(effects)),
        "behavior_rows": int(len(behavior)),
    }
    (output_dir / "rsc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V5.5/V5.6 residual-stability and regime-switch report",
        "",
        f"Decision: **{gate['decision']}**",
        f"Input complete: `{complete}`; effect rows: `{len(effects)}`; behavior rows: `{len(behavior)}`.",
        "",
        "RSC multiplies the frozen residual-AUC adaptation weight by a label-free target-prior stability weight; RASS keeps V5.1 in the accepted regime and switches to RSC above residual AUC 0.60.",
        "Negative deltas mean the candidate policy improves the reference metric.",
        "",
        "## Gate criteria",
        "",
        f"- Candidate policy `{gate.get('candidate', RSC)}` utility at target priors 0.20 and 0.50: `{len(gate['criteria'].get('utility_pass_methods', []))}` methods pass.",
        f"- Mixed-shift practical safety: `{len(gate['criteria'].get('mixed_practical_safety_pass_methods', []))}` methods pass.",
        f"- Mixed-shift non-inferiority to V5.1 shrinkage: `{len(gate['criteria'].get('mixed_noninferior_to_v51_methods', []))}` methods pass.",
        "",
        "A NO-GO result is retained as a boundary and does not license post-hoc threshold changes.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.input, args.output, n_bootstrap=args.replicates), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
