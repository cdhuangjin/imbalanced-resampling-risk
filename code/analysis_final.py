"""Confirmatory statistical summaries for the Elor follow-up experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from scipy.stats import ttest_1samp
from sklearn.neighbors import NearestNeighbors

from full_experiment import sha_bytes, split_indices, stable_hash


ANALYSIS_METRICS = (
    "auc",
    "ap",
    "logloss",
    "loss_minority",
    "loss_majority",
    "brier",
    "f1_fixed",
    "f1_tuned",
)


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(values) - np.arange(len(values))) * values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def cluster_bootstrap_mean(values, n_boot=10000, seed=20260915):
    """Bootstrap a dataset-level mean and return mean, lower, upper."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    bootstrap_means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def classify_transfer(
    *, auc_ci, minority_relative_ci, majority_delta, overall_logloss_delta=None
):
    """Apply the preregistered ranking/probability transfer labels."""
    auc_low, auc_high = auc_ci
    minority_low, _ = minority_relative_ci
    ranking = auc_low > 0
    calibration_only = auc_low <= 0 <= auc_high and minority_low > 0.05
    tradeoff = minority_low > 0.05 and majority_delta > 0
    if ranking and minority_low > 0.05:
        label = "ranking-transfer"
    elif calibration_only:
        label = "calibration/decision-transfer"
    elif ranking:
        label = "ranking-only"
    elif minority_low > 0.05:
        label = "minority-risk-transfer"
    else:
        label = "inconclusive"
    if tradeoff:
        label += "+tradeoff"
    return label


def _wide(raw):
    return raw.pivot_table(
        index=["dataset", "learner", "method", "scenario", "severity", "seed"],
        columns="metric",
        values="value",
    ).reset_index()


def _paired_task_deltas(wide, learner, method, metric):
    subset = wide[
        (wide["learner"] == learner)
        & (wide["scenario"] == "iid")
        & (wide["method"].isin(["none", method]))
    ]
    table = subset.pivot_table(
        index=["dataset", "seed"], columns="method", values=metric
    )
    return (table[method] - table["none"]).groupby(level="dataset").mean()


def _paired_relative_minority(wide, learner, method):
    subset = wide[
        (wide["learner"] == learner)
        & (wide["scenario"] == "iid")
        & (wide["method"].isin(["none", method]))
    ]
    table = subset.pivot_table(
        index=["dataset", "seed"], columns="method", values="loss_minority"
    )
    relative = (table["none"] - table[method]) / table["none"].abs()
    return relative.groupby(level="dataset").mean()


def build_mechanism_variables(datasets, seeds):
    """Compute train-fold-only minority support and k=5 overlap diagnostics."""
    loaded = fetch_datasets(
        filter_data=tuple(datasets), download_if_missing=False
    )
    rows = []
    for dataset_name in datasets:
        dataset = loaded[dataset_name]
        x = np.asarray(dataset["data"], dtype=float)
        y = (np.asarray(dataset["target"]) == 1).astype(int)
        for seed in seeds:
            train, validation, test = split_indices(y, int(seed))
            x_train = x[train]
            y_train = y[train]
            mean = np.mean(x_train, axis=0)
            scale = np.std(x_train, axis=0)
            safe_scale = np.where(scale > 1e-12, scale, 1.0)
            x_standard = (x_train - mean) / safe_scale
            minority_positions = np.flatnonzero(y_train == 1)
            neighbor_count = min(6, len(x_train))
            neighbor_model = NearestNeighbors(n_neighbors=neighbor_count)
            neighbor_model.fit(x_standard)
            neighbors = neighbor_model.kneighbors(
                x_standard[minority_positions], return_distance=False
            )
            overlap_values = []
            for position, row_neighbors in zip(minority_positions, neighbors):
                filtered = [idx for idx in row_neighbors if idx != position][:5]
                overlap_values.append(float(np.mean(y_train[filtered] == 0)))
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "minority_support": int(len(minority_positions)),
                    "minority_fraction": float(np.mean(y_train)),
                    "overlap5": float(np.mean(overlap_values)),
                    "data_hash": sha_bytes(x, y),
                    "split_hash": stable_hash(
                        {
                            "train": train.tolist(),
                            "validation": validation.tolist(),
                            "test": test.tolist(),
                        }
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_analysis(main_output: Path, shift_output: Path | None, output: Path):
    main_output = Path(main_output)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(main_output / "raw_results.csv")
    wide = _wide(raw)
    learners = ["dt", "cat"]
    methods = ["random", "smote"]
    rows = []
    relative_rows = []
    classification_rows = []
    pvalue_positions = []
    for learner in learners:
        for method in methods:
            relative = _paired_relative_minority(wide, learner, method)
            relative_mean, relative_low, relative_high = cluster_bootstrap_mean(
                relative.to_numpy(), seed=stable_hash((learner, method, "relative"))[:8]
                and int(stable_hash((learner, method, "relative"))[:8], 16)
            )
            relative_rows.append(
                {
                    "learner": learner,
                    "method": method,
                    "metric": "minority_relative_improvement",
                    "mean": relative_mean,
                    "ci_low": relative_low,
                    "ci_high": relative_high,
                    "n_datasets": len(relative),
                }
            )
            metric_summary = {}
            for metric in ANALYSIS_METRICS:
                task_deltas = _paired_task_deltas(wide, learner, method, metric)
                mean, low, high = cluster_bootstrap_mean(
                    task_deltas.to_numpy(),
                    seed=int(stable_hash((learner, method, metric))[:8], 16),
                )
                p_value = float(ttest_1samp(task_deltas.to_numpy(), 0.0).pvalue)
                metric_summary[metric] = (mean, low, high)
                rows.append(
                    {
                        "learner": learner,
                        "method": method,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "positive_datasets": int((task_deltas > 0).sum()),
                        "n_datasets": len(task_deltas),
                        "p_unadjusted": p_value,
                        "primary_endpoint": bool(method == "smote" and metric == "auc"),
                    }
                )
                if not (method == "smote" and metric == "auc"):
                    pvalue_positions.append((len(rows) - 1, learner, method, p_value))
            auc = metric_summary["auc"]
            majority = metric_summary["loss_majority"][0]
            relative_metric = relative_mean, relative_low, relative_high
            classification_rows.append(
                {
                    "learner": learner,
                    "method": method,
                    "auc_ci_low": auc[1],
                    "auc_ci_high": auc[2],
                    "minority_relative_ci_low": relative_low,
                    "minority_relative_ci_high": relative_high,
                    "majority_logloss_delta": majority,
                    "classification": classify_transfer(
                        auc_ci=(auc[1], auc[2]),
                        minority_relative_ci=(relative_low, relative_high),
                        majority_delta=majority,
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    for learner in learners:
        for method in methods:
            mask = (summary["learner"] == learner) & (summary["method"] == method)
            secondary = summary.loc[mask & ~summary["primary_endpoint"], "p_unadjusted"]
            adjusted = holm_adjust(secondary.to_numpy())
            summary.loc[secondary.index, "p_holm"] = adjusted
    _write_csv(summary, output / "primary_effects.csv")
    _write_csv(pd.DataFrame(relative_rows), output / "minority_relative_effects.csv")
    _write_csv(pd.DataFrame(classification_rows), output / "transfer_classification.csv")

    failure_rows = []
    for learner in learners:
        for method in methods:
            auc_delta = _paired_task_deltas(wide, learner, method, "auc")
            minority_delta = _paired_task_deltas(wide, learner, method, "loss_minority")
            majority_delta = _paired_task_deltas(wide, learner, method, "loss_majority")
            brier_delta = _paired_task_deltas(wide, learner, method, "brier")
            for dataset in auc_delta.index:
                failure_rows.append(
                    {
                        "dataset": dataset,
                        "learner": learner,
                        "method": method,
                        "delta_auc": float(auc_delta.loc[dataset]),
                        "delta_loss_minority": float(minority_delta.loc[dataset]),
                        "delta_loss_majority": float(majority_delta.loc[dataset]),
                        "delta_brier": float(brier_delta.loc[dataset]),
                        "ranking_harm": bool(auc_delta.loc[dataset] < 0),
                        "majority_risk_harm": bool(majority_delta.loc[dataset] > 0),
                    }
                )
    failures = pd.DataFrame(failure_rows)
    _write_csv(failures, output / "failure_cases.csv")

    datasets = sorted(raw["dataset"].unique())
    seeds = [int(value) for value in sorted(raw["seed"].unique())]
    mechanism = build_mechanism_variables(datasets, seeds)
    _write_csv(mechanism, output / "mechanism_variables.csv")

    mechanism_rows = []
    for learner in learners:
        delta_table = wide[
            (wide["learner"] == learner) & (wide["method"].isin(["none", "smote"]))
        ].pivot_table(index=["dataset", "seed"], columns="method", values="auc")
        delta_table = (delta_table["smote"] - delta_table["none"]).rename("delta_auc")
        joined = delta_table.reset_index().merge(mechanism, on=["dataset", "seed"])
        joined["learner_cat"] = int(learner == "cat")
        joined["log_support"] = np.log(joined["minority_support"])
        joined["cat_x_overlap5"] = joined["learner_cat"] * joined["overlap5"]
        mechanism_rows.append(joined)
    mechanism_data = pd.concat(mechanism_rows, ignore_index=True)
    mechanism_data.to_csv(output / "mechanism_model_input.csv", index=False)
    try:
        import statsmodels.api as sm

        x_columns = ["learner_cat", "log_support", "overlap5", "cat_x_overlap5"]
        x_design = sm.add_constant(mechanism_data[x_columns])
        model = sm.OLS(mechanism_data["delta_auc"], x_design).fit(
            cov_type="cluster", cov_kwds={"groups": mechanism_data["dataset"]}
        )
        model_table = pd.DataFrame(
            {
                "term": model.params.index,
                "coef": model.params.to_numpy(),
                "se_cluster_dataset": model.bse.to_numpy(),
                "p_value": model.pvalues.to_numpy(),
            }
        )
        _write_csv(model_table, output / "mechanism_model.csv")
    except Exception as exc:
        (output / "mechanism_model_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    analysis_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_output": str(main_output),
        "shift_output": str(shift_output) if shift_output else None,
        "main_raw_rows": int(len(raw)),
        "main_datasets": datasets,
        "main_seeds": seeds,
        "bootstrap_replicates": 10000,
        "cluster_unit": "dataset",
        "holm_family": "learner x method secondary endpoints",
        "mechanism_variables": ["minority_support", "minority_fraction", "overlap5"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return analysis_manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--shift-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_analysis(args.main_output, args.shift_output, args.output))
