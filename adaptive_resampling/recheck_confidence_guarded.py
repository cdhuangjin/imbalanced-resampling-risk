"""Independent integrity and effect recheck for the confidence-guarded Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap(values, *, seed, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_tasks": int(values.size),
    }


def task_delta(frame, candidate, reference, metric):
    selected = frame[
        (frame["scenario"] == "iid")
        & (frame["severity"].astype(int) == 0)
        & (frame["policy"].isin([candidate, reference]))
        & (frame["metric"] == metric)
    ]
    wide = selected.pivot_table(
        index=["dataset", "learner", "seed"],
        columns="policy",
        values="value",
        aggfunc="first",
    )
    return (wide[candidate] - wide[reference]).dropna().groupby(level="dataset").mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.input / "raw_results.csv")
    decisions = pd.read_csv(args.input / "selection_decisions.csv")
    status = json.loads((args.input / "status.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.input / "confidence_guarded_protocol.json").read_text(encoding="utf-8"))
    expected_fit_count = int(status["expected_fit_count"])
    duplicate_keys = int(
        raw.duplicated(["fit_id", "policy", "scenario", "severity", "metric"]).sum()
    )
    task_names = sorted(raw["dataset"].astype(str).unique())
    selection_rates = (
        decisions.assign(nonbaseline=decisions["selected_candidate"].astype(str) != "none")
        .groupby("dataset")["nonbaseline"]
        .mean()
        .sort_index()
        .to_dict()
    )
    effects = {
        "adaptive_vs_smote_majority_logloss": bootstrap(
            task_delta(raw, "adaptive_rc", "smote_q05_k5", "loss_majority").to_numpy(),
            seed=20260916,
        ),
        "adaptive_vs_smote_brier": bootstrap(
            task_delta(raw, "adaptive_rc", "smote_q05_k5", "brier").to_numpy(),
            seed=20260917,
        ),
        "adaptive_vs_none_minority_logloss": bootstrap(
            task_delta(raw, "adaptive_rc", "none", "loss_minority").to_numpy(),
            seed=20260918,
        ),
    }
    primary = json.loads((args.primary / "gate_summary.json").read_text(encoding="utf-8"))
    primary_summaries = {
        f"{row['candidate']}_vs_{row['reference']}_{row['metric']}": row
        for row in primary["summaries"]
    }
    effect_match = {
        "adaptive_vs_smote_majority_logloss": np.isclose(
            effects["adaptive_vs_smote_majority_logloss"]["mean"],
            primary_summaries["adaptive_rc_vs_smote_q05_k5_loss_majority"]["mean_delta"],
        ),
        "adaptive_vs_smote_brier": np.isclose(
            effects["adaptive_vs_smote_brier"]["mean"],
            primary_summaries["adaptive_rc_vs_smote_q05_k5_brier"]["mean_delta"],
        ),
        "adaptive_vs_none_minority_logloss": np.isclose(
            effects["adaptive_vs_none_minority_logloss"]["mean"],
            primary_summaries["adaptive_rc_vs_none_loss_minority"]["mean_delta"],
        ),
    }
    payload = {
        "integrity_pass": bool(
            status["status"] == "COMPLETE"
            and status["fit_count"] == expected_fit_count
            and status["raw_rows"] == status["expected_raw_rows"]
            and len(decisions) == expected_fit_count
            and duplicate_keys == 0
        ),
        "fit_count": int(raw["fit_id"].nunique()),
        "raw_rows": int(len(raw)),
        "decision_rows": int(len(decisions)),
        "duplicate_keys": duplicate_keys,
        "task_names": task_names,
        "selection_rates": {str(k): float(v) for k, v in selection_rates.items()},
        "effects": effects,
        "effect_match_to_primary": {key: bool(value) for key, value in effect_match.items()},
        "confidence_guarded_protocol": protocol,
    }
    payload["recheck_pass"] = bool(payload["integrity_pass"] and all(effect_match.values()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recheck.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["recheck_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
