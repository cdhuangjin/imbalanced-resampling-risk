"""Independent recheck for the utility-first selective-treatment Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def boot(values, seed, n=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(n), values.size))
    samples = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975)), "n_tasks": int(values.size)}


def delta(frame, candidate, reference, metric):
    selected = frame[(frame["scenario"] == "iid") & (frame["severity"].astype(int) == 0) & (frame["policy"].isin([candidate, reference])) & (frame["metric"] == metric)]
    wide = selected.pivot_table(index=["dataset", "learner", "seed"], columns="policy", values="value", aggfunc="first")
    return (wide[candidate] - wide[reference]).dropna().groupby(level="dataset").mean().to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input / "raw_results.csv")
    decisions = pd.read_csv(args.input / "selection_decisions.csv")
    status = json.loads((args.input / "status.json").read_text(encoding="utf-8"))
    primary = json.loads((args.primary / "gate_summary.json").read_text(encoding="utf-8"))
    duplicate_keys = int(frame.duplicated(["fit_id", "policy", "scenario", "severity", "metric"]).sum())
    rates = decisions.assign(nonbaseline=decisions["selected_candidate"].astype(str) != "none").groupby("dataset")["nonbaseline"].mean().sort_index().to_dict()
    effects = {
        "utility_vs_none": boot(delta(frame, "utility_guarded", "none", "decision_utility_c5"), 20260916),
        "utility_vs_smote": boot(delta(frame, "utility_guarded", "smote_q05_k5", "decision_utility_c5"), 20260917),
        "majority_vs_none": boot(delta(frame, "utility_guarded", "none", "loss_majority"), 20260918),
        "brier_vs_none": boot(delta(frame, "utility_guarded", "none", "brier"), 20260919),
    }
    primary_rows = {(row["candidate"], row["reference"], row["metric"]): row for row in primary["summaries"]}
    matching = {
        "utility_vs_none": np.isclose(effects["utility_vs_none"]["mean"], primary_rows[("utility_guarded", "none", "decision_utility_c5")]["mean"]),
        "utility_vs_smote": np.isclose(effects["utility_vs_smote"]["mean"], primary_rows[("utility_guarded", "smote_q05_k5", "decision_utility_c5")]["mean"]),
        "majority_vs_none": np.isclose(effects["majority_vs_none"]["mean"], primary_rows[("utility_guarded", "none", "loss_majority")]["mean"]),
        "brier_vs_none": np.isclose(effects["brier_vs_none"]["mean"], primary_rows[("utility_guarded", "none", "brier")]["mean"]),
    }
    payload = {
        "integrity_pass": bool(status["status"] == "COMPLETE" and status["fit_count"] == 36 and status["raw_rows"] == 5040 and len(decisions) == 36 and duplicate_keys == 0),
        "fit_count": int(frame["fit_id"].nunique()), "raw_rows": int(len(frame)), "decision_rows": int(len(decisions)),
        "duplicate_keys": duplicate_keys, "selection_rates": {str(k): float(v) for k, v in rates.items()},
        "effects": effects, "effect_match_to_primary": {key: bool(value) for key, value in matching.items()},
    }
    payload["recheck_pass"] = bool(payload["integrity_pass"] and all(matching.values()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recheck.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["recheck_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
