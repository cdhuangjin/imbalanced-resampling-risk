"""Low-cost, pre-registered Gate for the Elor mechanism-map direction."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official_code" / "src"))
from classifiers import _fit_catboost, _fit_decision_tree  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DATASETS = ("ecoli", "yeast_me2", "abalone_19", "mammography", "oil", "spectrometer")
SEEDS = tuple(range(5))
METHODS = ("none", "random", "smote")
CONFIGS = (
    ("dt_coarse", "dt", {"min_samples_leaf_factor": 0.04}),
    ("dt_fine", "dt", {"min_samples_leaf_factor": 0.005}),
    ("cat_default", "cat", {}),
    ("cat_500", "cat", {"iterations": 500}),
)
PROTOCOL = ROOT / "gate_protocol.md"
DEFAULT_OUTPUT = ROOT / "outputs" / "gate"
EXPECTED_RECORDS = len(DATASETS) * len(SEEDS) * len(CONFIGS) * len(METHODS)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def package_versions():
    names = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "catboost", "scipy"]
    return {name: importlib.metadata.version(name) for name in names}


def split_indices(y: np.ndarray, seed: int):
    trval, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    train, validation = train_test_split(trval, test_size=0.25, stratify=y[trval], random_state=seed)
    return train, validation, test


def target_count(y: np.ndarray, ratio: float = 0.5) -> int:
    return int(np.round(np.sum(y == 0) * ratio / (1 - ratio)))


def metrics(y: np.ndarray, p: np.ndarray):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    losses = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(losses.mean()),
        "loss_minority": float(losses[y == 1].mean()),
        "loss_majority": float(losses[y == 0].mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def all_run_ids():
    return [f"{dataset}__{config}__s{seed}__{method}"
            for dataset in DATASETS for seed in SEEDS
            for config, _, _ in CONFIGS for method in METHODS]


def _overlap5(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    xtr, ytr = np.asarray(x[train_idx], dtype=float), np.asarray(y[train_idx], dtype=int)
    n_min = int(np.sum(ytr == 1))
    if n_min < 6:
        return float("nan"), n_min, False
    center = xtr.mean(axis=0)
    scale = xtr.std(axis=0)
    scale[scale == 0] = 1.0
    z = (xtr - center) / scale
    minority = np.flatnonzero(ytr == 1)
    nn = NearestNeighbors(n_neighbors=6, algorithm="auto", n_jobs=1).fit(z)
    neighbors = nn.kneighbors(z[minority], return_distance=False)[:, 1:]
    overlap = float(np.mean(ytr[neighbors] == 0))
    return overlap, n_min, True


def prepare(out: Path):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing nonempty Gate output directory: {out}")
    for name in ("data", "splits", "runs", "predictions"):
        (out / name).mkdir(parents=True, exist_ok=True)
    datasets = fetch_datasets(filter_data=DATASETS, download_if_missing=False)
    data_manifest, split_hashes, factors = [], {}, {}
    for name in DATASETS:
        d = datasets[name]
        x = np.asarray(d["data"], dtype=float)
        y = (np.asarray(d["target"]) == 1).astype(int)
        if not np.isfinite(x).all() or np.unique(y).size != 2:
            raise ValueError(f"Invalid binary data: {name}")
        data_path = out / "data" / f"{name}.npz"
        np.savez_compressed(data_path, x=x, y=y)
        factor_rows = []
        for seed in SEEDS:
            train, validation, test = split_indices(y, seed)
            split_path = out / "splits" / f"{name}_{seed}.npz"
            np.savez_compressed(split_path, train=train, validation=validation, test=test)
            split_hashes[f"{name}_{seed}"] = sha(split_path)
            overlap, n_min, feasible = _overlap5(x, y, train)
            factor_rows.append(dict(seed=seed, minority_support=n_min,
                                    overlap5=overlap, smote_feasible=feasible))
        factors[name] = factor_rows
        data_manifest.append(dict(dataset=name, shape=list(x.shape), minority=int(y.sum()),
                                  sha256=sha(data_path),
                                  source="https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz"))
    official_sources = {p.name: sha(p) for p in (ROOT / "official_code" / "src").glob("*.py")}
    manifest = dict(schema_version=1, analysis="pre-registered Elor Gate",
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                    official_sources=official_sources, packages=package_versions(),
                    datasets=data_manifest, split_sha256=split_hashes, factors=factors,
                    data=DATASETS, seeds=SEEDS, configs=[c[0] for c in CONFIGS], methods=METHODS,
                    expected_records=EXPECTED_RECORDS, test_used_for_selection=False,
                    max_workers=1, resource_limits=dict(wall_seconds=7200, cpu_seconds=7200,
                                                         rss_bytes=4 * 1024**3, output_bytes=1 * 1024**3))
    write_json(out / "manifest.json", manifest)
    return manifest


def _fit_predict(x, y, train, validation, test, seed, model_name, params, method):
    xr, yr = x[train], y[train]
    if method != "none":
        target = target_count(yr, 0.5)
        sampler = (RandomOverSampler(sampling_strategy={1: target}, random_state=seed)
                   if method == "random" else
                   SMOTE(sampling_strategy={1: target}, k_neighbors=5, random_state=seed))
        xr, yr = sampler.fit_resample(xr, yr)
    local = deepcopy(params)
    if model_name == "cat":
        local["allow_writing_files"] = False
    with threadpool_limits(limits=1):
        model = (_fit_decision_tree if model_name == "dt" else _fit_catboost)(
            xr, yr, x[validation], y[validation], seed, local)
        kwargs = {"thread_count": 1} if model_name == "cat" else {}
        pv = model.predict_proba(x[validation], **kwargs)[:, 1]
        pt = model.predict_proba(x[test], **kwargs)[:, 1]
    complexity = int(model.tree_count_) if model_name == "cat" else int(model.get_n_leaves())
    return pv, pt, metrics(y[validation], pv), metrics(y[test], pt), len(yr), complexity


def run_one(out: Path, manifest: dict, dataset: str, seed: int, config_id: str,
            model_name: str, params: dict, method: str):
    run_id = f"{dataset}__{config_id}__s{seed}__{method}"
    data_path = out / "data" / f"{dataset}.npz"
    split_path = out / "splits" / f"{dataset}_{seed}.npz"
    with np.load(data_path) as data:
        x, y = data["x"], data["y"]
    with np.load(split_path) as split:
        train, validation, test = (split[k] for k in ("train", "validation", "test"))
    record = dict(run_id=run_id, dataset=dataset, seed=seed, config=config_id,
                  model=model_name, method=method, ratio=None if method == "none" else 0.5,
                  k=None if method != "smote" else 5,
                  manifest_sha256=sha(out / "manifest.json"),
                  data_sha256=sha(data_path), split_sha256=sha(split_path),
                  test_used_for_selection=False)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    try:
        pv, pt, validation_metrics, test_metrics, n_resampled, complexity = _fit_predict(
            x, y, train, validation, test, seed, model_name, params, method)
        pred_path = out / "predictions" / f"{run_id}.npz"
        np.savez_compressed(pred_path, validation_prob=pv, test_prob=pt,
                            validation_y=y[validation], test_y=y[test],
                            validation_idx=validation, test_idx=test)
        record.update(status="success", validation=validation_metrics, test=test_metrics,
                      n_train=len(train), n_resampled=n_resampled, n_validation=len(validation),
                      n_test=len(test), capacity_proxy=complexity,
                      prediction=str(pred_path.relative_to(out)), prediction_sha256=sha(pred_path))
    except Exception as error:
        record.update(status="failed", failure=repr(error), traceback=traceback.format_exc())
    record.update(wall_seconds=time.perf_counter() - started_wall,
                  cpu_seconds=time.process_time() - started_cpu,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "runs" / f"{run_id}.json", record)
    return record


def cluster_bootstrap_ci(values, seed=20260913, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def decide_gate(summary: dict):
    reasons = []
    contrast_ok = (summary.get("contrast_mean", float("nan")) >= 0.015 and
                   summary.get("contrast_ci_low", float("nan")) > 0 and
                   summary.get("positive_datasets", 0) >= 4)
    transfer = summary.get("minority_relative_improvements", [])
    transfer_ok = sum(value >= 0.05 for value in transfer) >= 4
    complete_ok = (summary.get("complete", False) and summary.get("all_hashes_valid", False)
                   and summary.get("leakage_free", False)
                   and summary.get("resource_ok", False)
                   and summary.get("successful_records", 0) == summary.get("expected_records", EXPECTED_RECORDS))
    if not contrast_ok:
        reasons.append("interaction criterion failed")
    if not transfer_ok:
        reasons.append("minority-loss transfer criterion failed")
    if not complete_ok:
        reasons.append("integrity/completeness/resource criterion failed")
    return dict(decision="GO" if contrast_ok and transfer_ok and complete_ok else "NO-GO",
                reasons=reasons or ["all pre-registered criteria passed"],
                criteria=dict(interaction=contrast_ok, minority_transfer=transfer_ok,
                              integrity=complete_ok))


def analyze(out: Path, manifest: dict, started_wall: float):
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((out / "runs").glob("*.json"))]
    expected = set(all_run_ids())
    observed = {record.get("run_id") for record in records}
    complete = len(records) == EXPECTED_RECORDS and observed == expected and all(
        record.get("status") == "success" for record in records)
    hashes_valid, leakage_free = True, True
    for record in records:
        leakage_free &= record.get("test_used_for_selection") is False
        if record.get("status") != "success":
            hashes_valid = False
            continue
        path = out / record["prediction"]
        if not path.is_file() or sha(path) != record.get("prediction_sha256"):
            hashes_valid = False
    frame = pd.json_normalize([r for r in records if r.get("status") == "success"])
    contrasts, transfer_rows = [], []
    if not frame.empty:
        pivot = frame.pivot_table(index=["dataset", "seed", "config"], columns="method",
                                  values=["test.auc", "test.loss_minority"], aggfunc="first")
        for dataset in DATASETS:
            ds = pivot.loc[dataset]
            auc_delta = ds["test.auc"]["smote"] - ds["test.auc"]["none"]
            configs = auc_delta.index.get_level_values("config").astype(str)
            dt_delta = float(auc_delta[configs.str.startswith("dt_")].mean())
            cat_delta = float(auc_delta[configs.str.startswith("cat_")].mean())
            contrasts.append(dict(dataset=dataset, dt=dt_delta, cat=cat_delta,
                                  contrast=dt_delta - cat_delta))
            none_loss = float(ds["test.loss_minority"]["none"].mean())
            smote_loss = float(ds["test.loss_minority"]["smote"].mean())
            transfer_rows.append(dict(dataset=dataset, none=none_loss, smote=smote_loss,
                                      relative=(none_loss - smote_loss) / none_loss))
    contrast_values = np.array([row["contrast"] for row in contrasts], dtype=float)
    ci_low, ci_high = cluster_bootstrap_ci(contrast_values) if len(contrast_values) == 6 else (float("nan"), float("nan"))
    summary = dict(expected_records=EXPECTED_RECORDS, observed_records=len(records),
                   successful_records=int(len(frame)), complete=complete,
                   all_hashes_valid=bool(hashes_valid), leakage_free=bool(leakage_free),
                   resource_ok=(time.perf_counter() - started_wall <= 7200 and
                                sum(float(r.get("cpu_seconds", 0)) for r in records) <= 7200 and
                                sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= 1 * 1024**3),
                   contrast_mean=float(np.nanmean(contrast_values)) if len(contrast_values) else float("nan"),
                   contrast_ci_low=ci_low, contrast_ci_high=ci_high,
                   positive_datasets=int(sum(v > 0 for v in contrast_values)),
                   contrasts=contrasts, minority_relative_improvements=[r["relative"] for r in transfer_rows],
                   transfer=transfer_rows, wall_seconds=time.perf_counter() - started_wall,
                   cpu_seconds=float(sum(float(r.get("cpu_seconds", 0)) for r in records)))
    summary.update(decide_gate(summary))
    write_json(out / "run_summary.json", summary)
    write_json(out / "gate_decision.json", summary)
    if not frame.empty:
        frame.to_csv(out / "records.csv", index=False)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyze an existing complete output without retraining")
    args = parser.parse_args(argv)
    out = args.output
    started = time.perf_counter()
    if args.analyze_only:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        summary = analyze(out, manifest, started)
        print(json.dumps({"decision": summary["decision"], "contrast_mean": summary["contrast_mean"],
                          "contrast_ci": [summary["contrast_ci_low"], summary["contrast_ci_high"]],
                          "positive_datasets": summary["positive_datasets"],
                          "successful_records": summary["successful_records"]}, ensure_ascii=False), flush=True)
        return 0 if summary["decision"] == "GO" else 2
    manifest = prepare(out)
    for dataset in DATASETS:
        for seed in SEEDS:
            for config_id, model_name, params in CONFIGS:
                for method in METHODS:
                    record = run_one(out, manifest, dataset, seed, config_id, model_name, params, method)
                    print(json.dumps({"run_id": record["run_id"], "status": record["status"]}), flush=True)
    summary = analyze(out, manifest, started)
    print(json.dumps({"decision": summary["decision"], "contrast_mean": summary["contrast_mean"],
                      "contrast_ci": [summary["contrast_ci_low"], summary["contrast_ci_high"]],
                      "positive_datasets": summary["positive_datasets"],
                      "successful_records": summary["successful_records"]}, ensure_ascii=False), flush=True)
    return 0 if summary["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
