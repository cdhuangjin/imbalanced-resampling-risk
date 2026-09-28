"""R0: stream the author's complete results CSV and reproduce its aggregation.

Usage (from any directory): python analyze_official.py [--chunksize 100000]
The CLI accepts only the exact Git LFS object identified by OFFICIAL_SHA256;
there is deliberately no skip-hash or alternative-hash option. A .gz suffix
means gzip CSV, as in the official README. No training or download is performed.

Processing: round numeric raw fields to seven decimals; drop rows with any NA
(official README); normalize=False, clean_early_stopping=False, repeats=1;
then consider both consistency settings and four HP-selection metrics. Only a
chunk and one best row per (dataset, seed, learner, oversampler) per analysis
remain in memory. Identical duplicate configurations are therefore harmless.

Selection maximizes the chosen score, breaking ties by descending param_set.
The official first sort does not specify stable ordering for an exact tie in
both score and param_set. We make that residual case reproducible by retaining
the earliest input row, and report it explicitly in the manifest.

Ranking, dataset means and seed mean/sample-SD follow filter_optimal_hps.
All metrics, including positive class-log-loss fields, are ranked descending
to reproduce that function. Positive-loss ranks must not be interpreted as
"smaller is better" ranks. No sign conversion or synthetic class-loss metric
is added. The test.* selection views are oracle/a-priori analyses, not honest
validation-selected performance estimates.

The pure selection/aggregation functions also accept small synthetic data for
behavioral tests. Only main() labels a successful hash-verified run official.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SHA256 = "c7db60622f7facb58e6f77260032832b88513e2d62b3c28f13484c69e930327b"
DEFAULT_INPUT = ROOT / "official_code" / "data" / "results.gz"
DEFAULT_OUTPUT = ROOT / "outputs" / "r0"
GROUP = ["dataset", "seed", "learner", "oversampler"]
OPT_METRICS = ["validation.roc_auc", "test.roc_auc", "validation.f1", "test.f1"]
REQUIRED = GROUP + ["learner_params_set", "oversampler_params_set", "param_set",
                    "ratio", "consistent", "normalize", "clean_early_stopping", "repeats"]


def verify_official_file(path: Path) -> str:
    """Hash bounded blocks before CSV parsing/output; reject pointers/partials."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != OFFICIAL_SHA256:
        raise ValueError(f"SHA256 mismatch for {path}: expected {OFFICIAL_SHA256}, got {actual}. "
                         "Only the complete official Git LFS object is accepted.")
    return actual


def _boolean(series: pd.Series) -> pd.Series:
    # read_csv normally infers bool; tolerate its textual and numeric forms.
    normalized = series.astype(str).str.lower().str.strip()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False,
                             "1.0": True, "0.0": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid boolean values in {series.name}: "
                         f"{sorted(normalized[mapped.isna()].unique())[:5]}")
    return mapped.astype(bool)


def select_best_rows(input_path: Path, chunksize: int = 100000):
    """Internal stream reducer; caller must verify SHA before an official run.

    Returns {(consistent: bool, opt_metric: str): selected DataFrame}, audit.
    _source_row is a zero-based CSV data-row offset, retained for traceability.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    input_path = Path(input_path)
    header = pd.read_csv(input_path, nrows=0)
    missing = sorted(set(REQUIRED + OPT_METRICS) - set(header.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    if "_source_row" in header.columns:
        raise ValueError("CSV uses reserved column _source_row")
    metrics = [column for column in header.columns
               if column.startswith(("validation.", "test."))]
    # Keep identifier ordering independent of pandas chunk inference. The
    # released param_set is an opaque identifier, sorted lexicographically.
    text_columns = ["dataset", "learner", "oversampler", "learner_params_set",
                    "oversampler_params_set", "param_set"]
    state = {(consistent, metric): None
             for consistent in [False, True] for metric in OPT_METRICS}
    audit = {"rows_read": 0, "rows_dropped_na": 0, "rows_excluded_settings": 0,
             "rows_eligible": 0, "chunks_read": 0, "chunksize": chunksize,
             "peak_retained_winner_rows": 0, "output_metrics": metrics,
             "raw_columns": header.columns.tolist()}

    with pd.read_csv(input_path, chunksize=chunksize,
                     dtype={column: str for column in text_columns}) as reader:
        for chunk in reader:
            start = audit["rows_read"]
            audit["rows_read"] += len(chunk)
            audit["chunks_read"] += 1
            # Source IDs are assigned before NA removal so they reference the
            # original CSV, including invalid/excluded rows.
            chunk = chunk.round(7)
            chunk["_source_row"] = range(start, start + len(chunk))
            before = len(chunk)
            chunk = chunk.dropna().copy()
            audit["rows_dropped_na"] += before - len(chunk)
            if chunk.empty:
                continue
            for column in ["consistent", "normalize", "clean_early_stopping"]:
                chunk[column] = _boolean(chunk[column])
            for column in ["seed", "repeats", "ratio", *metrics]:
                chunk[column] = pd.to_numeric(chunk[column], errors="raise").round(7)
            eligible = (~chunk.normalize & ~chunk.clean_early_stopping & (chunk.repeats == 1))
            audit["rows_excluded_settings"] += int((~eligible).sum())
            chunk = chunk.loc[eligible].copy()
            audit["rows_eligible"] += len(chunk)
            for consistent in [False, True]:
                candidates = chunk.loc[chunk.consistent == consistent]
                if candidates.empty:
                    continue
                for opt_metric in OPT_METRICS:
                    key = (consistent, opt_metric)
                    combined = candidates if state[key] is None else pd.concat(
                        [state[key], candidates], ignore_index=True)
                    best = combined.sort_values(
                        [opt_metric, "param_set", "_source_row"],
                        ascending=[False, False, True], kind="stable")
                    # Drop whole rows, never independently take the best value
                    # of each output metric. This also eliminates duplicates.
                    state[key] = best.drop_duplicates(GROUP, keep="first").copy()
            audit["peak_retained_winner_rows"] = max(
                audit["peak_retained_winner_rows"],
                sum(len(frame) for frame in state.values() if frame is not None))
    missing_views = [str(key) for key, frame in state.items() if frame is None or frame.empty]
    if missing_views:
        raise ValueError(f"No eligible rows for analyses: {missing_views}")
    selected = {key: frame.sort_values(GROUP).reset_index(drop=True)
                for key, frame in state.items()}
    return selected, audit


def aggregate_selected(selected: pd.DataFrame, output_metrics: list[str]) -> pd.DataFrame:
    """Official descending average ranks, dataset mean, then seed mean/std."""
    if selected.empty:
        raise ValueError("No selected rows to aggregate")
    if selected.duplicated(GROUP).any():
        raise ValueError("Selected rows contain duplicate dataset/seed/model groups")
    expected = selected.dataset.nunique() * selected.seed.nunique()
    counts = selected.groupby(["learner", "oversampler"]).size()
    if not counts.eq(expected).all():
        raise ValueError(f"Incomplete dataset/seed/model coverage: expected {expected} "
                         f"per model, observed {counts.to_dict()}")
    frame = selected[[*GROUP, *output_metrics]].copy()
    for metric in output_metrics:
        frame[f"{metric}.rank"] = frame.groupby(["dataset", "seed"])[metric].rank(
            ascending=False, method="average")
    rank_metrics = [f"{metric}.rank" for metric in output_metrics]
    by_seed = frame.groupby(["learner", "seed", "oversampler"]).agg(
        {**{metric: "mean" for metric in [*output_metrics, *rank_metrics]},
         "dataset": "count"})
    return by_seed.groupby(["learner", "oversampler"]).agg(
        {**{metric: ["mean", "std"] for metric in [*output_metrics, *rank_metrics]},
         "dataset": "sum"})


def write_outputs(winners, audit, output_dir: Path, *, input_path: Path,
                  verified_sha256: str | None):
    """Write row provenance, flat summary tables and a machine-readable manifest."""
    output_dir = Path(output_dir)
    # Validate every view before creating any output: a missing model×seed cell
    # must not leave an apparently successful partial result collection.
    summaries = {key: aggregate_selected(frame, audit["output_metrics"])
                 for key, frame in winners.items()}
    scopes = pd.concat([frame[GROUP] for frame in winners.values()], ignore_index=True)
    manifest = {
        "analysis": "R0 official published-results reanalysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).resolve()),
        "input_bytes": Path(input_path).stat().st_size,
        "input_sha256": verified_sha256,
        "expected_official_sha256": OFFICIAL_SHA256,
        "input_verified_official": verified_sha256 == OFFICIAL_SHA256,
        "pandas_version": pd.__version__,
        "numeric_rounding_before_selection": 7,
        "filter": {"dropna": "all original columns", "normalize": False,
                   "clean_early_stopping": False, "repeats": 1},
        "selection": "score descending, param_set descending; all metrics from one row",
        "exact_tie_policy": "same score and param_set: earliest CSV row; official first sort is unstable",
        "param_set_order": "lexicographic opaque identifier",
        "ranking": "descending average ranks across every learner/oversampler for each dataset and seed",
        "aggregation": "dataset mean within each seed, then mean and sample SD (ddof=1) across seeds",
        "positive_class_log_loss_rank_caveat": "descending reproduces official function; higher raw loss is worse",
        "datasets": sorted(scopes.dataset.unique().tolist()),
        "seeds": sorted(scopes.seed.unique().tolist()),
        "learners": sorted(scopes.learner.unique().tolist()),
        "oversamplers": sorted(scopes.oversampler.unique().tolist()),
        "scope_caveat": "Observed artifact contents only; no fabricated missing learner/dataset results",
        "streaming": audit,
        "analyses": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for (consistent, opt_metric), frame in winners.items():
        name = f"consistent_{str(consistent).lower()}__select_{opt_metric.replace('.', '_')}"
        selected_name = f"{name}_selected.csv.gz"
        summary_name = f"{name}_summary.csv"
        frame.to_csv(output_dir / selected_name, index=False, compression="gzip", float_format="%.7f")
        summary = summaries[(consistent, opt_metric)].copy()
        summary.columns = [f"{metric}.{stat}" for metric, stat in summary.columns]
        # Do not round aggregation arithmetic to seven decimals: only raw
        # fields are rounded. Preserve enough precision for comparisons.
        summary.reset_index().to_csv(output_dir / summary_name, index=False, float_format="%.15g")
        manifest["analyses"].append({
            "consistent": consistent, "selection_metric": opt_metric,
            "uses_test_for_selection": opt_metric.startswith("test."),
            "selection_label": "oracle/a-priori" if opt_metric.startswith("test.") else "validation-selected",
            "selected_rows": len(frame), "datasets": int(frame.dataset.nunique()),
            "seeds": int(frame.seed.nunique()), "selected_file": selected_name,
            "summary_file": summary_name,
        })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        print(f"Verifying SHA256: {args.input}", flush=True)
        digest = verify_official_file(args.input)
        initial = args.input.stat()
        print("SHA256 verified; streaming CSV selection.", flush=True)
        winners, audit = select_best_rows(args.input, chunksize=args.chunksize)
        final = args.input.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise ValueError("Input file changed during analysis; refusing to write official output")
        manifest = write_outputs(winners, audit, args.output_dir, input_path=args.input,
                                 verified_sha256=digest)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "rows_read": audit["rows_read"],
                      "datasets": len(manifest["datasets"]),
                      "learners": manifest["learners"], "analyses": len(manifest["analyses"])},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
