"""Small behavioral checks; never read or process the downloaded official archive."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stream_analysis", HERE / "analyze_official.py")
stream = None
if SPEC and Path(SPEC.origin).exists():
    stream = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(stream)


def implementation():
    assert stream is not None, "Streaming implementation has not been written yet"
    return stream


def fixture_frame():
    rows = []
    for consistent in [False, True]:
        for dataset in ["d1", "d2"]:
            for seed in [0, 1, 2]:
                for learner in ["cat", "dt"]:
                    for sampler in ["none", "smote"]:
                        for hp, validation_auc, test_auc, validation_f1, test_f1 in [
                            ("a", .8, .9, .4, .5),
                            ("b", .800000011, .7, .8, .4),
                            ("c", .800000022, .6, .6, .8),
                        ]:
                            offset = .01 * seed + .04 * (dataset == "d2")
                            offset += .005 * (learner == "cat")
                            offset += .002 * (sampler == "smote")
                            rows.append(dict(dataset=dataset, seed=seed, learner=learner,
                                learner_params_set=hp, oversampler=sampler,
                                oversampler_params_set="sampler_0", param_set=hp, ratio=.3,
                                consistent=consistent, normalize=False,
                                clean_early_stopping=False, repeats=1,
                                **{"validation.roc_auc": validation_auc + offset,
                                   "test.roc_auc": test_auc + offset,
                                   "validation.f1": validation_f1 + offset,
                                   "test.f1": test_f1 + offset,
                                   "test.neg_log_loss": -.5 - offset,
                                   "test.log_loss_0": .21 + offset,
                                   "test.log_loss_1": .73 + offset,
                                   "threshold.f1": .123456789}))
    # True duplicates do not change winners, ranks, or dataset counts.
    rows.append(dict(rows[0]))
    # These must be removed before selecting the attractive, but invalid score.
    for field, value in [("normalize", True), ("clean_early_stopping", True),
                         ("repeats", 2), ("threshold.f1", np.nan)]:
        invalid = dict(rows[0])
        invalid[field] = value
        invalid["validation.roc_auc"] = .999
        invalid["param_set"] = "z"
        rows.append(invalid)
    return pd.DataFrame(rows)


def test_chunked_winners_match_official_for_all_eight_views(tmp_path):
    api = implementation()
    frame = fixture_frame()
    path = tmp_path / "small.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    official_spec = importlib.util.spec_from_file_location(
        "official_analyze", HERE.parent / "official_code" / "src" / "analyze.py")
    official = importlib.util.module_from_spec(official_spec)
    official_spec.loader.exec_module(official)
    for chunksize in [1, 19, 1000]:
        winners, audit = api.select_best_rows(path, chunksize=chunksize)
        assert len(winners) == 8
        assert audit["rows_read"] == len(frame)
        assert audit["rows_dropped_na"] == 1
        assert audit["rows_excluded_settings"] == 3
        metrics = audit["output_metrics"]
        for (consistent, opt_metric), selected in winners.items():
            filtered = frame.round(7).dropna()
            filtered = filtered[(filtered.consistent == consistent)
                                & ~filtered.normalize & ~filtered.clean_early_stopping
                                & (filtered.repeats == 1)]
            expected = official.filter_optimal_hps(filtered, opt_metric, metrics)
            actual = api.aggregate_selected(selected, metrics)
            pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                          atol=1e-12, rtol=1e-12)
            assert len(selected) == 24


def test_rounding_precedes_selection_and_all_metrics_use_one_row(tmp_path):
    api = implementation()
    path = tmp_path / "fixture.csv"
    fixture_frame().to_csv(path, index=False)
    winners, _ = api.select_best_rows(path, chunksize=5)
    for consistent in [False, True]:
        selected = winners[(consistent, "validation.roc_auc")]
        assert set(selected.param_set) == {"c"}
        assert np.allclose(selected["validation.roc_auc"] - selected["test.roc_auc"], .2)
        assert set(selected["threshold.f1"]) == {.1234568}
        assert set(winners[(consistent, "validation.f1")].param_set) == {"b"}
        assert set(winners[(consistent, "test.roc_auc")].param_set) == {"a"}


def test_two_stage_mean_and_sample_sd_are_not_pooled_row_sd():
    api = implementation()
    frame = pd.DataFrame({"dataset": ["d1", "d2"] * 2,
                          "seed": [0, 0, 1, 1], "learner": ["dt"] * 4,
                          "oversampler": ["none"] * 4,
                          "test.f1": [0., 1., .2, .4]})
    result = api.aggregate_selected(frame, ["test.f1"])
    assert result.loc[("dt", "none"), ("test.f1", "mean")] == pytest.approx(.4)
    assert result.loc[("dt", "none"), ("test.f1", "std")] == pytest.approx(np.sqrt(.02))
    assert result.loc[("dt", "none"), ("dataset", "sum")] == 4


def test_incomplete_dataset_seed_model_coverage_fails():
    api = implementation()
    frame = fixture_frame().iloc[:3].copy()
    frame = frame.iloc[[0, 1]].copy()
    frame.loc[frame.index[1], "dataset"] = "d2"
    frame.loc[frame.index[1], "seed"] = 1
    with pytest.raises(ValueError, match="coverage"):
        api.aggregate_selected(frame, ["test.f1"])


def test_hash_mismatch_rejected_before_any_output(tmp_path):
    api = implementation()
    path = tmp_path / "results.gz"
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="SHA256"):
        api.verify_official_file(path)
    out = tmp_path / "outputs"
    proc = subprocess.run([sys.executable, str(HERE / "analyze_official.py"),
                           "--input", str(path), "--output-dir", str(out)],
                          capture_output=True)
    assert proc.returncode != 0
    assert b"SHA256" in proc.stderr
    assert not out.exists()


def test_missing_schema_is_not_silently_accepted(tmp_path):
    api = implementation()
    path = tmp_path / "bad.csv"
    fixture_frame().drop(columns="param_set").to_csv(path, index=False)
    with pytest.raises(ValueError, match="param_set"):
        api.select_best_rows(path)


def test_output_manifest_is_explicit_about_oracle_and_real_scope(tmp_path):
    api = implementation()
    path = tmp_path / "small.csv"
    fixture_frame().to_csv(path, index=False)
    winners, audit = api.select_best_rows(path, chunksize=13)
    # Writer works on already-selected rows; the CLI separately enforces SHA.
    api.write_outputs(winners, audit, tmp_path / "out", input_path=path,
                      verified_sha256=None)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_verified_official"] is False
    assert manifest["datasets"] == ["d1", "d2"]
    assert manifest["learners"] == ["cat", "dt"]
    assert len(manifest["analyses"]) == 8
    assert sum(item["uses_test_for_selection"] for item in manifest["analyses"]) == 4
    assert len(list((tmp_path / "out").glob("*_summary.csv"))) == 8
