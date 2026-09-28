import json
from pathlib import Path

import numpy as np
import pytest

import full_experiment


def test_build_run_specs_is_complete_and_deterministic():
    specs = full_experiment.build_run_specs(
        datasets=("d1", "d2"),
        learners=("dt", "cat"),
        methods=("none", "random", "smote"),
        seeds=(0, 1),
        scenarios=("iid",),
        severities=(0,),
    )

    assert len(specs) == 24
    assert specs == full_experiment.build_run_specs(
        datasets=("d1", "d2"),
        learners=("dt", "cat"),
        methods=("none", "random", "smote"),
        seeds=(0, 1),
        scenarios=("iid",),
        severities=(0,),
    )
    assert specs[0]["run_id"] == "d1__dt__s0__none__iid__sev0"


def test_order_run_specs_applies_a_complete_randomized_schedule(tmp_path: Path):
    specs = full_experiment.build_run_specs(
        datasets=("d1",),
        learners=("dt",),
        methods=("none", "random", "smote"),
        seeds=(0, 1),
    )
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(
        "run_order,dataset,learner,seed,method\n"
        "1,d1,dt,1,smote\n"
        "2,d1,dt,1,none\n"
        "3,d1,dt,1,random\n"
        "4,d1,dt,0,random\n"
        "5,d1,dt,0,smote\n"
        "6,d1,dt,0,none\n",
        encoding="utf-8",
    )

    ordered = full_experiment.order_run_specs(specs, schedule)

    assert [spec["run_id"] for spec in ordered] == [
        "d1__dt__s1__smote__iid__sev0",
        "d1__dt__s1__none__iid__sev0",
        "d1__dt__s1__random__iid__sev0",
        "d1__dt__s0__random__iid__sev0",
        "d1__dt__s0__smote__iid__sev0",
        "d1__dt__s0__none__iid__sev0",
    ]


@pytest.mark.parametrize(
    "body,error",
    [
        ("1,d1,dt,0,none\n2,d1,dt,0,none\n", "duplicate"),
        ("1,d1,dt,0,none\n", "does not match"),
    ],
)
def test_order_run_specs_rejects_duplicate_or_incomplete_schedules(
    tmp_path: Path, body: str, error: str
):
    specs = full_experiment.build_run_specs(
        datasets=("d1",), learners=("dt",), methods=("none", "random"), seeds=(0,)
    )
    schedule = tmp_path / "bad_schedule.csv"
    schedule.write_text("run_order,dataset,learner,seed,method\n" + body, encoding="utf-8")

    with pytest.raises(full_experiment.ProvenanceError, match=error):
        full_experiment.order_run_specs(specs, schedule)


def test_runtime_difference_from_main_is_explicitly_recorded():
    assert full_experiment.package_version_differences({"scikit-learn": "1.6.0"}) == {
        "scikit-learn": {"expected": "1.8.0", "observed": "1.6.0"}
    }
    assert full_experiment.package_version_differences({"scikit-learn": "1.8.0"}) == {}


def test_split_indices_are_disjoint_and_cover_all_rows():
    y = np.array([0, 1] * 20)
    train, validation, test = full_experiment.split_indices(y, seed=7)

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert sorted(np.concatenate([train, validation, test]).tolist()) == list(range(len(y)))


def test_build_data_manifest_records_shape_counts_and_hashes():
    data = {
        "toy": {
            "data": np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]),
            "target": np.array([-1, -1, 1]),
        }
    }

    manifest = full_experiment.build_data_manifest(data, ("toy",))

    assert manifest["toy"]["n_samples"] == 3
    assert manifest["toy"]["n_features"] == 2
    assert manifest["toy"]["class_counts"] == {"-1": 2, "1": 1}
    assert len(manifest["toy"]["data_hash"]) == 64


def test_raw_record_schema_rejects_missing_provenance():
    record = {"run_id": "x", "dataset": "d", "metric": "auc", "value": 0.5}

    with pytest.raises(full_experiment.ProvenanceError, match="config_hash"):
        full_experiment.validate_raw_record(record)


def test_aggregate_results_requires_all_declared_metrics(tmp_path: Path):
    raw = tmp_path / "raw_results.csv"
    raw.write_text(
        "run_id,dataset,learner,method,seed,scenario,severity,metric,value,config_hash,"
        "data_hash,split_hash,runtime_seconds,timestamp\n"
        "x,d,dt,none,0,iid,0,auc,0.5,c,d,s,0.1,t\n",
        encoding="utf-8",
    )

    with pytest.raises(full_experiment.ProvenanceError, match="metric coverage"):
        full_experiment.aggregate_results(raw, tmp_path / "summary.csv")
