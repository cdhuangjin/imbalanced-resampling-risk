import factor_experiment


def test_fit_factor_model_forwards_explicit_learner_parameters(monkeypatch):
    captured = {}

    def fake_fit(x_train, y_train, x_validation, y_validation, seed, params):
        captured["seed"] = seed
        captured["params"] = params
        return object()

    monkeypatch.setattr(factor_experiment, "_fit_decision_tree", fake_fit)
    factor_experiment.fit_factor_model(
        "dt",
        {"min_samples_leaf_factor": 0.005},
        None,
        None,
        None,
        None,
        7,
    )

    assert captured == {"seed": 7, "params": {"min_samples_leaf_factor": 0.005}}


def test_learner_and_method_specs_are_explicit_and_deterministic():
    assert factor_experiment.parse_learner_spec("dt_leaf005") == (
        "dt",
        {"min_samples_leaf_factor": 0.005},
    )
    assert factor_experiment.parse_learner_spec("cat_iter500") == (
        "cat",
        {"iterations": 500},
    )
    assert factor_experiment.parse_method_spec("smote_q02_k3") == (
        "smote",
        {"ratio": 0.2, "k": 3},
    )
    assert factor_experiment.parse_method_spec("none") == ("none", {})


def test_factor_specs_have_expected_cartesian_product():
    specs = factor_experiment.build_factor_specs(
        datasets=("d1", "d2"),
        learners=("dt_leaf005", "cat_iter500"),
        methods=("none", "smote_q05_k5"),
        seeds=(0, 1, 2),
    )

    assert len(specs) == 24
    assert specs[0]["run_id"] == "d1__dt_leaf005__s0__none__iid__sev0"
    assert specs == factor_experiment.build_factor_specs(
        datasets=("d1", "d2"),
        learners=("dt_leaf005", "cat_iter500"),
        methods=("none", "smote_q05_k5"),
        seeds=(0, 1, 2),
    )
