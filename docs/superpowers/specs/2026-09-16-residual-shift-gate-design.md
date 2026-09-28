# V5 Residual-Shift Gate Design

Date: 2026-09-16  
Status: frozen for implementation  
Scope: experiments only

## 1. Research question

V4 showed that an estimator-consensus policy can improve target-weighted Brier and logloss under controlled label-prior shift. Its validity still depends on the label-shift condition `P_t(X|Y) = P_s(X|Y)`. V5 asks:

> Can an unlabeled residual-shift diagnostic, applied after estimated-prior reweighting, decide when to use consensus prior correction and when to fall back to source-only Platt calibration under mixed shift?

The contribution is an empirical safety policy for imbalanced tabular deployment, not a new prior estimator or a universal domain-adaptation algorithm.

## 2. Literature-grounded design boundary

- BBSE defines label shift through invariant `P(X|Y)` and uses unlabeled target features to estimate and correct the target prior: Lipton, Wang, and Smola (ICML 2018), https://proceedings.mlr.press/v80/lipton18a.html.
- Classifier two-sample tests detect distributional change from source and target covariates: Jang et al. (ICML 2022), https://proceedings.mlr.press/v162/jang22a.html.
- Mixed-shift test-time adaptation motivates separating shift types rather than applying one correction indiscriminately: Zhou et al. (ICML 2023), https://proceedings.mlr.press/v202/zhou23e.html.

V5 combines these established ingredients only as a pre-registered empirical gate. It does not claim novelty for BBSE, EM, Platt calibration, or the two-sample statistic.

## 3. Frozen policy variants

Every model fit produces the following prediction policies:

1. `platt_source`: V4 source-only Platt calibration.
2. `platt_target_consensus`: V4 consensus prior correction, always applied.
3. `platt_target_gated`: apply the V4 consensus correction only when the residual-shift test accepts the label-shift assumption; otherwise use `platt_source`.
4. `platt_target_oracle`: target-prior oracle, reference only and never used for Gate qualification.

The gate receives source validation features/labels and an unlabeled target-feature sample. Target labels are not passed to estimation, detection, or policy selection. Target labels are used only afterward for evaluation under the known synthetic scenario.

## 4. Residual-shift diagnostic

For a fitted model and target sample:

1. Estimate target prior using the frozen V4 consensus estimator.
2. Assign source-validation rows class-conditional importance weights `pi_hat / pi_s` and `(1-pi_hat)/(1-pi_s)`.
3. Form a domain-classification problem between the reweighted source-validation features and the unlabeled target features.
4. Fit a standardized logistic domain classifier with deterministic three-fold out-of-fold predictions.
5. Compute weighted domain ROC-AUC. The primary acceptance rule is `AUC_residual <= 0.60`; `AUC_residual > 0.60` rejects the label-shift assumption and triggers source-Platt fallback.

The threshold is frozen before Gate execution. A secondary threshold sensitivity analysis at 0.55 and 0.65 is descriptive only and cannot change the primary decision. Invalid domain fits, non-finite statistics, or insufficient class support are conservative rejection/fallback events.

## 5. Experimental matrix

### Screening Gate

- Tasks: `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer`.
- Learners: `dt_leaf01`, `cat_iter500`.
- Training methods: `none`, `random_q05`, `smote_q05_k5`.
- Seeds: `0, 1, 2`.
- Scenarios:
  - `iid`;
  - pure label shift, target priors `0.05`, `0.20`, `0.50`;
  - pure covariate shift, first non-constant feature translated by `1` or `2` training standard deviations;
  - joint shift, target priors `0.20` and `0.50` combined with covariate translation `1` or `2`.
- Expected fits: `6 x 2 x 3 x 3 = 108`.
- Expected scenario cells: `1 + 3 + 2 + 4 = 10` per fit.
- Expected raw metric rows: `108 x 10 x 4 x 6 = 25,920`.
- In pure covariate shift the target prior is the source prior. In joint shift the target priors are `0.20` and `0.50`, and the same unlabeled target-feature pool is transformed before it reaches the detector or policy.

### Confirmation matrix, only if Gate is GO

- The same protocol expanded to the existing 24-task fixed matrix.
- Seeds: `0, 1, 2, 3, 4`.
- No threshold, learner, task, scenario, or metric may be changed after Gate inspection.

## 6. Endpoints and Gate criteria

Primary endpoints are target-weighted Brier and logloss. AUC, minority/majority logloss, ECE, domain AUC, acceptance rate, and fallback reason are secondary diagnostics.

Task-cluster bootstrap is the primary uncertainty protocol: average seeds within task, then bootstrap task clusters with 10,000 replicates. The Gate is `GO` only if all conditions hold:

1. Complete, finite, hash-linked output with zero duplicate `(fit_id, scenario, severity, target_prior, postprocessor, metric)` keys.
2. Under pure label shift at target priors `0.20` and `0.50`, `platt_target_gated` improves both Brier and logloss over `platt_source` with bootstrap CI upper bounds below zero in at least two of three training-method strata.
3. Under covariate and joint shift, `platt_target_gated` is no worse than `platt_target_consensus` for both primary endpoints by the task-cluster CI rule, or reduces the number of task-level catastrophic logloss failures by at least 25% without increasing source-relative mean logloss by more than 1.0. A catastrophic task-level failure is defined before analysis as a gated/consensus logloss increase over `platt_source` greater than `1.0` after seed averaging.
4. The gate is operationally non-degenerate: after averaging seeds within `(task, learner, method, scenario)`, it accepts at least 50% of pure-label-shift cells and rejects at least 50% of covariate/joint-shift cells in at least two of three training-method strata.

If any condition fails, V5 is retained as a bounded negative or conditional result and the 24-task confirmation is not unlocked.

## 7. Provenance and safety

- Reuse the existing local benchmark cache `<local path omitted>` and the existing split/sampling/model code.
- Check `<local path omitted>` before any data download; no new download is planned.
- Store protocol hash, data-manifest hash, source-code hashes, package versions, split hashes, target-sample hashes, detector diagnostics, and fallback reasons in every output.
- Resume only when the output protocol and schema hashes match.
- Never use target labels to estimate priors, fit the domain detector, select the threshold, or select a policy.

## 8. Tests and analysis

Unit tests cover weighted domain-AUC calculation, deterministic cross-fitting, invalid-input fallback, and target-label isolation. Runner tests cover scenario construction, row completeness, hashes, and resume behavior. Analysis tests cover task-cluster aggregation, gate criteria, and threshold sensitivity.

The analysis must report both utility and abstention behavior. It must preserve negative boundary cases and cannot discard tasks because a gate rejects them.
