# V5.5 residual-stability-calibrated experiment protocol

## Question

Does adding a label-free stability diagnostic to the frozen V5.1 residual-AUC shrinkage policy preserve its label-shift utility while reducing dependence on an unstable target-prior estimate?

This is an empirical policy extension around established components. It is not presented as a new prior estimator, a new domain-test theorem, or a new sampler.

## Literature basis

- Unlabeled target calibration work motivates conservative blending toward the source prior and entropy-aware target calibration: [Li, Shi and Cui, ICASSP 2026](https://ieeexplore.ieee.org/document/11464109/).
- Class-probability matching provides a recent calibrated-network baseline for label-shift adaptation: [Wen et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html).
- Label-free harmful-shift detection motivates using an error-relevant proxy before adapting: [Amoukou et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html).
- Recent tabular benchmark work shows that dataset/domain choice can change method rankings, motivating a separate external audit rather than a single iid table: [CLIMB, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/45ee52e746c65e5e60205bedf2d18386-Abstract-Conference.html).

The present extension combines these ideas as a pre-specified confidence policy: residual domain separability controls shift compatibility, while repeated target-probability resampling controls prior-estimate stability.

## Frozen matrix

| Item | Frozen choice |
|---|---|
| Tasks | Six Gate tasks: `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer` |
| Learners | `dt_leaf01`, `cat_iter500` |
| Training treatments | `none`, `random_q05`, `smote_q05_k5` |
| Seeds | `0, 1, 2` |
| Shift scenarios | iid; label shift at `.05/.20/.50`; covariate shift at severities `1/2`; joint shift at priors `.20/.50` and severities `1/2` |
| Target context | `max(2000, n_test)` unlabeled feature rows; target labels are unavailable to detector, calibration, and policy selection |
| Prior stability | 16 deterministic bootstrap resamples of target probabilities only; EM/soft-BBSE agreement tolerance `.10`; EM dispersion scale `.05` |
| Residual policy | frozen V5.1 knots `lower=.55`, `upper=.75`; weight is `clip((.75-AUC)/(.75-.55),0,1)` |
| RSC weight | residual-policy weight × stability weight |
| Metrics | AUC, log-loss, minority/majority log-loss, Brier, ECE-10 |
| Inference | task-cluster paired bootstrap, 10,000 deterministic replicates |

## Stability diagnostic

For each target context, resample the unlabeled target probabilities with replacement. Each replicate estimates the positive target prior using calibrated EM and soft BBSE. The stability weight is:

`valid_rate × agreement_rate × clip(1 − EM_SD/.05, 0, 1)`.

If no valid replicate exists, or the weight is numerically invalid, RSC falls back to source calibration. The target labels are not passed to this computation.

## Pre-specified screening decision

RSC is marked `GO` only if at least two of the three sampling treatments satisfy all of the following:

1. At target priors `.20` and `.50`, both Brier and log-loss RSC-minus-source 95% task-cluster bootstrap upper bounds are below zero.
2. Under mixed shift, Brier and log-loss RSC-minus-source upper bounds are within `.01` and `.10`, respectively.
3. Under mixed shift, RSC is non-inferior to V5.1 shrinkage for both primary metrics (upper bound `≤ 0`).

Otherwise the result is `NO-GO` or `INCONCLUSIVE`; no threshold or task list is changed after test inspection.

## Outputs

The run writes `raw_results.csv`, `manifest.json`, `data_manifest.json`, `status.json`, and the analysis writes `rsc_effects.csv`, `rsc_behavior.csv`, `rsc_summary.json`, `analysis_manifest.json`, and `report.md`. The primary raw output is treated as immutable after completion.
