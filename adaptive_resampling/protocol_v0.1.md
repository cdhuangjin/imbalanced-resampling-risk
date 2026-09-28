# Risk-Constrained Support-Gated Resampling Protocol v0.1

Date frozen: 2026-09-16  
Status: Gate protocol; the 24-task extension is conditional on a pre-registered `GO`.

## Research Question

Can a validation-only policy that (i) excludes minority seeds with weak local support from synthetic interpolation and (ii) constrains majority-class and aggregate probability risk retain minority-risk improvements without reproducing the fixed-SMOTE risk cost observed in the main study?

The policy is designed as a decision rule around established resampling ideas. ADASYN allocates synthetic samples according to learning difficulty ([He et al., IEEE IJCNN 2008](https://ieeexplore.ieee.org/document/4633969)); Safe-Level-SMOTE uses local minority support to avoid outcast seeds ([Bunkhumpornpat et al., PAKDD 2009](https://doi.org/10.1007/978-3-642-01307-2_43)); guided resampling was proposed earlier as an alternative to blind class equalization ([Nickerson et al., AISTATS 2001](https://proceedings.mlr.press/r3/nickerson01a.html)); and Neyman-Pearson classification motivates optimizing one class under a constraint on the other ([Rigollet and Tong, JMLR 2011](https://jmlr.org/papers/volume12/rigollet11a/)). This extension does **not** claim a new interpolation primitive or a new label-shift estimator.

## Frozen Matrix

| Item | Frozen value |
|---|---|
| Gate tasks | `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer` |
| Full tasks | Existing frozen 24-task list from the #15 main protocol; run only after Gate `GO` |
| Learners | `dt_leaf01`, `cat_iter500` |
| Gate seeds | `0, 1, 2` |
| Full seeds | `0, 1, 2, 3, 4`, only after Gate `GO` |
| Candidate order | `none`, `random_q05`, `smote_q05_k5`, `support_smote_q05_k5` |
| Sampling ratio | minority-to-majority ratio `0.5` for fixed candidates and the gated candidate |
| Support score | fraction of positive labels among the five nearest non-self neighbors in the training fold |
| Support threshold | keep a minority seed when its score is at least `0.5` |
| Selector | choose the feasible candidate with lowest mean validation minority logloss; candidate order breaks exact ties |
| Majority constraint | validation majority logloss delta from `none` is at most `0.02` at every declared target prior |
| Aggregate constraint | validation Brier delta from `none` is at most `0.005` at every declared target prior |
| Target-prior grid | source prior, `0.05`, `0.20`, `0.50`; target-prior weights are used only for validation selection and held-out reporting |
| Primary metrics | minority logloss, majority logloss, Brier |
| Secondary metrics | AUC, overall logloss, 10-bin ECE |
| Resampling seed | fit seed, with the support sampler using the same deterministic seed |

## Data and Leakage Rules

1. The existing `fetch_datasets(..., download_if_missing=False)` loader is used. Before any data action, inspect `<local path omitted>`; if the already-used local Zenodo artifact is used, do not download a duplicate.
2. All support scores and synthetic points use only the training fold. Validation labels and predictions are used only by the policy selector. Test labels are used only for final metrics and label-shift importance weighting.
3. No threshold, target prior, support threshold, candidate, or Gate rule may be changed after observing held-out test metrics.
4. The main #15 outputs under `repro/15_elor_smote/code/outputs/` are immutable. This extension writes only under `repro/15_elor_smote/adaptive_resampling/results/`.

## Gate Rule

The Gate is `GO` only if all four conditions pass:

1. The adaptive selector is different from `none` in at least 4 of 6 task clusters.
2. Adaptive minus fixed SMOTE has a task-cluster 95% upper bootstrap bound at most `0` for majority logloss and at most `0.002` for Brier.
3. Adaptive minus `none` improves minority logloss by at least 5% in at least 4 of 6 tasks and has a task-cluster 95% upper bootstrap bound below zero.
4. All declared fits and metric rows are present, finite, hash-linked, leakage-clean, and generated under the unchanged protocol hash.

The bootstrap averages seeds within each task before resampling task clusters and uses 10,000 deterministic resamples. A non-`GO` result is retained as a bounded negative or inconclusive policy experiment; it does not unlock the full 24-task matrix and does not trigger post-hoc changes.

## Output Contract

The runner must write `raw_results.csv`, `selection_decisions.csv`, `manifest.json`, `data_manifest.json`, `status.json`, and an analysis directory containing `gate_summary.json`, `gate_effects.csv`, and `gate_report.md`. Every raw row must include task, learner, candidate/policy, seed, scenario, severity, metric, value, data hash, split hash, configuration hash, and runtime metadata.
