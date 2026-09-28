# Support-filtered SMOTE policy Gate — protocol v0.1

## Purpose

The cross-learner synthesis found a repeatable SMOTE minority-logloss gain but
also a Brier/majority-risk cost. The next candidate tests whether a training-only
local-support filter can reduce that cost without changing the learner,
test-set evaluation, or target labels.

This is a policy experiment, not an unqualified novelty claim. Safe-Level-SMOTE,
SMOTE-IPF, RCSMOTE, and recent filtering-based methods already establish that
neighbourhood/overlap-aware filtering is a populated research area. The narrow
question here is whether one transparent, fixed support rule improves the
specific risk trade-off observed in this project under the same Elor protocol.

## Hypothesis card

- **Observation:** q05/k5 SMOTE lowers minority logloss across the cross-learner
  synthesis but increases majority logloss and Brier.
- **Question:** Can filtering synthetic-seed points by training-only local
  minority support retain the minority-risk gain while reducing probability-risk
  harm?
- **Candidate:** keep all observed rows, but generate new SMOTE points only from
  minority seeds whose `k=5` nearest-neighbour minority fraction is at least
  `0.5`; choose the neighbour from the same safe pool.
- **Rivals:** standard SMOTE is already adequate; filtering removes useful
  boundary information; the apparent harm is learner/calibration-specific.
- **Primary endpoints:** XGBoost task-level paired effects on Brier,
  majority logloss, minority logloss, and decision utility `c_FN=5,c_FP=1`.
- **Secondary endpoints:** AUC, AP, overall logloss, F1, safe-seed fraction,
  runtime, and worst-task effects.
- **Failure boundary:** Gate NO-GO if no candidate utility/risk signal appears,
  if either safety margin fails, or if safe-seed fallback is frequent enough to
  make the policy indistinguishable from ordinary SMOTE.

## Frozen implementation

- Learner: prior `xgb_gate` (`n_estimators=1000`, `learning_rate=0.1`,
  validation early stopping 10, `tree_method=hist`, `n_jobs=1`).
- Tasks: six Gate tasks; iid only; seeds `0..4`.
- Treatments: `none`, `random_q05`, `smote_q05_k5`, `support_smote_q05`.
- Target training ratio: `q=0.5`.
- Neighbourhood: Euclidean distance on the unchanged training feature matrix;
  `k=5`; support threshold `0.5`; ties use deterministic RNG seeded by the run.
- All original training rows remain. Only the synthetic rows are filtered.
- Thresholds: validation-only 0.01 grid; utility threshold minimizes
  `5*FN+1*FP`, lowest threshold wins ties.
- Inference: task-level paired bootstrap, 10,000 replicates, seed `20260917`.
- Safety margins: Brier increase `<=0.01`; majority-logloss increase `<=0.10`.

## Gate rule

Require exact four-treatment completeness and finite, provenance-hashed rows.
The candidate passes only if it has either a positive utility CI or a minority
logloss improvement with utility no worse than `-0.02`, and both practical
safety margins hold on every Gate task. A result that merely improves AUC is
not sufficient.

## Literature boundary

Filtering/overlap-aware SMOTE is not an empty space. Safe-Level-SMOTE,
SMOTE-IPF, RCSMOTE, CFS-SMOTE, and instance-hardness filtering are treated as
prior art. If this Gate passes, the defensible contribution remains a
transparent, reproducible risk-control policy under this evaluation protocol;
if it fails, retain the negative boundary and do not rename it as a new method.

