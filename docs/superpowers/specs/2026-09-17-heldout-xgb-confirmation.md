# Held-out XGBoost learner confirmation — protocol v0.1

Date: 2026-09-17  
Iteration: `AIEP-20260917-XGB-CONF-01`  
Unlock condition: `AIEP-20260917-XGB-GATE-01` integrity `GO`

## Purpose

Confirm the frozen XGBoost learner-family result on the existing 24-task
library. This is a confirmation of the Gate hypothesis, not a new search for a
favorable task subset and not a new learner hyperparameter sweep.

## Frozen extension

- Tasks: `abalone`, `abalone_19`, `arrhythmia`, `car_eval_34`, `car_eval_4`,
  `coil_2000`, `ecoli`, `letter_img`, `libras_move`, `mammography`, `oil`,
  `optical_digits`, `ozone_level`, `pen_digits`, `satimage`, `scene`,
  `sick_euthyroid`, `solar_flare_m0`, `spectrometer`, `thyroid_sick`,
  `us_crime`, `wine_quality`, `yeast_me2`, `yeast_ml8`.
- Learner: `xgb_gate` only, unchanged from the Gate.
- Treatments: `none`, `random_q05`, `smote_q05_k5`, unchanged from the Gate.
- Seeds: `0,1,2,3,4`, unchanged from the Gate.
- Environment: iid only, unchanged from the Gate.
- Expected fits/rows: `24 × 1 × 3 × 5 = 360` fits and `2,880` metric rows.
- Analysis: the same `analyze_heldout_xgb_gate.py`, task-level paired bootstrap,
  10,000 replicates, seed `20260917`; only `--tasks` and `--stage` change.
- No new tuning, no task removal, no endpoint change and no test-label access
  before final metric computation.

## Interpretation rule

The confirmation supports a broader conditional learner-family statement only
if integrity remains complete, no primary safety margin is exceeded, and the
effect pattern is not confined to the six Gate tasks. A change in sign,
substantial heterogeneity, or loss of the primary pattern is retained as a
generalization boundary. Neither outcome justifies a universal recommendation.

## Planned artifacts

- output: `code/outputs/heldout_xgb_full24_20260917/`
- analysis: `code/outputs/heldout_xgb_full24_20260917/analysis/`
- independent reanalysis: `analysis_recheck_20260917/`
