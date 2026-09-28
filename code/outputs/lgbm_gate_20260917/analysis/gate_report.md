# LightGBM learner-family gate report

- Decision: `NO-GO`
- LightGBM fits: `90/90`
- Matched XGBoost fits: `90/90`
- Integrity: `True`
- Candidate utility signal: `False`
- Candidate safety failures: `none`

Effects are LightGBM minus matched XGBoost; treatment interactions follow.

| comparison | method | metric | mean | 95% CI | tasks |
|---|---|---|---:|---:|---:|
| lgbm_minus_xgb | none | auc | -0.043001 | [-0.127214, 0.007176] | 6 |
| lgbm_minus_xgb | none | ap | -0.006930 | [-0.020083, 0.005465] | 6 |
| lgbm_minus_xgb | none | logloss | -0.001619 | [-0.003715, 0.000410] | 6 |
| lgbm_minus_xgb | none | loss_minority | -0.022532 | [-0.108194, 0.080668] | 6 |
| lgbm_minus_xgb | none | loss_majority | 0.002187 | [-0.000120, 0.004609] | 6 |
| lgbm_minus_xgb | none | brier | -0.000258 | [-0.001328, 0.000946] | 6 |
| lgbm_minus_xgb | none | f1_fixed | 0.002393 | [-0.036792, 0.046387] | 6 |
| lgbm_minus_xgb | none | f1_tuned | -0.000793 | [-0.014706, 0.013119] | 6 |
| lgbm_minus_xgb | none | decision_utility_c5 | 0.002314 | [-0.001274, 0.006428] | 6 |
| lgbm_minus_xgb | random_q05 | auc | -0.036632 | [-0.084320, -0.001269] | 6 |
| lgbm_minus_xgb | random_q05 | ap | 0.019150 | [-0.003167, 0.043967] | 6 |
| lgbm_minus_xgb | random_q05 | logloss | -0.001763 | [-0.002967, -0.000390] | 6 |
| lgbm_minus_xgb | random_q05 | loss_minority | -0.042439 | [-0.118058, 0.016636] | 6 |
| lgbm_minus_xgb | random_q05 | loss_majority | 0.002538 | [-0.002195, 0.010691] | 6 |
| lgbm_minus_xgb | random_q05 | brier | -0.001092 | [-0.001703, -0.000516] | 6 |
| lgbm_minus_xgb | random_q05 | f1_fixed | 0.018118 | [-0.003619, 0.040412] | 6 |
| lgbm_minus_xgb | random_q05 | f1_tuned | 0.028926 | [-0.009320, 0.074764] | 6 |
| lgbm_minus_xgb | random_q05 | decision_utility_c5 | 0.001206 | [-0.021614, 0.018968] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | auc | -0.001036 | [-0.002931, 0.000860] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | ap | 0.002115 | [-0.029515, 0.028025] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | logloss | 0.002106 | [-0.000713, 0.004653] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | loss_minority | 0.071833 | [-0.055674, 0.192220] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | loss_majority | 0.000352 | [-0.008045, 0.008029] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | brier | -0.000425 | [-0.001792, 0.000828] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | f1_fixed | 0.017061 | [-0.023987, 0.053117] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | f1_tuned | 0.005985 | [-0.018471, 0.031998] | 6 |
| lgbm_minus_xgb | smote_q05_k5 | decision_utility_c5 | -0.000021 | [-0.008808, 0.005962] | 6 |
| lgbm_treatment_minus_none | random_q05 | auc | 0.000999 | [-0.029783, 0.050851] | 6 |
| lgbm_treatment_minus_none | random_q05 | ap | 0.005921 | [-0.007610, 0.019079] | 6 |
| lgbm_treatment_minus_none | random_q05 | logloss | 0.014774 | [0.009725, 0.022010] | 6 |
| lgbm_treatment_minus_none | random_q05 | loss_minority | -0.267440 | [-0.352297, -0.157109] | 6 |
| lgbm_treatment_minus_none | random_q05 | loss_majority | 0.027801 | [0.015213, 0.046417] | 6 |
| lgbm_treatment_minus_none | random_q05 | brier | 0.002996 | [0.001291, 0.004984] | 6 |
| lgbm_treatment_minus_none | random_q05 | f1_fixed | 0.113820 | [0.042858, 0.189361] | 6 |
| lgbm_treatment_minus_none | random_q05 | f1_tuned | -0.011712 | [-0.044450, 0.019721] | 6 |
| lgbm_treatment_minus_none | random_q05 | decision_utility_c5 | -0.007014 | [-0.021286, 0.002454] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | auc | 0.044562 | [-0.012549, 0.142929] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | ap | -0.001945 | [-0.027795, 0.017958] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | logloss | 0.019240 | [0.011979, 0.027438] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | loss_minority | -0.416558 | [-0.600879, -0.188812] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | loss_majority | 0.036564 | [0.017656, 0.061667] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | brier | 0.004530 | [0.001460, 0.007451] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | f1_fixed | 0.119516 | [0.061445, 0.178285] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | f1_tuned | -0.009215 | [-0.036589, 0.015175] | 6 |
| lgbm_treatment_minus_none | smote_q05_k5 | decision_utility_c5 | -0.006272 | [-0.015561, 0.001013] | 6 |
