# Support-filtered SMOTE XGBoost gate report

- Decision: `GO`
- Fits: `120/120`
- Raw metric rows: `1080/1080`
- Integrity: `True`
- Candidate utility signal: `True`
- Candidate safety failures: `none`
- Diagnostics: `{'fallback_rate': 0.09166666666666666, 'mean_safe_fraction': 0.6925235556955988, 'mean_synthetic_rows': 215.625}`

| comparison | method | metric | mean | 95% CI | tasks |
|---|---|---|---:|---:|---:|
| treatment_minus_none | random_q05 | auc | -0.005370 | [-0.014958, 0.006773] | 6 |
| treatment_minus_none | random_q05 | ap | -0.020158 | [-0.048130, 0.008957] | 6 |
| treatment_minus_none | random_q05 | logloss | 0.014917 | [0.011341, 0.020307] | 6 |
| treatment_minus_none | random_q05 | loss_minority | -0.247534 | [-0.355216, -0.133252] | 6 |
| treatment_minus_none | random_q05 | loss_majority | 0.027450 | [0.017903, 0.039190] | 6 |
| treatment_minus_none | random_q05 | brier | 0.003831 | [0.002516, 0.005230] | 6 |
| treatment_minus_none | random_q05 | f1_fixed | 0.098096 | [0.029037, 0.182348] | 6 |
| treatment_minus_none | random_q05 | f1_tuned | -0.041431 | [-0.084543, -0.006741] | 6 |
| treatment_minus_none | random_q05 | decision_utility_c5 | -0.005905 | [-0.020143, 0.007147] | 6 |
| treatment_minus_none | smote_q05_k5 | auc | 0.002598 | [-0.008161, 0.017572] | 6 |
| treatment_minus_none | smote_q05_k5 | ap | -0.010990 | [-0.045611, 0.023317] | 6 |
| treatment_minus_none | smote_q05_k5 | logloss | 0.015515 | [0.008233, 0.023060] | 6 |
| treatment_minus_none | smote_q05_k5 | loss_minority | -0.510923 | [-0.653288, -0.373973] | 6 |
| treatment_minus_none | smote_q05_k5 | loss_majority | 0.038399 | [0.024495, 0.056362] | 6 |
| treatment_minus_none | smote_q05_k5 | brier | 0.004697 | [0.002695, 0.007104] | 6 |
| treatment_minus_none | smote_q05_k5 | f1_fixed | 0.104848 | [0.024055, 0.193318] | 6 |
| treatment_minus_none | smote_q05_k5 | f1_tuned | -0.015993 | [-0.037582, 0.006540] | 6 |
| treatment_minus_none | smote_q05_k5 | decision_utility_c5 | -0.003936 | [-0.011788, 0.002154] | 6 |
| treatment_minus_none | support_smote_q05 | auc | 0.004119 | [-0.002298, 0.012226] | 6 |
| treatment_minus_none | support_smote_q05 | ap | 0.007054 | [-0.020086, 0.035660] | 6 |
| treatment_minus_none | support_smote_q05 | logloss | 0.009010 | [0.002623, 0.015603] | 6 |
| treatment_minus_none | support_smote_q05 | loss_minority | -0.403397 | [-0.580976, -0.225306] | 6 |
| treatment_minus_none | support_smote_q05 | loss_majority | 0.025756 | [0.014630, 0.039518] | 6 |
| treatment_minus_none | support_smote_q05 | brier | 0.002291 | [0.000489, 0.004593] | 6 |
| treatment_minus_none | support_smote_q05 | f1_fixed | 0.105674 | [0.029019, 0.193522] | 6 |
| treatment_minus_none | support_smote_q05 | f1_tuned | 0.007806 | [-0.015821, 0.032307] | 6 |
| treatment_minus_none | support_smote_q05 | decision_utility_c5 | 0.007080 | [-0.002411, 0.016349] | 6 |
| support_smote_minus_smote | support_smote_q05 | auc | 0.001521 | [-0.006803, 0.009488] | 6 |
| support_smote_minus_smote | support_smote_q05 | ap | 0.018044 | [0.010284, 0.028774] | 6 |
| support_smote_minus_smote | support_smote_q05 | logloss | -0.006505 | [-0.013482, 0.000418] | 6 |
| support_smote_minus_smote | support_smote_q05 | loss_minority | 0.107526 | [0.023283, 0.184193] | 6 |
| support_smote_minus_smote | support_smote_q05 | loss_majority | -0.012643 | [-0.020075, -0.004905] | 6 |
| support_smote_minus_smote | support_smote_q05 | brier | -0.002405 | [-0.005020, -0.000259] | 6 |
| support_smote_minus_smote | support_smote_q05 | f1_fixed | 0.000825 | [-0.022349, 0.023556] | 6 |
| support_smote_minus_smote | support_smote_q05 | f1_tuned | 0.023799 | [-0.003533, 0.054208] | 6 |
| support_smote_minus_smote | support_smote_q05 | decision_utility_c5 | 0.011016 | [-0.000129, 0.025204] | 6 |
