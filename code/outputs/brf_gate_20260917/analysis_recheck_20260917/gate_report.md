# Balanced Random Forest learner-family gate report

- Decision: `NO-GO`
- BRF fits: `90/90`
- Matched XGBoost fits: `90/90`
- Integrity: `True`
- Candidate utility signal: `True`
- Candidate safety failures: `brf_minus_xgb/none/loss_majority, brf_minus_xgb/none/brier`

Effects in the first block are BRF minus matched XGBoost; the second block is BRF treatment minus BRF none.

| comparison | method | metric | mean | 95% CI | tasks |
|---|---|---|---:|---:|---:|
| brf_minus_xgb | none | auc | 0.012835 | [0.002849, 0.023588] | 6 |
| brf_minus_xgb | none | ap | -0.022524 | [-0.060621, 0.009275] | 6 |
| brf_minus_xgb | none | logloss | 0.171952 | [0.098099, 0.261187] | 6 |
| brf_minus_xgb | none | loss_minority | -1.490177 | [-2.101839, -0.979629] | 6 |
| brf_minus_xgb | none | loss_majority | 0.235594 | [0.173746, 0.308909] | 6 |
| brf_minus_xgb | none | brier | 0.052789 | [0.027973, 0.084480] | 6 |
| brf_minus_xgb | none | f1_fixed | 0.061307 | [-0.073629, 0.171376] | 6 |
| brf_minus_xgb | none | f1_tuned | -0.025764 | [-0.055821, -0.000232] | 6 |
| brf_minus_xgb | none | decision_utility_c5 | -0.001127 | [-0.015671, 0.012968] | 6 |
| brf_minus_xgb | random_q05 | auc | 0.011437 | [0.001514, 0.018409] | 6 |
| brf_minus_xgb | random_q05 | ap | 0.018064 | [-0.008046, 0.042024] | 6 |
| brf_minus_xgb | random_q05 | logloss | 0.005308 | [-0.011623, 0.020014] | 6 |
| brf_minus_xgb | random_q05 | loss_minority | 0.402776 | [-0.359480, 1.579176] | 6 |
| brf_minus_xgb | random_q05 | loss_majority | 0.016014 | [0.005324, 0.028583] | 6 |
| brf_minus_xgb | random_q05 | brier | -0.003045 | [-0.006686, -0.000358] | 6 |
| brf_minus_xgb | random_q05 | f1_fixed | 0.014958 | [-0.040594, 0.079530] | 6 |
| brf_minus_xgb | random_q05 | f1_tuned | 0.020205 | [-0.016235, 0.052804] | 6 |
| brf_minus_xgb | random_q05 | decision_utility_c5 | 0.007249 | [-0.001895, 0.018022] | 6 |
| brf_minus_xgb | smote_q05_k5 | auc | 0.009290 | [-0.002011, 0.016352] | 6 |
| brf_minus_xgb | smote_q05_k5 | ap | 0.022437 | [0.001050, 0.047773] | 6 |
| brf_minus_xgb | smote_q05_k5 | logloss | 0.022090 | [0.004655, 0.041900] | 6 |
| brf_minus_xgb | smote_q05_k5 | loss_minority | -0.158114 | [-0.470838, 0.134634] | 6 |
| brf_minus_xgb | smote_q05_k5 | loss_majority | 0.030820 | [0.021185, 0.044137] | 6 |
| brf_minus_xgb | smote_q05_k5 | brier | 0.002622 | [-0.003408, 0.008459] | 6 |
| brf_minus_xgb | smote_q05_k5 | f1_fixed | 0.035853 | [-0.017468, 0.101528] | 6 |
| brf_minus_xgb | smote_q05_k5 | f1_tuned | 0.013393 | [-0.012559, 0.042588] | 6 |
| brf_minus_xgb | smote_q05_k5 | decision_utility_c5 | 0.008472 | [0.000137, 0.016828] | 6 |
| brf_treatment_minus_none | random_q05 | auc | -0.006768 | [-0.016444, 0.002101] | 6 |
| brf_treatment_minus_none | random_q05 | ap | 0.020430 | [-0.017628, 0.055347] | 6 |
| brf_treatment_minus_none | random_q05 | logloss | -0.151727 | [-0.233669, -0.089891] | 6 |
| brf_treatment_minus_none | random_q05 | loss_minority | 1.645420 | [0.486364, 3.470406] | 6 |
| brf_treatment_minus_none | random_q05 | loss_majority | -0.192130 | [-0.279028, -0.126811] | 6 |
| brf_treatment_minus_none | random_q05 | brier | -0.052004 | [-0.082852, -0.028987] | 6 |
| brf_treatment_minus_none | random_q05 | f1_fixed | 0.051747 | [-0.012680, 0.133146] | 6 |
| brf_treatment_minus_none | random_q05 | f1_tuned | 0.004538 | [-0.015347, 0.019620] | 6 |
| brf_treatment_minus_none | random_q05 | decision_utility_c5 | 0.002472 | [-0.007031, 0.013880] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | auc | -0.000948 | [-0.010121, 0.009905] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | ap | 0.033971 | [-0.010404, 0.073385] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | logloss | -0.134347 | [-0.212369, -0.065097] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | loss_minority | 0.821140 | [0.396461, 1.470070] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | loss_majority | -0.166375 | [-0.235673, -0.106706] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | brier | -0.045470 | [-0.071834, -0.025082] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | f1_fixed | 0.079395 | [0.042016, 0.121902] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | f1_tuned | 0.023164 | [-0.014191, 0.069110] | 6 |
| brf_treatment_minus_none | smote_q05_k5 | decision_utility_c5 | 0.005664 | [-0.004708, 0.022726] | 6 |
