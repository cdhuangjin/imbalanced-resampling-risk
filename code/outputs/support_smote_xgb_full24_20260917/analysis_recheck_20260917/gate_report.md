# Support-filtered SMOTE XGBoost confirmation report

- Decision: `NO-GO`
- Fits: `480/480`
- Raw metric rows: `4320/4320`
- Integrity: `True`
- Candidate utility signal: `True`
- Candidate safety failures: `support_smote_q05/brier`
- Diagnostics: `{'fallback_rate': 0.05, 'mean_safe_fraction': 0.5162122444458138, 'mean_synthetic_rows': 264.09583333333336}`

| comparison | method | metric | mean | 95% CI | tasks |
|---|---|---|---:|---:|---:|
| treatment_minus_none | random_q05 | auc | -0.005906 | [-0.014314, 0.000976] | 24 |
| treatment_minus_none | random_q05 | ap | -0.005225 | [-0.015946, 0.004908] | 24 |
| treatment_minus_none | random_q05 | logloss | 0.017152 | [0.007310, 0.029964] | 24 |
| treatment_minus_none | random_q05 | loss_minority | -0.263342 | [-0.350341, -0.187149] | 24 |
| treatment_minus_none | random_q05 | loss_majority | 0.034646 | [0.020269, 0.052861] | 24 |
| treatment_minus_none | random_q05 | brier | 0.005774 | [0.002176, 0.010460] | 24 |
| treatment_minus_none | random_q05 | f1_fixed | 0.091113 | [0.055152, 0.130808] | 24 |
| treatment_minus_none | random_q05 | f1_tuned | -0.015962 | [-0.033354, -0.001475] | 24 |
| treatment_minus_none | random_q05 | decision_utility_c5 | -0.003097 | [-0.008012, 0.001374] | 24 |
| treatment_minus_none | smote_q05_k5 | auc | 0.000293 | [-0.003775, 0.005058] | 24 |
| treatment_minus_none | smote_q05_k5 | ap | -0.004362 | [-0.015635, 0.006442] | 24 |
| treatment_minus_none | smote_q05_k5 | logloss | 0.011325 | [0.006143, 0.016681] | 24 |
| treatment_minus_none | smote_q05_k5 | loss_minority | -0.333697 | [-0.424015, -0.244469] | 24 |
| treatment_minus_none | smote_q05_k5 | loss_majority | 0.031764 | [0.022162, 0.041980] | 24 |
| treatment_minus_none | smote_q05_k5 | brier | 0.003897 | [0.001898, 0.005944] | 24 |
| treatment_minus_none | smote_q05_k5 | f1_fixed | 0.095194 | [0.058286, 0.135362] | 24 |
| treatment_minus_none | smote_q05_k5 | f1_tuned | -0.011704 | [-0.024311, 0.001353] | 24 |
| treatment_minus_none | smote_q05_k5 | decision_utility_c5 | -0.003702 | [-0.007437, -0.000112] | 24 |
| treatment_minus_none | support_smote_q05 | auc | -0.000156 | [-0.002903, 0.002850] | 24 |
| treatment_minus_none | support_smote_q05 | ap | 0.000560 | [-0.014193, 0.013906] | 24 |
| treatment_minus_none | support_smote_q05 | logloss | 0.006860 | [0.003736, 0.010426] | 24 |
| treatment_minus_none | support_smote_q05 | loss_minority | -0.207149 | [-0.301843, -0.117609] | 24 |
| treatment_minus_none | support_smote_q05 | loss_majority | 0.017622 | [0.011697, 0.024106] | 24 |
| treatment_minus_none | support_smote_q05 | brier | 0.002010 | [0.000885, 0.003318] | 24 |
| treatment_minus_none | support_smote_q05 | f1_fixed | 0.072213 | [0.034378, 0.114158] | 24 |
| treatment_minus_none | support_smote_q05 | f1_tuned | -0.007513 | [-0.034302, 0.013453] | 24 |
| treatment_minus_none | support_smote_q05 | decision_utility_c5 | -0.000058 | [-0.007209, 0.005673] | 24 |
| support_smote_minus_smote | support_smote_q05 | auc | -0.000449 | [-0.003543, 0.002575] | 24 |
| support_smote_minus_smote | support_smote_q05 | ap | 0.004922 | [-0.008839, 0.016387] | 24 |
| support_smote_minus_smote | support_smote_q05 | logloss | -0.004464 | [-0.009073, 0.000452] | 24 |
| support_smote_minus_smote | support_smote_q05 | loss_minority | 0.126549 | [0.057324, 0.207996] | 24 |
| support_smote_minus_smote | support_smote_q05 | loss_majority | -0.014142 | [-0.020785, -0.008313] | 24 |
| support_smote_minus_smote | support_smote_q05 | brier | -0.001888 | [-0.003586, -0.000132] | 24 |
| support_smote_minus_smote | support_smote_q05 | f1_fixed | -0.022981 | [-0.048048, -0.002917] | 24 |
| support_smote_minus_smote | support_smote_q05 | f1_tuned | 0.004191 | [-0.018298, 0.021733] | 24 |
| support_smote_minus_smote | support_smote_q05 | decision_utility_c5 | 0.003643 | [-0.002861, 0.009698] | 24 |
