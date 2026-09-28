# Held-out XGBoost learner confirmation report

- Decision: `NO-GO`
- Fits: `360/360`
- Raw metric rows: `2880/2880`
- Integrity: `True`
- Utility signal: `True`
- Safety failures: `random_q05/loss_majority, random_q05/brier, smote_q05_k5/brier`

Effects are treatment minus `none`; inference uses task-level paired bootstrap.

| method | metric | mean | 95% CI | tasks |
|---|---|---:|---:|---:|
| random_q05 | auc | -0.005906 | [-0.014314, 0.000976] | 24 |
| random_q05 | loss_minority | -0.263342 | [-0.350341, -0.187149] | 24 |
| random_q05 | loss_majority | 0.034646 | [0.020269, 0.052861] | 24 |
| random_q05 | brier | 0.005774 | [0.002176, 0.010460] | 24 |
| random_q05 | ap | -0.005225 | [-0.015946, 0.004908] | 24 |
| random_q05 | logloss | 0.017152 | [0.007310, 0.029964] | 24 |
| random_q05 | f1_fixed | 0.091113 | [0.055152, 0.130808] | 24 |
| random_q05 | f1_tuned | -0.015962 | [-0.033354, -0.001475] | 24 |
| smote_q05_k5 | auc | 0.000293 | [-0.003775, 0.005058] | 24 |
| smote_q05_k5 | loss_minority | -0.333697 | [-0.424015, -0.244469] | 24 |
| smote_q05_k5 | loss_majority | 0.031764 | [0.022162, 0.041980] | 24 |
| smote_q05_k5 | brier | 0.003897 | [0.001898, 0.005944] | 24 |
| smote_q05_k5 | ap | -0.004362 | [-0.015635, 0.006442] | 24 |
| smote_q05_k5 | logloss | 0.011325 | [0.006143, 0.016681] | 24 |
| smote_q05_k5 | f1_fixed | 0.095194 | [0.058286, 0.135362] | 24 |
| smote_q05_k5 | f1_tuned | -0.011704 | [-0.024311, 0.001353] | 24 |
