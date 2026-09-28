# Cost-sensitive XGBoost gate report

- Decision: `NO-GO`
- Fits: `120/120`
- Raw metric rows: `1080/1080`
- Integrity: `True`
- Candidate utility signal: `True`
- Candidate safety failures: `cost_weight_q05/brier`
- All treatment safety failures: `cost_weight_q05/brier`

Effects are treatment minus `none`; inference uses task-level paired bootstrap.

| method | metric | mean | 95% CI | tasks |
|---|---|---:|---:|---:|
| random_q05 | auc | -0.005370 | [-0.014958, 0.006773] | 6 |
| random_q05 | ap | -0.020158 | [-0.048130, 0.008957] | 6 |
| random_q05 | logloss | 0.014917 | [0.011341, 0.020307] | 6 |
| random_q05 | loss_minority | -0.247534 | [-0.355216, -0.133252] | 6 |
| random_q05 | loss_majority | 0.027450 | [0.017903, 0.039190] | 6 |
| random_q05 | brier | 0.003831 | [0.002516, 0.005230] | 6 |
| random_q05 | f1_fixed | 0.098096 | [0.029037, 0.182348] | 6 |
| random_q05 | f1_tuned | -0.041431 | [-0.084543, -0.006741] | 6 |
| random_q05 | decision_utility_c5 | -0.005905 | [-0.020143, 0.007147] | 6 |
| smote_q05_k5 | auc | 0.002598 | [-0.008161, 0.017572] | 6 |
| smote_q05_k5 | ap | -0.010990 | [-0.045611, 0.023317] | 6 |
| smote_q05_k5 | logloss | 0.015515 | [0.008233, 0.023060] | 6 |
| smote_q05_k5 | loss_minority | -0.510923 | [-0.653288, -0.373973] | 6 |
| smote_q05_k5 | loss_majority | 0.038399 | [0.024495, 0.056362] | 6 |
| smote_q05_k5 | brier | 0.004697 | [0.002695, 0.007104] | 6 |
| smote_q05_k5 | f1_fixed | 0.104848 | [0.024055, 0.193318] | 6 |
| smote_q05_k5 | f1_tuned | -0.015993 | [-0.037582, 0.006540] | 6 |
| smote_q05_k5 | decision_utility_c5 | -0.003936 | [-0.011788, 0.002154] | 6 |
| cost_weight_q05 | auc | -0.004802 | [-0.020038, 0.012104] | 6 |
| cost_weight_q05 | ap | -0.011471 | [-0.031271, 0.011998] | 6 |
| cost_weight_q05 | logloss | 0.025005 | [0.016879, 0.036121] | 6 |
| cost_weight_q05 | loss_minority | -0.309065 | [-0.456111, -0.161481] | 6 |
| cost_weight_q05 | loss_majority | 0.042609 | [0.024650, 0.067097] | 6 |
| cost_weight_q05 | brier | 0.006217 | [0.003830, 0.009606] | 6 |
| cost_weight_q05 | f1_fixed | 0.103985 | [0.026797, 0.193845] | 6 |
| cost_weight_q05 | f1_tuned | -0.031203 | [-0.057421, -0.002410] | 6 |
| cost_weight_q05 | decision_utility_c5 | -0.011898 | [-0.021987, -0.001952] | 6 |
