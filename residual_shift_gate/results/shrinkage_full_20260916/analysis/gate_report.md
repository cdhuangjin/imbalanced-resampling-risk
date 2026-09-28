# Residual-shift Gate report

Decision: **GO**
Input complete: `True`; effect rows: `288`; behavior rows: `9`.

Negative deltas mean the candidate improves the metric relative to its reference.

## Gate criteria

```json
{
  "utility_pass_methods": [
    "none",
    "random_q05",
    "smote_q05_k5"
  ],
  "mixed_practical_safety_pass_methods": [
    "none",
    "random_q05",
    "smote_q05_k5"
  ],
  "shrinkage_vs_binary_mean_pass_methods": [
    "none",
    "random_q05",
    "smote_q05_k5"
  ],
  "utility_pass": true,
  "mixed_safety_pass": true,
  "shrinkage_vs_binary_pass": true,
  "practical_brier_margin": 0.01,
  "practical_logloss_margin": 0.1
}
```

## Primary effects

| Candidate | Reference | Scenario | Prior | Method | Metric | Mean delta | 95% CI |
|---|---|---|---:|---|---|---:|---:|
| platt_target_consensus | platt_source | label_shift | 0.05 | none | brier | 0.00137806 | [0.000327684, 0.00249351] |
| platt_target_consensus | platt_source | label_shift | 0.05 | none | logloss | 0.0177807 | [0.00745165, 0.0295808] |
| platt_target_consensus | platt_source | label_shift | 0.05 | random_q05 | brier | 0.000879633 | [1.77143e-05, 0.00194766] |
| platt_target_consensus | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.0151251 | [0.00500774, 0.0271267] |
| platt_target_consensus | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.00129632 | [0.000317669, 0.00234085] |
| platt_target_consensus | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0278648 | [0.0113417, 0.046282] |
| platt_target_consensus | platt_source | label_shift | 0.2 | none | brier | -0.0121122 | [-0.0170814, -0.00754518] |
| platt_target_consensus | platt_source | label_shift | 0.2 | none | logloss | -0.0420049 | [-0.0627577, -0.0240919] |
| platt_target_consensus | platt_source | label_shift | 0.2 | random_q05 | brier | -0.00939949 | [-0.0130901, -0.00594678] |
| platt_target_consensus | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0350579 | [-0.0527705, -0.019155] |
| platt_target_consensus | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.010989 | [-0.0148255, -0.00747853] |
| platt_target_consensus | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0390128 | [-0.056135, -0.0235674] |
| platt_target_consensus | platt_source | label_shift | 0.5 | none | brier | -0.0669224 | [-0.089033, -0.0463618] |
| platt_target_consensus | platt_source | label_shift | 0.5 | none | logloss | -0.208424 | [-0.301566, -0.104747] |
| platt_target_consensus | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0574488 | [-0.0739154, -0.0424089] |
| platt_target_consensus | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.205064 | [-0.262157, -0.150241] |
| platt_target_consensus | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0617443 | [-0.0770408, -0.0473539] |
| platt_target_consensus | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.208231 | [-0.26055, -0.157309] |
| platt_target_gated | platt_source | label_shift | 0.05 | none | brier | 0.000716308 | [-3.60963e-05, 0.00150445] |
| platt_target_gated | platt_source | label_shift | 0.05 | none | logloss | 0.014543 | [0.00483597, 0.0256749] |
| platt_target_gated | platt_source | label_shift | 0.05 | random_q05 | brier | 0.000327527 | [-0.000247688, 0.00103637] |
| platt_target_gated | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.00978177 | [0.00177093, 0.0197409] |
| platt_target_gated | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.000652538 | [-3.45734e-05, 0.00143015] |
| platt_target_gated | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0178759 | [0.00546373, 0.0326088] |
| platt_target_gated | platt_source | label_shift | 0.2 | none | brier | -0.0107868 | [-0.0158253, -0.00637707] |
| platt_target_gated | platt_source | label_shift | 0.2 | none | logloss | -0.0370699 | [-0.0564638, -0.0197269] |
| platt_target_gated | platt_source | label_shift | 0.2 | random_q05 | brier | -0.00789612 | [-0.0114753, -0.0046985] |
| platt_target_gated | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0294679 | [-0.0460961, -0.0153084] |
| platt_target_gated | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.00905671 | [-0.0128705, -0.00563416] |
| platt_target_gated | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0319268 | [-0.0481392, -0.0175409] |
| platt_target_gated | platt_source | label_shift | 0.5 | none | brier | -0.0444899 | [-0.0632083, -0.0268011] |
| platt_target_gated | platt_source | label_shift | 0.5 | none | logloss | -0.143702 | [-0.209125, -0.085677] |
| platt_target_gated | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0355519 | [-0.0517153, -0.021328] |
| platt_target_gated | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.119301 | [-0.173567, -0.073846] |
| platt_target_gated | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0430722 | [-0.0606008, -0.0282001] |
| platt_target_gated | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.138923 | [-0.196062, -0.0901282] |
| platt_target_oracle | platt_source | label_shift | 0.05 | none | brier | -0.000825455 | [-0.00125253, -0.000423113] |
| platt_target_oracle | platt_source | label_shift | 0.05 | none | logloss | -0.00539456 | [-0.00967692, -0.00259894] |
| platt_target_oracle | platt_source | label_shift | 0.05 | random_q05 | brier | -0.000742781 | [-0.00116283, -0.000365101] |
| platt_target_oracle | platt_source | label_shift | 0.05 | random_q05 | logloss | -0.00511386 | [-0.00949294, -0.00226472] |
| platt_target_oracle | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | -0.000778986 | [-0.00118763, -0.000412806] |
| platt_target_oracle | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | -0.00515553 | [-0.00941931, -0.00236921] |
| platt_target_oracle | platt_source | label_shift | 0.2 | none | brier | -0.0201785 | [-0.0259987, -0.0149135] |
| platt_target_oracle | platt_source | label_shift | 0.2 | none | logloss | -0.0849502 | [-0.127665, -0.0530628] |
| platt_target_oracle | platt_source | label_shift | 0.2 | random_q05 | brier | -0.0180495 | [-0.0234409, -0.0132095] |
| platt_target_oracle | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0827088 | [-0.126141, -0.0500963] |
| platt_target_oracle | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.019156 | [-0.0251017, -0.0137652] |
| platt_target_oracle | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0824561 | [-0.12498, -0.0504108] |
| platt_target_oracle | platt_source | label_shift | 0.5 | none | brier | -0.126004 | [-0.155371, -0.0971473] |
| platt_target_oracle | platt_source | label_shift | 0.5 | none | logloss | -0.458789 | [-0.607565, -0.333716] |
| platt_target_oracle | platt_source | label_shift | 0.5 | random_q05 | brier | -0.118526 | [-0.148069, -0.0899892] |
| platt_target_oracle | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.45135 | [-0.602437, -0.319788] |
| platt_target_oracle | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.122699 | [-0.152684, -0.0935735] |
| platt_target_oracle | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.447063 | [-0.594324, -0.315982] |
| platt_target_shrinkage | platt_source | covariate_shift |  | none | brier | 0.000290592 | [-3.95389e-07, 0.000811729] |
| platt_target_shrinkage | platt_source | covariate_shift |  | none | logloss | 0.00106054 | [6.61924e-07, 0.0028527] |
| platt_target_shrinkage | platt_source | covariate_shift |  | random_q05 | brier | 0.000111951 | [-1.18562e-08, 0.000329549] |
| platt_target_shrinkage | platt_source | covariate_shift |  | random_q05 | logloss | 0.000419876 | [5.73416e-07, 0.00121679] |
| platt_target_shrinkage | platt_source | covariate_shift |  | smote_q05_k5 | brier | 0.000303631 | [3.15786e-08, 0.000783096] |
| platt_target_shrinkage | platt_source | covariate_shift |  | smote_q05_k5 | logloss | 0.00113484 | [8.6228e-07, 0.00272773] |
| platt_target_shrinkage | platt_source | joint_shift |  | none | brier | -0.000395839 | [-0.00108286, 7.97645e-05] |
| platt_target_shrinkage | platt_source | joint_shift |  | none | logloss | -0.00118098 | [-0.00333682, 0.000284074] |
| platt_target_shrinkage | platt_source | joint_shift |  | random_q05 | brier | -5.68401e-05 | [-0.000285242, 0.000199429] |
| platt_target_shrinkage | platt_source | joint_shift |  | random_q05 | logloss | -0.00012455 | [-0.00108466, 0.000999398] |
| platt_target_shrinkage | platt_source | joint_shift |  | smote_q05_k5 | brier | -0.000123318 | [-0.000506853, 0.000191172] |
| platt_target_shrinkage | platt_source | joint_shift |  | smote_q05_k5 | logloss | -0.00036184 | [-0.00112473, 0.000386743] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | none | brier | 0.000717146 | [-2.84359e-05, 0.00150776] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | none | logloss | 0.0116373 | [0.00339118, 0.0215233] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | random_q05 | brier | 0.000317058 | [-0.000226545, 0.000953306] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.0097124 | [0.00166233, 0.0195009] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.000681275 | [3.39056e-06, 0.00147324] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0135895 | [0.0029396, 0.0272106] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | none | brier | -0.0122407 | [-0.0170369, -0.0080104] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | none | logloss | -0.043109 | [-0.0637048, -0.0258574] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | random_q05 | brier | -0.00927629 | [-0.0129673, -0.00609027] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0355846 | [-0.0527114, -0.0205788] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.0105853 | [-0.0143954, -0.0072117] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.038279 | [-0.0555294, -0.0234112] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | none | brier | -0.0615262 | [-0.0813008, -0.0425159] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | none | logloss | -0.211913 | [-0.285414, -0.143941] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0490006 | [-0.0651536, -0.0343462] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.17583 | [-0.230667, -0.123421] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0534251 | [-0.0697339, -0.038559] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.177802 | [-0.23288, -0.127149] |
| platt_target_shrinkage | platt_source | mixed_shift |  | none | brier | -0.000167029 | [-0.000629198, 0.000256166] |
| platt_target_shrinkage | platt_source | mixed_shift |  | none | logloss | -0.000433803 | [-0.00184784, 0.000839088] |
| platt_target_shrinkage | platt_source | mixed_shift |  | random_q05 | brier | -5.76559e-07 | [-0.000186233, 0.000239009] |
| platt_target_shrinkage | platt_source | mixed_shift |  | random_q05 | logloss | 5.69252e-05 | [-0.000703751, 0.00111921] |
| platt_target_shrinkage | platt_source | mixed_shift |  | smote_q05_k5 | brier | 1.89985e-05 | [-0.000294922, 0.000357187] |
| platt_target_shrinkage | platt_source | mixed_shift |  | smote_q05_k5 | logloss | 0.000137055 | [-0.000587928, 0.00100553] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | none | brier | -0.00616311 | [-0.00966372, -0.00296558] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | none | logloss | -0.0257184 | [-0.0407532, -0.0121626] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | random_q05 | brier | -0.00494644 | [-0.00873842, -0.00200326] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | random_q05 | logloss | -0.020905 | [-0.0349244, -0.00913235] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | smote_q05_k5 | brier | -0.00395091 | [-0.00616895, -0.00197851] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | smote_q05_k5 | logloss | -0.0165058 | [-0.0259189, -0.00836095] |

## Gate behavior

| method | scenario_family | cells | acceptance_rate | residual_auc_mean | residual_auc_median | rejection_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | iid | 240 | 0.8583333333333333 | 0.552987505267783 | 0.530987610497551 | 0.14166666666666672 | 0.6 |
| none | mixed_shift | 240 | 0.001388888888888889 | 0.9543411963480172 | 0.9906538267269676 | 0.9986111111111111 | 0.6 |
| none | pure_label_shift | 240 | 0.7152777777777778 | 0.5756493540126073 | 0.546579912488633 | 0.2847222222222222 | 0.6 |
| random_q05 | iid | 240 | 0.85 | 0.5527033391590079 | 0.5295775266286791 | 0.15000000000000002 | 0.6 |
| random_q05 | mixed_shift | 240 | 0.001388888888888889 | 0.9551125433686405 | 0.9907441987065431 | 0.9986111111111111 | 0.6 |
| random_q05 | pure_label_shift | 240 | 0.7069444444444445 | 0.5764499327124865 | 0.547839663199226 | 0.2930555555555555 | 0.6 |
| smote_q05_k5 | iid | 240 | 0.8458333333333333 | 0.5534950398477229 | 0.5303557918215465 | 0.15416666666666667 | 0.6 |
| smote_q05_k5 | mixed_shift | 240 | 0.001388888888888889 | 0.9554373744454725 | 0.9906772624963195 | 0.9986111111111111 | 0.6 |
| smote_q05_k5 | pure_label_shift | 240 | 0.7083333333333334 | 0.576349293324305 | 0.5455789925143725 | 0.29166666666666663 | 0.6 |
