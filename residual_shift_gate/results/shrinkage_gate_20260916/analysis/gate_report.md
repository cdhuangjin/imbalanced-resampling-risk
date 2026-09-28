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
| platt_target_consensus | platt_source | label_shift | 0.05 | none | brier | 0.00191132 | [-0.000740266, 0.00508074] |
| platt_target_consensus | platt_source | label_shift | 0.05 | none | logloss | 0.00651988 | [-5.2563e-05, 0.0126641] |
| platt_target_consensus | platt_source | label_shift | 0.05 | random_q05 | brier | 0.00131066 | [-0.000295618, 0.0029812] |
| platt_target_consensus | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.00556338 | [-0.0010695, 0.0129147] |
| platt_target_consensus | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.00183975 | [-0.000345165, 0.00409315] |
| platt_target_consensus | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0430813 | [0.000325245, 0.0925069] |
| platt_target_consensus | platt_source | label_shift | 0.2 | none | brier | -0.0156694 | [-0.0269635, -0.00531202] |
| platt_target_consensus | platt_source | label_shift | 0.2 | none | logloss | -0.0531667 | [-0.0955746, -0.0136847] |
| platt_target_consensus | platt_source | label_shift | 0.2 | random_q05 | brier | -0.0125506 | [-0.0198565, -0.00565983] |
| platt_target_consensus | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0525168 | [-0.0895775, -0.0193162] |
| platt_target_consensus | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.0142915 | [-0.024188, -0.00526375] |
| platt_target_consensus | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0552232 | [-0.0955552, -0.0178119] |
| platt_target_consensus | platt_source | label_shift | 0.5 | none | brier | -0.0431425 | [-0.0916749, -0.00497783] |
| platt_target_consensus | platt_source | label_shift | 0.5 | none | logloss | -0.0292365 | [-0.337376, 0.40339] |
| platt_target_consensus | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0534165 | [-0.0757657, -0.029894] |
| platt_target_consensus | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.231022 | [-0.333078, -0.122331] |
| platt_target_consensus | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0737056 | [-0.10603, -0.0372827] |
| platt_target_consensus | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.272768 | [-0.409337, -0.147358] |
| platt_target_gated | platt_source | label_shift | 0.05 | none | brier | 0.00115397 | [-0.00112055, 0.00445292] |
| platt_target_gated | platt_source | label_shift | 0.05 | none | logloss | 0.00432053 | [-0.00165895, 0.0104301] |
| platt_target_gated | platt_source | label_shift | 0.05 | random_q05 | brier | 0.000472832 | [-0.000572534, 0.00153682] |
| platt_target_gated | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.00204196 | [-0.00235203, 0.00769569] |
| platt_target_gated | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.000971247 | [-0.00075329, 0.00308767] |
| platt_target_gated | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0229764 | [-0.00144838, 0.0586367] |
| platt_target_gated | platt_source | label_shift | 0.2 | none | brier | -0.0133537 | [-0.0263269, -0.00220758] |
| platt_target_gated | platt_source | label_shift | 0.2 | none | logloss | -0.0400428 | [-0.0888193, 0.00200442] |
| platt_target_gated | platt_source | label_shift | 0.2 | random_q05 | brier | -0.00971 | [-0.0185726, -0.00178614] |
| platt_target_gated | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0367988 | [-0.0792168, -0.00229643] |
| platt_target_gated | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.010744 | [-0.0208548, -0.00204219] |
| platt_target_gated | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0385388 | [-0.0800947, -0.00295168] |
| platt_target_gated | platt_source | label_shift | 0.5 | none | brier | -0.0346802 | [-0.0848027, -0.00342102] |
| platt_target_gated | platt_source | label_shift | 0.5 | none | logloss | -0.137833 | [-0.327217, -0.0210733] |
| platt_target_gated | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0352646 | [-0.0652284, -0.0091422] |
| platt_target_gated | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.139878 | [-0.283868, -0.0313307] |
| platt_target_gated | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0520516 | [-0.0911287, -0.0156287] |
| platt_target_gated | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.182975 | [-0.338317, -0.0573948] |
| platt_target_oracle | platt_source | label_shift | 0.05 | none | brier | -0.000628274 | [-0.00203517, 0.000772961] |
| platt_target_oracle | platt_source | label_shift | 0.05 | none | logloss | -0.0110713 | [-0.0261489, -0.00227758] |
| platt_target_oracle | platt_source | label_shift | 0.05 | random_q05 | brier | -0.000988117 | [-0.0021236, -4.71792e-05] |
| platt_target_oracle | platt_source | label_shift | 0.05 | random_q05 | logloss | -0.0111757 | [-0.0259849, -0.00227919] |
| platt_target_oracle | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | -0.00113207 | [-0.00191688, -0.000313916] |
| platt_target_oracle | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | -0.0114218 | [-0.0258908, -0.00283452] |
| platt_target_oracle | platt_source | label_shift | 0.2 | none | brier | -0.0303835 | [-0.0425161, -0.0180339] |
| platt_target_oracle | platt_source | label_shift | 0.2 | none | logloss | -0.152239 | [-0.278229, -0.0599152] |
| platt_target_oracle | platt_source | label_shift | 0.2 | random_q05 | brier | -0.02833 | [-0.0406088, -0.0166938] |
| platt_target_oracle | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.152347 | [-0.275573, -0.0621125] |
| platt_target_oracle | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.0298986 | [-0.0426727, -0.0171244] |
| platt_target_oracle | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.149762 | [-0.273527, -0.0608144] |
| platt_target_oracle | platt_source | label_shift | 0.5 | none | brier | -0.152897 | [-0.212379, -0.0963579] |
| platt_target_oracle | platt_source | label_shift | 0.5 | none | logloss | -0.665112 | [-1.06767, -0.347535] |
| platt_target_oracle | platt_source | label_shift | 0.5 | random_q05 | brier | -0.150647 | [-0.20956, -0.100359] |
| platt_target_oracle | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.6744 | [-1.06507, -0.373249] |
| platt_target_oracle | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.15355 | [-0.210723, -0.0998876] |
| platt_target_oracle | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.653295 | [-1.05632, -0.358614] |
| platt_target_shrinkage | platt_source | covariate_shift |  | none | brier | 0.00190856 | [0, 0.00497414] |
| platt_target_shrinkage | platt_source | covariate_shift |  | none | logloss | 0.00680887 | [0, 0.0168725] |
| platt_target_shrinkage | platt_source | covariate_shift |  | random_q05 | brier | 0.000744947 | [0, 0.00215915] |
| platt_target_shrinkage | platt_source | covariate_shift |  | random_q05 | logloss | 0.00273532 | [0, 0.00789028] |
| platt_target_shrinkage | platt_source | covariate_shift |  | smote_q05_k5 | brier | 0.00107381 | [0, 0.00272356] |
| platt_target_shrinkage | platt_source | covariate_shift |  | smote_q05_k5 | logloss | 0.00408215 | [0, 0.00992933] |
| platt_target_shrinkage | platt_source | joint_shift |  | none | brier | -0.000878573 | [-0.00412215, 0.00148643] |
| platt_target_shrinkage | platt_source | joint_shift |  | none | logloss | -0.00240951 | [-0.0117518, 0.00452327] |
| platt_target_shrinkage | platt_source | joint_shift |  | random_q05 | brier | 0.000252646 | [-0.000995662, 0.0017536] |
| platt_target_shrinkage | platt_source | joint_shift |  | random_q05 | logloss | 0.0021832 | [-0.00186633, 0.00841595] |
| platt_target_shrinkage | platt_source | joint_shift |  | smote_q05_k5 | brier | 0.000187825 | [-2.54681e-05, 0.000588943] |
| platt_target_shrinkage | platt_source | joint_shift |  | smote_q05_k5 | logloss | 0.00153953 | [0, 0.00432428] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | none | brier | 0.00134585 | [-0.000977994, 0.00449984] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | none | logloss | 0.0041437 | [-0.00164175, 0.0101635] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | random_q05 | brier | 0.000655118 | [-0.000461234, 0.00167694] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | random_q05 | logloss | 0.00202883 | [-0.00209604, 0.00738298] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | smote_q05_k5 | brier | 0.00121689 | [-0.000675628, 0.00329629] |
| platt_target_shrinkage | platt_source | label_shift | 0.05 | smote_q05_k5 | logloss | 0.0226097 | [-0.00127572, 0.0582657] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | none | brier | -0.0153005 | [-0.0270303, -0.00505199] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | none | logloss | -0.0502881 | [-0.0937767, -0.0104689] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | random_q05 | brier | -0.0119563 | [-0.0192916, -0.004952] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | random_q05 | logloss | -0.0493465 | [-0.0859156, -0.01722] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | smote_q05_k5 | brier | -0.0133476 | [-0.0229154, -0.00479545] |
| platt_target_shrinkage | platt_source | label_shift | 0.2 | smote_q05_k5 | logloss | -0.0513731 | [-0.091263, -0.0165281] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | none | brier | -0.0428487 | [-0.0912426, -0.0081258] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | none | logloss | -0.16948 | [-0.348147, -0.0375818] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | random_q05 | brier | -0.0420542 | [-0.0699958, -0.0152894] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | random_q05 | logloss | -0.173512 | [-0.300547, -0.0548283] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | smote_q05_k5 | brier | -0.0573601 | [-0.0941091, -0.0250096] |
| platt_target_shrinkage | platt_source | label_shift | 0.5 | smote_q05_k5 | logloss | -0.20548 | [-0.349305, -0.0917985] |
| platt_target_shrinkage | platt_source | mixed_shift |  | none | brier | 5.04725e-05 | [-0.00237232, 0.00252374] |
| platt_target_shrinkage | platt_source | mixed_shift |  | none | logloss | 0.000663285 | [-0.00605745, 0.00804731] |
| platt_target_shrinkage | platt_source | mixed_shift |  | random_q05 | brier | 0.000416746 | [-0.00062593, 0.00187617] |
| platt_target_shrinkage | platt_source | mixed_shift |  | random_q05 | logloss | 0.00236724 | [-0.00108637, 0.00818811] |
| platt_target_shrinkage | platt_source | mixed_shift |  | smote_q05_k5 | brier | 0.000483154 | [0, 0.00129482] |
| platt_target_shrinkage | platt_source | mixed_shift |  | smote_q05_k5 | logloss | 0.00238707 | [0, 0.00619263] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | none | brier | -0.00330783 | [-0.00676597, -0.000537962] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | none | logloss | -0.0140233 | [-0.027749, -0.00322385] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | random_q05 | brier | -0.00295123 | [-0.0061664, -0.000588402] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | random_q05 | logloss | -0.0153982 | [-0.0276228, -0.00473217] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | smote_q05_k5 | brier | -0.00255551 | [-0.00573382, -9.59437e-06] |
| platt_target_shrinkage | platt_target_gated | label_shift |  | smote_q05_k5 | logloss | -0.011902 | [-0.024055, -4.21255e-05] |

## Gate behavior

| method | scenario_family | cells | acceptance_rate | residual_auc_mean | residual_auc_median | rejection_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | iid | 36 | 0.7777777777777778 | 0.5511638612439467 | 0.5161584205895176 | 0.2222222222222222 | 0.6 |
| none | mixed_shift | 36 | 0.004629629629629629 | 0.9275211770263394 | 0.972181954547234 | 0.9953703703703703 | 0.6 |
| none | pure_label_shift | 36 | 0.6111111111111112 | 0.5930161366331862 | 0.5664528481496081 | 0.38888888888888884 | 0.6 |
| random_q05 | iid | 36 | 0.7777777777777778 | 0.5511154325826548 | 0.5168936477417335 | 0.2222222222222222 | 0.6 |
| random_q05 | mixed_shift | 36 | 0.004629629629629629 | 0.9292141528292563 | 0.972069292173981 | 0.9953703703703703 | 0.6 |
| random_q05 | pure_label_shift | 36 | 0.6296296296296297 | 0.5928894321516339 | 0.569226141575102 | 0.37037037037037035 | 0.6 |
| smote_q05_k5 | iid | 36 | 0.75 | 0.5518785368280361 | 0.517335369335185 | 0.25 | 0.6 |
| smote_q05_k5 | mixed_shift | 36 | 0.0 | 0.933094140441703 | 0.9703926645027285 | 1.0 | 0.6 |
| smote_q05_k5 | pure_label_shift | 36 | 0.6481481481481481 | 0.5869487018824814 | 0.5614907434489305 | 0.35185185185185186 | 0.6 |
