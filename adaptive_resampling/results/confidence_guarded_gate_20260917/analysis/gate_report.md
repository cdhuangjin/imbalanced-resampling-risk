# Adaptive Resampling Gate Report

Decision: **NO-GO**

Completeness: `True`; task clusters with non-baseline selection rate at least 0.5: `1/6`.

## Gate Conditions

| Condition | Result |
|---|---:|
| Adaptive vs fixed SMOTE majority-logloss upper bound <= 0 | `-0.094012`; `True` |
| Adaptive vs fixed SMOTE Brier upper bound <= 0.002 | `-0.00850151`; `True` |
| Adaptive improves minority logloss by >=5% in task clusters | `2/6` |
| Adaptive vs none minority-logloss upper bound < 0 | `0`; `False` |

The selector was frozen before the Gate and uses validation predictions only. A non-GO result does not unlock the 24-task extension and does not justify changing the support threshold, risk tolerances, candidates, or task list.

## Selection Rates

- `abalone_19`: `0.000` non-baseline selections
- `ecoli`: `0.167` non-baseline selections
- `mammography`: `0.500` non-baseline selections
- `oil`: `0.000` non-baseline selections
- `spectrometer`: `0.167` non-baseline selections
- `yeast_me2`: `0.000` non-baseline selections
