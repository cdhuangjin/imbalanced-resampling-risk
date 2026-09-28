# Gate protocol summary

| Rule | Frozen implementation |
|---|---|
| Candidate signal | At least one prespecified utility or conditional minority-risk signal |
| Ranking endpoint | AUC difference, with task-cluster 95% CI |
| Probability endpoints | Minority logloss, majority logloss and Brier |
| Operational endpoint | `decision_utility_c5 = -(5*FN + FP)/N` with validation-only threshold grid |
| Safety margins | Brier `+0.01`, logloss `+0.1`, utility tolerance `0.02`, plus worst-task checks |
| Inference | 10,000 paired task-cluster bootstrap replicates, seed `20260917` for recent branches |
| Confirmation | 24-task extension only after the relevant Gate is complete and unlocked |
| Reanalysis | Independent analyzer and byte comparison of effects/summary/report |

## Exact branch protocols

- XGB learner: `docs/superpowers/specs/2026-09-17-heldout-xgb-learner-gate.md` and `...heldout-xgb-confirmation.md`.
- Cost weighting: `...cost-sensitive-xgb-gate.md`.
- BRF: `...balanced-rf-gate.md`.
- LightGBM: `...lightgbm-gate.md`.
- Cross-learner mechanism: `...cross-learner-mechanism-analysis.md`.
- Support-filtered SMOTE: `...support-filtered-smote-gate.md`.

The branch files and output manifests, rather than this summary, define exact parameter values.
