# Preregistration and protocol lock

## Lock principle

Each branch is treated as a separately frozen experiment. A failed Gate cannot be repaired by changing the task list, thresholds, candidates, seed set, endpoint or margin after seeing the result. A post-hoc follow-up must be labeled as such and receives a new protocol hash.

## Frozen common rules

- Resampling is training-fold-only.
- Validation is used for early stopping, threshold selection or policy selection; test data are untouched until final evaluation.
- Every fit records learner, task, method, seed, split/data hash and status.
- Primary inference is task-cluster paired bootstrap with 10,000 replicates.
- Preserve task-level failures and do not remove adverse datasets because they lower the mean.
- `GO` requires integrity plus the prespecified signal and safety rules; `NO-GO` is a valid scientific result.

## Retrospective boundary

This document organizes experiments already run. It does not rewrite them as prospective preregistration. The existing protocol files under `docs/superpowers/specs/` remain the authoritative branch-level records.
