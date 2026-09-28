# Compute budget and completed evidence

## Completed

| Branch | Fits | Raw metric rows | Decision |
|---|---:|---:|---|
| V6 XGB Gate | 90 | 720 | GO, conditional |
| V6 XGB confirmation | 360 | 2880 | NO-GO |
| V7 cost-sensitive XGB Gate | 120 | 1080 | NO-GO |
| V8 BRF Gate + comparator | 90 + 90 | 810 + 810 | NO-GO |
| V9 LightGBM Gate + comparator | 90 | 810 | NO-GO |
| V9 cross-learner synthesis | — | 3240 retained | boundary/inconclusive |
| V10 support-filtered SMOTE Gate | 120 | 1080 | GO, conditional |
| V10 support-filtered SMOTE confirmation | 480 | 4320 | NO-GO |
| V11 confidence-guarded selection Gate | 36 | 4320 | NO-GO |
| V12 utility-first selection Gate | 36 | 5040 | NO-GO |

Recent full branches use one CPU thread where specified, deterministic seeds, fixed task lists and 10,000 bootstrap replicates. No new large experiment is unlocked by a failed confirmation. Future candidates must justify additional compute before execution.
