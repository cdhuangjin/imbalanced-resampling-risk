# V5.6 regime-adaptive stability experiment protocol

V5.6 is a locked follow-up to the V5.5 RSC screening result. V5.5 reduced mixed-shift risk but was slightly more conservative than V5.1 under pure label shift. V5.6 therefore uses the residual-AUC threshold that was frozen in V5 (`0.60`) as a regime switch:

- residual AUC `≤ 0.60`: retain the V5.1 continuous shrinkage probability;
- residual AUC `> 0.60`: use the V5.5 RSC probability;
- invalid detector: retain V5.1 shrinkage.

The threshold and switch direction are fixed before inspecting V5.6 held-out metrics. No target labels enter detector fitting, prior stability estimation, policy selection, or probability construction.

## Matrix and comparison

The matrix is identical to V5.5: six Gate tasks, two learners, three training treatments, three seeds, ten shift scenarios, and 10,000 task-cluster bootstrap replicates. The seven postprocessors are source calibration, V4 consensus, V5 binary gate, V5.1 shrinkage, V5.5 RSC, V5.6 RASS, and oracle correction.

The primary comparisons are RASS versus source calibration and RASS versus V5.1 shrinkage on Brier and log-loss. The screening rule requires at least two of the three sampling treatments to satisfy utility at target priors `.20/.50`, practical mixed-shift safety, and mixed-shift non-inferiority to V5.1; otherwise V5.6 is recorded as `NO-GO`.

The policy is an empirical regime-switching policy assembled from established calibration and shift-detection components. It is not claimed as a new estimator or a universal robustness theorem.

After the six-task screening result, the same frozen code and thresholds may be run on the existing 24-task, five-seed library only as a confirmatory extension. The extension cannot change the six-task Gate decision and is reported separately from the prespecified screening result.
