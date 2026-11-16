# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| cartpole | 1500 | 8.00e-05 | 6 | 10% | 0.217 | [0.761, 0.803] | 0.217 |
| cartpole | 1500 | 8.00e-05 | 6 | 15% | 0.998 | [0.000, 0.006] | 0.998 |
| cartpole | 1500 | 8.00e-05 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| cartpole | 1500 | 8.00e-05 | 12 | 10% | 0.000 | [0.998, 1.000] | 0.000 |
| cartpole | 1500 | 8.00e-05 | 12 | 15% | 0.965 | [0.026, 0.045] | 0.965 |
| cartpole | 1500 | 8.00e-05 | 12 | 20% | 1.000 | [0.000, 0.002] | 1.000 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| cartpole | fresh | Ours | 3.89 +- 1.58 | 3.60 | 1.00 | 600 | 0 |
| cartpole | fresh | PKNN | 134.58 +- 126.99 | 95.53 | 0.00 | 600 | 0 |
| cartpole | fresh | PKNN (re-lift) | 37.34 +- 19.99 | 42.82 | 0.17 | 600 | 0 |
| cartpole | fresh | Deep Koopman | 90.60 +- 24.20 | 92.11 | 0.01 | 600 | 0 |
| cartpole | fresh | EDMD | 7.55 +- 4.51 | 6.54 | 0.94 | 600 | 0 |
| cartpole | fresh | Kernel-EDMD | 17.19 +- 11.06 | 14.26 | 0.53 | 600 | 0 |
