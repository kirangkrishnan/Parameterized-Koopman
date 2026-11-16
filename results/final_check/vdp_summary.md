# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| vdp | 1500 | 1.52e-03 | 6 | 10% | 0.999 | [0.000, 0.004] | 0.999 |
| vdp | 1500 | 1.52e-03 | 6 | 15% | 1.000 | [0.000, 0.002] | 1.000 |
| vdp | 1500 | 1.52e-03 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| vdp | 1500 | 1.52e-03 | 12 | 10% | 0.773 | [0.206, 0.249] | 0.773 |
| vdp | 1500 | 1.52e-03 | 12 | 15% | 0.999 | [0.000, 0.004] | 0.999 |
| vdp | 1500 | 1.52e-03 | 12 | 20% | 1.000 | [0.000, 0.002] | 1.000 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| vdp | fresh | Ours | 3.57 +- 2.41 | 2.89 | 1.00 | 600 | 0 |
| vdp | fresh | PKNN | 0.63 +- 0.53 | 0.47 | 1.00 | 600 | 0 |
| vdp | fresh | PKNN (re-lift) | 0.29 +- 0.18 | 0.24 | 1.00 | 600 | 0 |
| vdp | fresh | Deep Koopman | 31.31 +- 24.89 | 26.22 | 0.27 | 600 | 0 |
| vdp | fresh | EDMD | 21.38 +- 14.94 | 18.68 | 0.38 | 600 | 0 |
| vdp | fresh | Kernel-EDMD | 23.79 +- 14.72 | 21.04 | 0.33 | 600 | 0 |
