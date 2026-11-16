# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| manipulator | 1500 | 3.50e-05 | 6 | 10% | 0.000 | [0.998, 1.000] | 0.000 |
| manipulator | 1500 | 3.50e-05 | 6 | 15% | 0.000 | [0.998, 1.000] | 0.000 |
| manipulator | 1500 | 3.50e-05 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| manipulator | 1500 | 3.50e-05 | 12 | 10% | 0.000 | [0.998, 1.000] | 0.000 |
| manipulator | 1500 | 3.50e-05 | 12 | 15% | 0.000 | [0.998, 1.000] | 0.000 |
| manipulator | 1500 | 3.50e-05 | 12 | 20% | 1.000 | [0.000, 0.002] | 1.000 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| manipulator | fresh | Ours | 5.61 +- 2.02 | 5.40 | 1.00 | 600 | 0 |
| manipulator | fresh | PKNN | 6.09 +- 3.72 | 5.04 | 0.97 | 600 | 0 |
| manipulator | fresh | PKNN (re-lift) | 4.38 +- 2.52 | 3.76 | 1.00 | 600 | 0 |
| manipulator | fresh | Deep Koopman | 85.97 +- 22.40 | 85.38 | 0.00 | 600 | 0 |
| manipulator | fresh | EDMD | 9.20 +- 5.94 | 8.01 | 0.85 | 600 | 0 |
| manipulator | fresh | Kernel-EDMD | 24.49 +- 13.82 | 20.44 | 0.24 | 600 | 0 |
