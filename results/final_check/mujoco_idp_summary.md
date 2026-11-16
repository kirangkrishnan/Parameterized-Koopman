# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| mujoco_idp | 1500 | 0.00e+00 | 6 | 10% | 0.000 | [0.998, 1.000] | 0.000 |
| mujoco_idp | 1500 | 0.00e+00 | 6 | 15% | 0.963 | [0.028, 0.048] | 0.963 |
| mujoco_idp | 1500 | 0.00e+00 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| mujoco_idp | 1500 | 0.00e+00 | 12 | 10% | 0.000 | [0.998, 1.000] | 0.000 |
| mujoco_idp | 1500 | 0.00e+00 | 12 | 15% | 0.963 | [0.028, 0.048] | 0.963 |
| mujoco_idp | 1500 | 0.00e+00 | 12 | 20% | 1.000 | [0.000, 0.002] | 1.000 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| mujoco_idp | fresh | Ours | 4.32 +- 2.01 | 3.96 | 1.00 | 600 | 0 |
| mujoco_idp | fresh | PKNN | 87.65 +- 35.53 | 77.86 | 0.00 | 600 | 0 |
| mujoco_idp | fresh | PKNN (re-lift) | 59.38 +- 27.57 | 55.30 | 0.00 | 600 | 0 |
| mujoco_idp | fresh | Deep Koopman | 514.57 +- 266.73 | 469.78 | 0.00 | 600 | 0 |
| mujoco_idp | fresh | EDMD | 16.00 +- 10.97 | 13.05 | 0.56 | 600 | 0 |
| mujoco_idp | fresh | Kernel-EDMD | 97.16 +- 44.69 | 87.64 | 0.00 | 600 | 0 |
