# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| mujoco_ip | 1500 | 7.24e-02 | 6 | 10% | 0.694 | [0.283, 0.330] | 0.694 |
| mujoco_ip | 1500 | 7.24e-02 | 6 | 15% | 0.982 | [0.012, 0.026] | 0.982 |
| mujoco_ip | 1500 | 7.24e-02 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| mujoco_ip | 1500 | 7.24e-02 | 12 | 10% | 0.431 | [0.544, 0.595] | 0.431 |
| mujoco_ip | 1500 | 7.24e-02 | 12 | 15% | 0.963 | [0.028, 0.048] | 0.963 |
| mujoco_ip | 1500 | 7.24e-02 | 12 | 20% | 0.999 | [0.000, 0.004] | 0.999 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| mujoco_ip | fresh | Ours | 7.76 +- 3.39 | 7.47 | 0.98 | 600 | 0 |
| mujoco_ip | fresh | PKNN | 8.04 +- 4.13 | 7.23 | 0.95 | 600 | 0 |
| mujoco_ip | fresh | PKNN (re-lift) | 8.26 +- 4.22 | 7.48 | 0.95 | 600 | 0 |
| mujoco_ip | fresh | Deep Koopman | 216.38 +- 103.04 | 192.48 | 0.00 | 600 | 0 |
| mujoco_ip | fresh | EDMD | 58.10 +- 39.95 | 46.72 | 0.04 | 600 | 0 |
| mujoco_ip | fresh | Kernel-EDMD | 61.20 +- 34.15 | 53.12 | 0.01 | 600 | 0 |
