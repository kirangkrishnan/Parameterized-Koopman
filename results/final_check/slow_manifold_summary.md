# Experiment summary

## Region check (Phase 3 on fresh parameters)

Survival = fraction of fresh parameters from the hull whose worst-case error over the verification initial conditions is below the tolerance. The interval is the 95% Clopper-Pearson interval for the violation probability under uniform sampling of the hull. 'valid' also requires the true closed loop to pass the system check used when the region was built.

| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |
|---|---|---|---|---|---|---|---|
| slow_manifold | 1500 | 7.25e-03 | 6 | 10% | 1.000 | [0.000, 0.002] | 1.000 |
| slow_manifold | 1500 | 7.25e-03 | 6 | 15% | 1.000 | [0.000, 0.002] | 1.000 |
| slow_manifold | 1500 | 7.25e-03 | 6 | 20% | 1.000 | [0.000, 0.002] | 1.000 |
| slow_manifold | 1500 | 7.25e-03 | 12 | 10% | 0.869 | [0.114, 0.149] | 0.869 |
| slow_manifold | 1500 | 7.25e-03 | 12 | 15% | 0.998 | [0.000, 0.006] | 0.998 |
| slow_manifold | 1500 | 7.25e-03 | 12 | 20% | 1.000 | [0.000, 0.002] | 1.000 |

## Prediction

Worst-case error over 12 steps (%), mean +- std over scored cases. '<15%' is the fraction of cases below the threshold; 'not scored' counts cases whose true trajectory left the bounds.

| system | test parameters | method | mean +- std | median | <15% | scored | not scored |
|---|---|---|---|---|---|---|---|
| slow_manifold | fresh | Ours | 2.38 +- 1.63 | 1.94 | 1.00 | 600 | 0 |
| slow_manifold | fresh | PKNN | 1.11 +- 0.86 | 0.86 | 1.00 | 600 | 0 |
| slow_manifold | fresh | PKNN (re-lift) | 0.59 +- 0.24 | 0.52 | 1.00 | 600 | 0 |
| slow_manifold | fresh | Deep Koopman | 32.30 +- 21.85 | 27.39 | 0.24 | 600 | 0 |
| slow_manifold | fresh | EDMD | 11.29 +- 8.32 | 9.54 | 0.74 | 600 | 0 |
| slow_manifold | fresh | Kernel-EDMD | 18.80 +- 9.36 | 17.19 | 0.42 | 600 | 0 |
